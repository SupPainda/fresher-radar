"""An LLM (Claude, Gemini, OpenRouter, Grok, or any OpenAI-compatible API): build a profile from resumes, and
rescore the best offline matches with a reason each.

Which provider is used is worked out from the API key in .env (see resolve()); base URL and model come from
config.json "ranking.llm.providers" and can be overridden with LLM_BASE_URL / LLM_MODEL. Claude goes through
the official Anthropic SDK; the others through the OpenAI SDK, which each of them documents for its
OpenAI-compatible endpoint.

Optional by design: without a key, or on any API problem, rescore() logs one line and the offline ranking
stands; --build-profile reports the problem. Scores are cached in the database per (job, profile).
"""
import json
import logging
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Mapping

import anthropic
import openai
import pandas as pd
import yaml
from pydantic import BaseModel, ValidationError

from jsearch import store
from jsearch.profile import Profile, Resume, profile_hash
from jsearch.ranking import reserves, top_picks
from jsearch.runlog import IST

logger = logging.getLogger("jsearch.llm")

# where a key can come from, in order; the second item is the provider that variable implies
KEY_VARS = [("LLM_API_KEY", None), ("ANTHROPIC_API_KEY", "anthropic"), ("GEMINI_API_KEY", "gemini"),
            ("GOOGLE_API_KEY", "gemini"), ("OPENROUTER_API_KEY", "openrouter"), ("XAI_API_KEY", "xai")]
# how each provider's keys start
KEY_PREFIXES = [("sk-ant-", "anthropic"), ("sk-or-", "openrouter"), ("xai-", "xai"), ("AIza", "gemini"), ("AQ.", "gemini")]
# Anthropic models that support the server-side refusal fallback (see the claude-api skill)
FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5-5"}
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class LLMUnavailable(RuntimeError):
    """The LLM couldn't be used (no key, unknown provider, network, API error, refusal); the message says why."""


class ModelFailed(LLMUnavailable):
    """One model failed (not found, no quota, timeout, server error, unusable answer); the next one may work.
    A plain LLMUnavailable (rejected key, no network) means no other model would do better."""


# --- which provider, key, base URL and model ---------------------------------------------------------------

@dataclass(frozen=True)
class Settings:
    provider: str
    api_key: str | None
    base_url: str | None
    model: str | None
    key_source: str             # the env variable the key came from, for messages
    fallbacks: tuple[str, ...] = ()   # tried in order after model

    def __repr__(self) -> str:  # never print the key
        return (f"Settings({self.provider}, model={self.model}, fallbacks={list(self.fallbacks)}, "
                f"base_url={self.base_url}, key from {self.key_source})")

    @property
    def models(self) -> list[str]:
        return [self.model, *self.fallbacks] if self.model else list(self.fallbacks)


def detect_provider(key: str) -> str | None:
    """The provider a key belongs to, from how it starts; None when it can't be told."""
    return next((p for prefix, p in KEY_PREFIXES if key.startswith(prefix)), None)


def resolve(cfg: dict, env: Mapping[str, str] = os.environ) -> Settings:
    """Provider, key, base URL and model for this run. Raises LLMUnavailable when they can't be settled.

    Key: LLM_API_KEY, else a provider-specific variable (ANTHROPIC_API_KEY, GEMINI_API_KEY, ...).
    Provider: LLM_PROVIDER if set, else from the key's prefix, else from the variable's name, else
    "openai-compatible" when LLM_BASE_URL is set. No key at all: Anthropic (an `ant auth login` profile works).
    Base URL and model: LLM_BASE_URL / LLM_MODEL, else the provider's entry in config.json.
    Fallbacks: the config model (when LLM_MODEL replaced it), then the provider's fallback_models.
    """
    providers = cfg["ranking"]["llm"]["providers"]
    key, source, hint = next(((env[v].strip(), v, p) for v, p in KEY_VARS if env.get(v, "").strip()),
                             (None, "nowhere", None))
    base_url = env.get("LLM_BASE_URL", "").strip() or None
    provider = env.get("LLM_PROVIDER", "").strip().lower() or None
    if provider is None and key:
        provider = detect_provider(key) or hint or ("openai-compatible" if base_url else None)
        if provider is None:
            raise LLMUnavailable(f"can't tell which provider the key in {source} is for; set LLM_PROVIDER to one "
                                 f"of {sorted(providers)} (and LLM_BASE_URL for openai-compatible)")
    provider = provider or "anthropic"
    if provider not in providers:
        raise LLMUnavailable(f"LLM_PROVIDER={provider!r} is not one of {sorted(providers)}")
    entry = providers[provider]
    base_url = base_url or entry["base_url"]
    if provider == "openai-compatible" and not base_url:
        raise LLMUnavailable("an openai-compatible provider needs LLM_BASE_URL")
    if provider != "anthropic" and not key:
        raise LLMUnavailable(f"no API key for {provider}: set LLM_API_KEY in .env")
    model = env.get("LLM_MODEL", "").strip() or entry["model"]
    fallbacks = tuple(dict.fromkeys(m for m in [entry["model"], *entry["fallback_models"]] if m and m != model))
    return Settings(provider, key, base_url, model, source, fallbacks)


# --- backends: one parse() each ----------------------------------------------------------------------------

@dataclass
class Usage:
    input_tokens: int
    output_tokens: int


class ModelChain:
    """parse() tries each model in turn, moving on at once when one fails (ModelFailed): no waiting, no
    backoff. self.model is the model that answered. A plain LLMUnavailable stops the chain."""

    models: list[str]
    model: str
    provider: str

    def parse(self, system: str, user: str, output: type[BaseModel]) -> tuple[BaseModel, Usage]:
        failures: list[str] = []
        for i, model in enumerate(self.models):
            try:
                result = self._parse_one(model, system, user, output)
            except ModelFailed as e:
                if len(self.models) == 1:
                    raise
                failures.append(f"{model}: {e}")
                if i + 1 < len(self.models):
                    logger.info(f"{self.provider} model {model} failed ({e}); trying {self.models[i + 1]}")
                continue
            self.model = model
            return result
        raise LLMUnavailable(f"all {len(self.models)} {self.provider} models failed: " + "; ".join(failures))

    def _parse_one(self, model: str, system: str, user: str, output: type[BaseModel]) -> tuple[BaseModel, Usage]:
        raise NotImplementedError


class AnthropicBackend(ModelChain):
    """Claude through the official Anthropic SDK: structured output, adaptive effort, refusal fallbacks."""

    def __init__(self, client: anthropic.Anthropic, models: list[str], cfg: dict):
        self.client, self.models, self.provider, self.llm = client, models, "anthropic", cfg["ranking"]["llm"]
        self.model = models[0]

    def _parse_one(self, model: str, system: str, user: str, output: type[BaseModel]) -> tuple[BaseModel, Usage]:
        extra = {"betas": [FALLBACK_BETA], "fallbacks": "default"} if model in FALLBACK_MODELS else {}
        try:
            response = self.client.beta.messages.parse(
                model=model, max_tokens=self.llm["max_tokens"], output_config={"effort": self.llm["effort"]},
                system=system, messages=[{"role": "user", "content": user}], output_format=output, **extra)
        except anthropic.AuthenticationError:
            raise LLMUnavailable("the Anthropic API key is missing or invalid") from None
        except anthropic.NotFoundError as e:
            raise ModelFailed(f"model {model!r} not found ({e.message}); set LLM_MODEL") from None
        except anthropic.PermissionDeniedError as e:
            raise ModelFailed(f"the key may not use {model}: {e.message}") from None
        except anthropic.RateLimitError:
            raise ModelFailed("rate limited by the Anthropic API; try again later") from None
        except anthropic.APIStatusError as e:
            raise ModelFailed(f"Anthropic API error {e.status_code}: {e.message}") from None
        except anthropic.APITimeoutError:
            raise ModelFailed(f"no answer within {self.llm['timeout_seconds']}s") from None
        except anthropic.APIConnectionError:
            raise LLMUnavailable("could not reach the Anthropic API (network)") from None
        except anthropic.AnthropicError as e:
            raise LLMUnavailable(str(e)) from None
        if response.stop_reason == "refusal":
            raise ModelFailed("Claude declined the request")
        if response.parsed_output is None:
            raise ModelFailed(f"no structured answer (stop reason: {response.stop_reason})")
        return response.parsed_output, Usage(response.usage.input_tokens, response.usage.output_tokens)


class OpenAICompatBackend(ModelChain):
    """Gemini, OpenRouter, Grok or any OpenAI-compatible API through the OpenAI SDK's chat completions.

    Asks for a strict JSON schema; if the endpoint rejects that, retries once in plain JSON mode with the
    schema in the prompt and validates the answer itself.
    """

    def __init__(self, client: openai.OpenAI, models: list[str], provider: str, cfg: dict):
        self.client, self.models, self.provider, self.llm = client, models, provider, cfg["ranking"]["llm"]
        self.model = models[0]

    def _call(self, fn, model: str, **kwargs):
        try:
            return fn(model=model, max_tokens=self.llm["max_tokens"], **kwargs)
        except openai.AuthenticationError:
            raise LLMUnavailable(f"the {self.provider} API key was rejected") from None
        except openai.NotFoundError:
            raise ModelFailed(f"model {model!r} not found at {self.provider}; "
                              f"set LLM_MODEL to one of: {available_models(self.client)}") from None
        except openai.PermissionDeniedError as e:
            raise ModelFailed(f"the {self.provider} key may not use {model}: {e.message}") from None
        except openai.RateLimitError:
            raise ModelFailed(f"rate limited by {self.provider} (or no quota for {model})") from None
        except openai.LengthFinishReasonError:
            raise ModelFailed("the answer was cut off; raise ranking.llm.max_tokens") from None
        except openai.ContentFilterFinishReasonError:
            raise ModelFailed(f"{self.provider} filtered the answer") from None
        except openai.APITimeoutError:
            raise ModelFailed(f"no answer within {self.llm['timeout_seconds']}s") from None
        except openai.APIConnectionError:
            raise LLMUnavailable(f"could not reach {self.provider} (network)") from None

    def _parse_one(self, model: str, system: str, user: str, output: type[BaseModel]) -> tuple[BaseModel, Usage]:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            completion = self._call(self.client.chat.completions.parse, model, messages=messages,
                                    response_format=output)
            message = completion.choices[0].message
            if message.refusal:
                raise ModelFailed(f"the model declined: {message.refusal[:120]}")
            parsed = message.parsed
        except openai.BadRequestError as e:  # e.g. strict JSON schema unsupported: plain JSON mode instead
            text = e.message.lower()
            if "model" in text and any(w in text for w in ("not found", "invalid", "does not exist", "unknown")):
                raise ModelFailed(f"model {model!r} not accepted by {self.provider} ({e.message[:120]}); "
                                  f"set LLM_MODEL to one of: {available_models(self.client)}") from None
            logger.info(f"{self.provider} rejected the strict schema ({e.message[:120]}); retrying in JSON mode")
            schema = json.dumps(output.model_json_schema())
            messages[0]["content"] += f"\n\nReply with only a JSON object matching this JSON schema:\n{schema}"
            try:
                completion = self._call(self.client.chat.completions.create, model, messages=messages,
                                        response_format={"type": "json_object"})
                parsed = output.model_validate_json(completion.choices[0].message.content or "")
            except openai.APIStatusError as e2:
                raise ModelFailed(f"{self.provider} API error {e2.status_code}: {e2.message}") from None
            except ValidationError as e2:
                raise ModelFailed(f"the answer didn't match the expected format ({e2.error_count()} problems)") \
                    from None
        except ValidationError as e:  # the strict answer wasn't valid JSON for the schema (e.g. wrapped in ```)
            raise ModelFailed(f"the answer didn't match the expected format ({e.error_count()} problems)") from None
        except openai.APIStatusError as e:
            raise ModelFailed(f"{self.provider} API error {e.status_code}: {e.message}") from None
        except openai.OpenAIError as e:
            raise LLMUnavailable(str(e)) from None
        if parsed is None:
            raise ModelFailed("no structured answer")
        usage = completion.usage
        return parsed, Usage(usage.prompt_tokens if usage else 0, usage.completion_tokens if usage else 0)


Backend = AnthropicBackend | OpenAICompatBackend


def available_models(client: openai.OpenAI, limit: int = 15) -> str:
    """Model ids the key can list, for error messages; a note when even listing fails."""
    try:
        ids = [m.id for m in client.models.list()]
    except openai.OpenAIError:
        return "(could not list models)"
    return ", ".join(ids[:limit]) + (" ..." if len(ids) > limit else "") if ids else "(none listed)"


def make_backend(cfg: dict, env: Mapping[str, str] = os.environ) -> Backend:
    """The backend for whichever key is configured. Raises LLMUnavailable when there's none to use."""
    s = resolve(cfg, env)
    # no SDK retries with exponential backoff: a failed model hands over to the next one in the chain
    limits = {"max_retries": 0, "timeout": cfg["ranking"]["llm"]["timeout_seconds"]}
    if s.provider == "anthropic":
        client = anthropic.Anthropic(**({"api_key": s.api_key} if s.api_key else {}),
                                     **({"base_url": s.base_url} if s.base_url else {}), **limits)
        # without credentials the SDK fails only while building the first request (TypeError); check the
        # same three sources it checks (_client.py _validate_headers) to say so up front
        if not (client.api_key or client.auth_token or getattr(client, "_token_cache", None) is not None):
            raise LLMUnavailable("no API key: put LLM_API_KEY=... in .env (see .env.example)")
        return AnthropicBackend(client, s.models, cfg)
    client = openai.OpenAI(api_key=s.api_key, base_url=s.base_url, **limits)
    models = s.models
    if not models:  # nothing configured: the first model the key can use
        try:
            models = [next(iter(client.models.list())).id]
        except (openai.OpenAIError, StopIteration):
            raise LLMUnavailable(f"no model set for {s.provider}, and none could be listed; set LLM_MODEL") from None
        logger.info(f"no model set for {s.provider}; using {models[0]}, the first one the key lists (set LLM_MODEL)")
    return OpenAICompatBackend(client, models, s.provider, cfg)


def _cost(usage: Usage, backend: Backend, cfg: dict) -> str:
    price = cfg["ranking"]["llm"]["providers"][backend.provider]["price_per_mtok"]
    text = f"{usage.input_tokens} in / {usage.output_tokens} out tokens"
    if price:
        text += f", about ${(usage.input_tokens * price[0] + usage.output_tokens * price[1]) / 1_000_000:.2f}"
    return text


# --- profile from resumes ----------------------------------------------------------------------------------

class DraftSkill(BaseModel):
    name: str
    weight: int


class DraftAlias(BaseModel):
    skill: str
    spellings: list[str]


class DraftResume(BaseModel):
    file: str
    label: str
    focus: list[str]


class ProfileDraft(BaseModel):
    name: str
    experience_years: float
    summary: str
    target_roles: list[str]
    skills: list[DraftSkill]
    aliases: list[DraftAlias]
    resumes: list[DraftResume]


BUILD_SYSTEM = """You turn one person's resumes into a profile used to match them against job postings.
The resumes may be role-specific variants of each other; combine everything they show into one profile.

- name: the person's name.
- experience_years: full-time professional experience in years; internships and student roles count as 0.
- summary: 3-4 factual sentences: education, strongest experience, what they build, what roles they target.
- target_roles: 5-12 job titles these resumes aim at, written as they appear in job postings.
- skills: 15-40 concrete technologies, methods and domains, lowercase, worded as job postings word them
  (e.g. "python", "rag", "fastapi", "product management"). weight 3 = used substantially in real work,
  2 = solid (projects or some work), 1 = familiar.
- aliases: other spellings postings use for a listed skill (e.g. rag -> "retrieval augmented generation").
  Only for skills in the skills list; skip skills with no common alternative.
- resumes: one entry per file given, with its exact file name; a short unique label (e.g. "GenAI", "PM",
  "DA"); focus = 4-8 lowercase terms that set that variant apart. Near-identical copies still get their own
  entry with a distinct label (e.g. "DA 2")."""


def build_profile(texts: dict[str, str], cfg: dict, backend: Backend) -> Profile:
    """One LLM call over all resume texts -> a validated Profile. Raises LLMUnavailable."""
    user = "\n\n".join(f'<resume file="{name}">\n{text}\n</resume>' for name, text in texts.items())
    draft, usage = backend.parse(BUILD_SYSTEM, user, ProfileDraft)
    logger.info(f"profile built with {backend.model}: {_cost(usage, backend, cfg)}")
    skills = {s.name.strip().lower(): min(3, max(1, s.weight)) for s in draft.skills if s.name.strip()}
    aliases = {a.skill.strip().lower(): [x for x in a.spellings if x.strip()] for a in draft.aliases}
    labels: set[str] = set()
    resumes = []
    for r in draft.resumes:
        label = r.label.strip() or r.file
        while label in labels:
            label += "+"
        labels.add(label)
        resumes.append(Resume(label=label, file=r.file, focus=[f.lower() for f in r.focus]))
    return Profile(name=draft.name, experience_years=max(0.0, draft.experience_years), summary=draft.summary,
                   target_roles=draft.target_roles, skills=skills,
                   aliases={k: v for k, v in aliases.items() if k in skills and v}, resumes=resumes)


# --- rescoring the top offline matches ---------------------------------------------------------------------

class JobScore(BaseModel):
    job_id: int
    score: int
    reason: str
    best_resume: str


class Rescore(BaseModel):
    scores: list[JobScore]


RESCORE_SYSTEM = """You judge how well job postings fit one candidate, described by their profile.

For every job id given, return:
- score 0-100: 85+ a strong fit they should apply to now; 70-84 good; 50-69 partial; below 50 poor.
  Weigh what matters most: whether the role is one they target, whether the experience asked for fits
  theirs (a posting needing far more years than they have scores low even when the skills match), overlap
  with their strongest skills, and the domain.
- reason: at most 25 words, specific, naming the deciding factor.
- best_resume: the label of the resume variant they should send, exactly as listed in the profile."""


def _job_block(job_id: int, row: pd.Series) -> str:
    fields = [("title", row.get("title")), ("company", row.get("company")), ("location", row.get("location")),
              ("skill tags", row.get("skills")), ("experience asked (years)", row.get("min_years")),
              ("description", row.get("description"))]
    lines = [f"{k}: {v}" for k, v in fields if pd.notna(v) and str(v).strip()]
    return f'<job id="{job_id}">\n' + "\n".join(lines) + "\n</job>"


def rescore(df: pd.DataFrame, profile: Profile, cfg: dict, con: sqlite3.Connection,
            backend_factory=make_backend) -> pd.DataFrame:
    """LLM scores for the top_n offline matches, resume reserves included (cached ones reused); others keep the
    offline score.

    scored_by becomes the model id for jobs with an LLM score.
    """
    llm = cfg["ranking"]["llm"]
    phash = profile_hash(profile)
    df = df.assign(_key=store.job_key(df["title"], df["company"]))
    cached = store.llm_scores(con, phash)
    top = top_picks(df, llm["top_n"], reserves(profile)) if llm["enabled"] and llm["top_n"] else df.iloc[0:0]
    todo = top[~top["_key"].isin(cached)]

    if len(todo):
        labels = [r.label for r in profile.resumes]
        profile_yaml = yaml.safe_dump(profile.model_dump(exclude={"aliases"}), sort_keys=False, allow_unicode=True)
        jobs = "\n\n".join(_job_block(i, row) for i, (_, row) in enumerate(todo.iterrows(), 1))
        user = f"<profile>\n{profile_yaml}</profile>\n\n{jobs}"
        try:
            backend = backend_factory(cfg)
            result, usage = backend.parse(RESCORE_SYSTEM, user, Rescore)
        except LLMUnavailable as e:
            logger.info(f"LLM rescoring skipped ({e}); offline ranking kept for {len(todo)} jobs")
        else:
            keys = todo["_key"].tolist()
            fresh = {}
            for s in result.scores:
                if not 1 <= s.job_id <= len(keys):
                    continue  # an id we never sent
                send = s.best_resume if s.best_resume in labels else todo.iloc[s.job_id - 1]["send"]
                fresh[keys[s.job_id - 1]] = (min(100, max(0, s.score)), s.reason.strip(), send, backend.model)
            store.save_llm_scores(con, phash, datetime.now(IST).isoformat(timespec="seconds"), fresh)
            cached.update(fresh)
            missing = len(keys) - len(fresh)
            logger.info(f"{backend.provider} rescored {len(fresh)} jobs with {backend.model}: "
                        f"{_cost(usage, backend, cfg)}"
                        + (f"; {missing} got no score and keep the offline one" if missing else ""))
    elif len(top):
        logger.info(f"LLM scores reused from the cache for all top {len(top)} jobs (no API call)")

    hit = df["_key"].isin(cached)
    if hit.any():
        got = df.loc[hit, "_key"].map(cached)
        for i, col in enumerate(("fit", "why", "send", "scored_by")):
            df.loc[hit, col] = got.map(lambda t, i=i: t[i])
    return df.drop(columns="_key")
