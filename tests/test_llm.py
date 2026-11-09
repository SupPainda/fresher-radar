"""The LLM layer: provider detection, backends (Anthropic, OpenAI-compatible), profile building and top-N
rescoring. Every client here is a fake; tests never call an API."""
import logging
from types import SimpleNamespace

import anthropic
import httpx2
import openai
import pandas as pd
import pytest

import fetch_jobs
from jsearch import llm, store
from jsearch.profile import Profile, Resume, load_profile
from jsearch.ranking import score
from tests.test_profile import tiny_docx

NO_ENV: dict[str, str] = {}
# conftest replaces llm.make_backend so no test can reach an API; these tests only construct clients
# (no request is ever sent), so they use the real one, captured at import
REAL_MAKE_BACKEND = llm.make_backend


def api_error(cls, status: int, message: str):
    """An SDK status error as the SDK would raise it (both SDKs build theirs on httpx2)."""
    response = httpx2.Response(status, request=httpx2.Request("POST", "https://api.example"))
    return cls(message, response=response, body=None)


def usage_ns(i=40_000, o=2_000):
    return SimpleNamespace(input_tokens=i, output_tokens=o, prompt_tokens=i, completion_tokens=o)


class FakeAnthropicClient:
    """beta.messages.parse records its kwargs and answers with a scripted parsed_output (or raises)."""

    def __init__(self, answer=None, stop_reason="end_turn", error: Exception | None = None,
                 errors_by_model: dict | None = None):
        self.calls: list[dict] = []
        self.answer, self.stop_reason, self.error = answer, stop_reason, error
        self.errors_by_model = errors_by_model or {}
        self.beta = SimpleNamespace(messages=SimpleNamespace(parse=self.parse))

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        error = self.errors_by_model.get(kwargs["model"], self.error)
        if error:
            raise error
        return SimpleNamespace(stop_reason=self.stop_reason, parsed_output=self.answer, usage=usage_ns())


class FakeOpenAIClient:
    """chat.completions.parse/create and models.list, scripted per call (a dict scripts each model)."""

    def __init__(self, parse=None, create=None, models=("model-a", "model-b")):
        self.calls: list[tuple[str, dict]] = []
        self._parse, self._create, self._models = parse, create, models
        self.chat = SimpleNamespace(completions=SimpleNamespace(parse=self.parse, create=self.create))
        self.models = SimpleNamespace(list=lambda: [SimpleNamespace(id=m) for m in self._models])

    @staticmethod
    def _answer(outcome, model):
        if isinstance(outcome, dict):
            outcome = outcome[model]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def parse(self, **kwargs):
        self.calls.append(("parse", kwargs))
        return self._answer(self._parse, kwargs["model"])

    def create(self, **kwargs):
        self.calls.append(("create", kwargs))
        return self._answer(self._create, kwargs["model"])


def completion(parsed=None, content=None, refusal=None):
    message = SimpleNamespace(parsed=parsed, content=content, refusal=refusal)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage_ns())


class FakeBackend:
    """A backend whose parse() answers from a script; records (system, user, output) per call."""

    def __init__(self, answer=None, error: Exception | None = None, model="fake-model", provider="anthropic"):
        self.answer, self.error, self.model, self.provider = answer, error, model, provider
        self.calls: list[tuple] = []

    def parse(self, system, user, output):
        self.calls.append((system, user, output))
        if self.error:
            raise self.error
        return self.answer, llm.Usage(40_000, 2_000)


def never(cfg):
    raise AssertionError("no backend should be needed here")


@pytest.fixture
def profile() -> Profile:
    return Profile(name="T", experience_years=0, target_roles=["AI Engineer"], skills={"python": 3, "rag": 3},
                   resumes=[Resume(label="GenAI", focus=["rag"]), Resume(label="PM", focus=["product"])])


@pytest.fixture
def ranked(cfg, profile) -> pd.DataFrame:
    """Three jobs scored offline: AI Engineer (best), Python Developer, Product Manager (worst)."""
    df = pd.DataFrame({"title": ["AI Engineer", "Python Developer", "Product Manager"],
                       "company": ["A", "B", "C"], "description": ["python rag", "python", "roadmap"]})
    return score(df, profile, cfg)


@pytest.fixture
def con(tmp_path):
    c = store.connect(tmp_path / "jobs.db")
    yield c
    c.close()


def scores(*items) -> llm.Rescore:
    return llm.Rescore(scores=[llm.JobScore(job_id=i, score=s, reason=r, best_resume=b) for i, s, r, b in items])


# --- resolve(): which provider, key, base URL, model --------------------------------------------------------

@pytest.mark.parametrize("key, provider", [
    ("sk-ant-api03-xyz", "anthropic"), ("sk-or-v1-abc", "openrouter"), ("xai-abc", "xai"), ("AIzaSyAbc", "gemini"),
    ("AQ.Ab8RN6Xyz", "gemini"),
])
def test_provider_is_detected_from_the_key(cfg, key, provider):
    s = llm.resolve(cfg, {"LLM_API_KEY": key})
    entry = cfg["ranking"]["llm"]["providers"][provider]
    assert (s.provider, s.api_key, s.base_url, s.model) == (provider, key, entry["base_url"], entry["model"])
    assert s.key_source == "LLM_API_KEY"


def test_provider_specific_variable_names_the_provider(cfg):
    s = llm.resolve(cfg, {"GEMINI_API_KEY": "some-key-without-a-known-prefix"})
    assert s.provider == "gemini" and s.key_source == "GEMINI_API_KEY"


def test_llm_api_key_wins_over_provider_variables(cfg):
    assert llm.resolve(cfg, {"LLM_API_KEY": "xai-1", "ANTHROPIC_API_KEY": "sk-ant-2"}).provider == "xai"


def test_overrides_for_provider_base_url_and_model(cfg):
    s = llm.resolve(cfg, {"LLM_API_KEY": "AIzaX", "LLM_PROVIDER": "OpenRouter".lower(),
                          "LLM_BASE_URL": "https://proxy.example/v1", "LLM_MODEL": "my/model"})
    assert (s.provider, s.base_url, s.model) == ("openrouter", "https://proxy.example/v1", "my/model")


def test_unknown_key_with_a_base_url_is_openai_compatible(cfg):
    s = llm.resolve(cfg, {"LLM_API_KEY": "sk-local", "LLM_BASE_URL": "http://localhost:11434/v1", "LLM_MODEL": "m"})
    assert s.provider == "openai-compatible" and s.base_url == "http://localhost:11434/v1"


@pytest.mark.parametrize("env, message", [
    ({"LLM_API_KEY": "sk-mystery"}, "can't tell which provider"),
    ({"LLM_API_KEY": "k", "LLM_PROVIDER": "openai-compatible"}, "needs LLM_BASE_URL"),
    ({"LLM_PROVIDER": "gemini"}, "no API key for gemini"),
    ({"LLM_API_KEY": "k", "LLM_PROVIDER": "bard"}, "not one of"),
])
def test_unresolvable_settings_say_why(cfg, env, message):
    with pytest.raises(llm.LLMUnavailable, match=message):
        llm.resolve(cfg, env)


def test_no_key_at_all_means_anthropic_profile_login(cfg):
    s = llm.resolve(cfg, NO_ENV)
    assert s.provider == "anthropic" and s.api_key is None and s.model == "claude-opus-5-5"


def test_config_fallback_models_follow_the_model(cfg):
    entry = cfg["ranking"]["llm"]["providers"]["gemini"]
    entry["fallback_models"] = ["g-2", "g-3"]
    s = llm.resolve(cfg, {"LLM_API_KEY": "AIzaX"})
    assert s.models == [entry["model"], "g-2", "g-3"]


def test_llm_model_goes_first_and_the_config_model_becomes_a_fallback(cfg):
    entry = cfg["ranking"]["llm"]["providers"]["gemini"]
    entry["model"], entry["fallback_models"] = "g-1", ["g-2", "g-3"]
    assert llm.resolve(cfg, {"LLM_API_KEY": "AIzaX", "LLM_MODEL": "g-9"}).models == ["g-9", "g-1", "g-2", "g-3"]
    # a model already in the chain isn't tried twice
    assert llm.resolve(cfg, {"LLM_API_KEY": "AIzaX", "LLM_MODEL": "g-2"}).models == ["g-2", "g-1", "g-3"]


def test_no_model_anywhere_gives_an_empty_chain(cfg):
    assert llm.resolve(cfg, {"LLM_API_KEY": "xai-1"}).models == []


def test_settings_never_show_the_key(cfg):
    assert "sk-ant-secret" not in repr(llm.resolve(cfg, {"LLM_API_KEY": "sk-ant-secret"}))


# --- make_backend() ----------------------------------------------------------------------------------------

def test_anthropic_without_credentials_is_unavailable(cfg, monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(llm.anthropic.Anthropic, "_token_cache", None, raising=False)
    with pytest.raises(llm.LLMUnavailable, match="no API key"):
        REAL_MAKE_BACKEND(cfg, NO_ENV)


def test_anthropic_key_gives_the_anthropic_backend(cfg):
    b = REAL_MAKE_BACKEND(cfg, {"LLM_API_KEY": "sk-ant-test"})
    assert isinstance(b, llm.AnthropicBackend) and b.model == "claude-opus-5-5" and b.client.api_key == "sk-ant-test"


def test_gemini_key_gives_an_openai_compatible_backend_on_its_endpoint(cfg):
    b = REAL_MAKE_BACKEND(cfg, {"LLM_API_KEY": "AIzaTest"})
    assert isinstance(b, llm.OpenAICompatBackend) and b.provider == "gemini"
    assert str(b.client.base_url).startswith("https://generativelanguage.googleapis.com/v1beta/openai")


@pytest.mark.parametrize("key", ["AIzaTest", "sk-ant-test"])
def test_clients_never_retry_with_backoff_and_time_out_per_config(cfg, key):
    cfg["ranking"]["llm"]["timeout_seconds"] = 42
    b = REAL_MAKE_BACKEND(cfg, {"LLM_API_KEY": key})
    assert b.client.max_retries == 0 and b.client.timeout == 42


def test_backend_gets_the_whole_model_chain(cfg):
    entry = cfg["ranking"]["llm"]["providers"]["gemini"]
    b = REAL_MAKE_BACKEND(cfg, {"LLM_API_KEY": "AIzaTest", "LLM_MODEL": "gemini-x"})
    assert b.models == ["gemini-x", entry["model"], *entry["fallback_models"]] and b.model == "gemini-x"


def test_missing_model_uses_the_first_listed(cfg, monkeypatch, caplog):
    monkeypatch.setattr(llm.openai, "OpenAI", lambda **kw: FakeOpenAIClient(models=("grok-x", "grok-y")))
    with caplog.at_level(logging.INFO, logger="jsearch"):
        b = REAL_MAKE_BACKEND(cfg, {"LLM_API_KEY": "xai-test"})
    assert b.model == "grok-x" and "first one the key lists" in caplog.text


def test_missing_model_and_nothing_listed_is_unavailable(cfg, monkeypatch):
    monkeypatch.setattr(llm.openai, "OpenAI", lambda **kw: FakeOpenAIClient(models=()))
    with pytest.raises(llm.LLMUnavailable, match="set LLM_MODEL"):
        REAL_MAKE_BACKEND(cfg, {"LLM_API_KEY": "xai-test"})


# --- AnthropicBackend --------------------------------------------------------------------------------------

def test_anthropic_request_follows_config(cfg):
    client = FakeAnthropicClient(scores())
    llm.AnthropicBackend(client, ["claude-opus-5-5"], cfg).parse("sys", "user", llm.Rescore)
    call, conf = client.calls[0], cfg["ranking"]["llm"]
    assert call["model"] == "claude-opus-5-5" and call["output_config"] == {"effort": conf["effort"]}
    assert call["output_format"] is llm.Rescore and call["system"] == "sys"
    assert call["fallbacks"] == "default" and call["betas"] == [llm.FALLBACK_BETA]


def test_anthropic_models_without_fallback_support_send_none(cfg):
    client = FakeAnthropicClient(scores())
    llm.AnthropicBackend(client, ["claude-haiku-5-5"], cfg).parse("s", "u", llm.Rescore)
    assert "fallbacks" not in client.calls[0]


@pytest.mark.parametrize("client, message", [
    (FakeAnthropicClient(error=api_error(anthropic.AuthenticationError, 401, "bad key")), "missing or invalid"),
    (FakeAnthropicClient(error=api_error(anthropic.NotFoundError, 404, "no model")), "set LLM_MODEL"),
    (FakeAnthropicClient(answer=None, stop_reason="refusal"), "declined"),
    (FakeAnthropicClient(answer=None, stop_reason="max_tokens"), "no structured answer"),
])
def test_anthropic_problems_become_llm_unavailable(cfg, client, message):
    with pytest.raises(llm.LLMUnavailable, match=message):
        llm.AnthropicBackend(client, ["claude-opus-5-5"], cfg).parse("s", "u", llm.Rescore)


# --- OpenAICompatBackend -----------------------------------------------------------------------------------

def compat(cfg, client) -> llm.OpenAICompatBackend:
    return llm.OpenAICompatBackend(client, ["gemini-3.5-flash"], "gemini", cfg)


def test_strict_schema_parse(cfg):
    client = FakeOpenAIClient(parse=completion(parsed=scores((1, 80, "ok", "GenAI"))))
    parsed, usage = compat(cfg, client).parse("sys", "user", llm.Rescore)
    assert parsed.scores[0].score == 80 and usage == llm.Usage(40_000, 2_000)
    kind, kwargs = client.calls[0]
    assert kind == "parse" and kwargs["response_format"] is llm.Rescore and kwargs["model"] == "gemini-3.5-flash"
    assert kwargs["messages"][0] == {"role": "system", "content": "sys"}


def test_rejected_schema_falls_back_to_json_mode(cfg):
    good = scores((1, 70, "fine", "PM")).model_dump_json()
    client = FakeOpenAIClient(parse=api_error(openai.BadRequestError, 400, "response_format json_schema unsupported"),
                              create=completion(content=good))
    parsed, _ = compat(cfg, client).parse("sys", "user", llm.Rescore)
    assert parsed.scores[0].best_resume == "PM"
    kind, kwargs = client.calls[1]
    assert kind == "create" and kwargs["response_format"] == {"type": "json_object"}
    assert "JSON schema" in kwargs["messages"][0]["content"]


def test_json_mode_answer_that_does_not_fit_is_unavailable(cfg):
    client = FakeOpenAIClient(parse=api_error(openai.BadRequestError, 400, "schema not supported"),
                              create=completion(content='{"scores": "nope"}'))
    with pytest.raises(llm.LLMUnavailable, match="didn't match the expected format"):
        compat(cfg, client).parse("s", "u", llm.Rescore)


@pytest.mark.parametrize("error, message", [
    (api_error(openai.NotFoundError, 404, "not here"), "set LLM_MODEL to one of: model-a, model-b"),
    (api_error(openai.BadRequestError, 400, "Model gemini-9 not found"), "set LLM_MODEL to one of: model-a"),
    (api_error(openai.AuthenticationError, 401, "bad"), "gemini API key was rejected"),
    (api_error(openai.RateLimitError, 429, "slow down"), "rate limited by gemini"),
    (api_error(openai.InternalServerError, 500, "boom"), "gemini API error 500"),
])
def test_openai_compatible_problems_become_llm_unavailable(cfg, error, message):
    with pytest.raises(llm.LLMUnavailable, match=message):
        compat(cfg, FakeOpenAIClient(parse=error)).parse("s", "u", llm.Rescore)


def test_openai_compatible_refusal(cfg):
    client = FakeOpenAIClient(parse=completion(refusal="I can't help with that"))
    with pytest.raises(llm.LLMUnavailable, match="declined"):
        compat(cfg, client).parse("s", "u", llm.Rescore)


def test_strict_answer_that_is_not_valid_json_is_a_model_failure(cfg):
    """e.g. gemma wraps its JSON in ``` fences; the SDK's parse raises pydantic's ValidationError."""
    try:
        llm.Rescore.model_validate_json('{"scores": []}\n```')
    except llm.ValidationError as e:
        fenced = e
    with pytest.raises(llm.ModelFailed, match="didn't match the expected format"):
        compat(cfg, FakeOpenAIClient(parse=fenced)).parse("s", "u", llm.Rescore)


# --- the model chain: a failed model hands over to the next at once ----------------------------------------

def timeout_error(cls):
    return cls(request=httpx2.Request("POST", "https://api.example"))


def chain(cfg, client, models=("m-1", "m-2", "m-3")) -> llm.OpenAICompatBackend:
    return llm.OpenAICompatBackend(client, list(models), "gemini", cfg)


@pytest.mark.parametrize("failure", [
    api_error(openai.NotFoundError, 404, "no such model"),
    api_error(openai.RateLimitError, 429, "quota"),
    api_error(openai.InternalServerError, 503, "overloaded"),
    api_error(openai.BadRequestError, 400, "Model m-1 is invalid"),
    timeout_error(openai.APITimeoutError),
    completion(refusal="no"),
    completion(parsed=None),
])
def test_a_failed_model_hands_over_to_the_next(cfg, caplog, failure):
    good = completion(parsed=scores((1, 80, "ok", "GenAI")))
    client = FakeOpenAIClient(parse={"m-1": failure, "m-2": good, "m-3": good})
    backend = chain(cfg, client)
    with caplog.at_level(logging.INFO, logger="jsearch"):
        parsed, _ = backend.parse("s", "u", llm.Rescore)
    assert parsed.scores[0].score == 80 and backend.model == "m-2"
    assert [kw["model"] for kind, kw in client.calls if kind == "parse"] == ["m-1", "m-2"]
    assert "model m-1 failed" in caplog.text and "trying m-2" in caplog.text


def test_every_model_failing_says_what_happened_to_each(cfg):
    client = FakeOpenAIClient(parse={"m-1": api_error(openai.RateLimitError, 429, "quota"),
                                     "m-2": timeout_error(openai.APITimeoutError)})
    with pytest.raises(llm.LLMUnavailable, match=r"all 2 gemini models failed: m-1: rate limited.*m-2: no answer"):
        chain(cfg, client, ("m-1", "m-2")).parse("s", "u", llm.Rescore)


@pytest.mark.parametrize("error, message", [
    (api_error(openai.AuthenticationError, 401, "bad"), "API key was rejected"),
    (openai.APIConnectionError(request=httpx2.Request("POST", "https://api.example")), "network"),
])
def test_key_or_network_problems_stop_the_chain(cfg, error, message):
    client = FakeOpenAIClient(parse=error)
    with pytest.raises(llm.LLMUnavailable, match=message):
        chain(cfg, client).parse("s", "u", llm.Rescore)
    assert len(client.calls) == 1  # no other model tried: none would do better


def test_anthropic_chain_moves_on_from_an_overloaded_model(cfg):
    client = FakeAnthropicClient(scores((1, 75, "ok", "PM")),
                                 errors_by_model={"claude-opus-5-5": api_error(anthropic.InternalServerError, 529,
                                                                               "overloaded")})
    backend = llm.AnthropicBackend(client, ["claude-opus-5-5", "claude-sonnet-5-5"], cfg)
    parsed, _ = backend.parse("s", "u", llm.Rescore)
    assert parsed.scores[0].best_resume == "PM" and backend.model == "claude-sonnet-5-5"
    assert [c["model"] for c in client.calls] == ["claude-opus-5-5", "claude-sonnet-5-5"]


def test_anthropic_timeout_is_a_model_failure(cfg):
    client = FakeAnthropicClient(error=timeout_error(anthropic.APITimeoutError))
    with pytest.raises(llm.ModelFailed, match="no answer within"):
        llm.AnthropicBackend(client, ["claude-opus-5-5"], cfg).parse("s", "u", llm.Rescore)


# --- rescore -----------------------------------------------------------------------------------------------

def test_top_n_jobs_are_sent_in_one_request_and_scored(cfg, profile, ranked, con):
    cfg["ranking"]["llm"]["top_n"] = 2
    backend = FakeBackend(scores((1, 91, "Core RAG role", "GenAI"), (2, 60, "Generic dev", "GenAI")),
                          model="gemini-3.5-flash", provider="gemini")
    out = llm.rescore(ranked, profile, cfg, con, backend_factory=lambda c: backend)
    assert len(backend.calls) == 1
    user = backend.calls[0][1]
    assert '<job id="1">' in user and '<job id="2">' in user and '<job id="3">' not in user
    assert "title: AI Engineer" in user and "<profile>" in user
    assert out["scored_by"].tolist() == ["gemini-3.5-flash", "gemini-3.5-flash", "offline"]
    assert out.loc[0, ["fit", "why", "send"]].tolist() == [91, "Core RAG role", "GenAI"]
    assert len(store.llm_scores(con, llm.profile_hash(profile))) == 2


def test_cached_scores_mean_no_second_call(cfg, profile, ranked, con):
    llm.rescore(ranked, profile, cfg, con, backend_factory=lambda c: FakeBackend(
        scores((1, 90, "a", "GenAI"), (2, 70, "b", "GenAI"), (3, 20, "c", "PM")), model="m1"))
    out = llm.rescore(ranked, profile, cfg, con, backend_factory=never)
    assert out["fit"].tolist() == [90, 70, 20] and set(out["scored_by"]) == {"m1"}


def test_a_changed_profile_is_rescored(cfg, profile, ranked, con):
    llm.rescore(ranked, profile, cfg, con, backend_factory=lambda c: FakeBackend(scores((1, 90, "a", "GenAI"))))
    backend = FakeBackend(scores())
    llm.rescore(ranked, profile.model_copy(update={"experience_years": 1}), cfg, con, backend_factory=lambda c: backend)
    assert len(backend.calls) == 1


def test_bad_answers_are_contained(cfg, profile, ranked, con):
    backend = FakeBackend(scores((1, 140, "x", "Mystery"), (2, -5, "y", "PM"), (9, 50, "z", "PM")))
    out = llm.rescore(ranked, profile, cfg, con, backend_factory=lambda c: backend)
    assert out.loc[0, "fit"] == 100 and out.loc[0, "send"] == ranked.loc[0, "send"]   # clamped; label kept
    assert out.loc[1, "fit"] == 0 and out.loc[1, "send"] == "PM"
    assert out.loc[2, "scored_by"] == "offline"                                        # id 9 was never sent


@pytest.mark.parametrize("factory", [
    lambda c: (_ for _ in ()).throw(llm.LLMUnavailable("no API key")),     # no backend at all
    lambda c: FakeBackend(error=llm.LLMUnavailable("rate limited")),       # backend fails mid-call
])
def test_llm_problems_keep_the_offline_ranking(cfg, profile, ranked, con, caplog, factory):
    with caplog.at_level(logging.INFO, logger="jsearch"):
        out = llm.rescore(ranked, profile, cfg, con, backend_factory=factory)
    pd.testing.assert_frame_equal(out, ranked)
    assert "LLM rescoring skipped" in caplog.text
    assert store.llm_scores(con, llm.profile_hash(profile)) == {}


def test_disabled_or_zero_top_n_makes_no_call(cfg, profile, ranked, con):
    cfg["ranking"]["llm"]["enabled"] = False
    llm.rescore(ranked, profile, cfg, con, backend_factory=never)
    cfg["ranking"]["llm"].update(enabled=True, top_n=0)
    llm.rescore(ranked, profile, cfg, con, backend_factory=never)


def test_cost_is_shown_only_when_a_price_is_configured(cfg):
    usage = llm.Usage(1_000_000, 100_000)
    assert llm._cost(usage, FakeBackend(provider="anthropic"), cfg).endswith("about $6.00")   # 4 + 0.1 x 20
    assert "$" not in llm._cost(usage, FakeBackend(provider="gemini"), cfg)


# --- build_profile -----------------------------------------------------------------------------------------

def draft(**kw) -> llm.ProfileDraft:
    base = dict(name="Asha", experience_years=-1, summary="s", target_roles=["AI Engineer"],
                skills=[llm.DraftSkill(name=" Python ", weight=5), llm.DraftSkill(name="RAG", weight=0)],
                aliases=[llm.DraftAlias(skill="rag", spellings=["retrieval augmented generation"]),
                         llm.DraftAlias(skill="java", spellings=["jvm"])],
                resumes=[llm.DraftResume(file="a.pdf", label="DA", focus=["SQL"]),
                         llm.DraftResume(file="b.pdf", label="DA", focus=["Tableau"])])
    return llm.ProfileDraft(**(base | kw))


def test_build_profile_normalises_the_draft(cfg):
    backend = FakeBackend(draft())
    p = llm.build_profile({"a.pdf": "text a", "b.pdf": "text b"}, cfg, backend)
    assert p.skills == {"python": 3, "rag": 1}                      # lowercased, weights clamped to 1..3
    assert p.aliases == {"rag": ["retrieval augmented generation"]}  # alias for an unknown skill dropped
    assert [r.label for r in p.resumes] == ["DA", "DA+"] and p.resumes[1].focus == ["tableau"]
    assert p.experience_years == 0
    _, user, output = backend.calls[0]
    assert '<resume file="a.pdf">\ntext a\n</resume>' in user and output is llm.ProfileDraft


def test_build_profile_cli_writes_and_never_overwrites(cfg, monkeypatch, tmp_path, caplog):
    folder = tmp_path / "resumes"
    folder.mkdir()
    tiny_docx(folder / "cv.docx", ["Asha, Python and RAG"])
    monkeypatch.setattr(llm, "make_backend", lambda c: FakeBackend(draft()))
    with caplog.at_level(logging.INFO, logger="jsearch"):
        assert fetch_jobs.build_profile_cli(folder, cfg) == 0
        assert fetch_jobs.build_profile_cli(folder, cfg) == 0
    assert load_profile(cfg["paths"]["profile"]).name == "Asha"
    assert cfg["paths"]["profile"].with_name("profile.new.yaml").exists()
    assert "already existed" in caplog.text


def test_build_profile_cli_reports_problems(cfg, monkeypatch, tmp_path, caplog):
    assert fetch_jobs.build_profile_cli(tmp_path / "missing", cfg) == 1
    folder = tmp_path / "r"
    folder.mkdir()
    tiny_docx(folder / "cv.docx", ["text"])
    monkeypatch.setattr(llm, "make_backend", lambda c: (_ for _ in ()).throw(llm.LLMUnavailable("no key")))
    assert fetch_jobs.build_profile_cli(folder, cfg) == 1
    assert "could not build the profile: no key" in caplog.text
    assert not cfg["paths"]["profile"].exists()
