"""config.json says what to search for; the environment (.env) says where files go and how to run."""
import json
import os
import re
from pathlib import Path
from typing import Annotated, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.json"
OFF = {"0", "false", "no", "off"}


class ConfigError(ValueError):
    """config.json is malformed; the message lists every problem with its location."""


# --- schema: what config.json must look like (unknown keys are errors, so typos fail fast) ----------------

class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


Delay = Annotated[list[Annotated[float, Field(ge=0)]], Field(min_length=2, max_length=2)]


def _delay_ok(v: list[float] | None) -> list[float] | None:
    if v is not None and v[0] > v[1]:
        raise ValueError(f"min {v[0]} is larger than max {v[1]}")
    return v


def _compiles(patterns: list[str]) -> list[str]:
    for i, p in enumerate(patterns):
        try:
            re.compile(p)
        except re.error as e:
            raise ValueError(f"[{i}] {p!r}: {e}") from None
    return patterns


class JobspySource(Strict):
    type: Literal["jobspy"]
    delay_seconds: Delay | None = None
    max_age_days: int | None = Field(default=None, gt=0)  # default: search.days
    options: dict = {}
    _delay = field_validator("delay_seconds")(_delay_ok)


class AtsSource(Strict):
    type: Literal["ats"]
    delay_seconds: Delay | None = None
    max_age_days: int | None = Field(default=None, gt=0)  # boards list open jobs dated by first posting
    boards: list[str]
    _delay = field_validator("delay_seconds")(_delay_ok)


class Location(Strict):
    label: str
    query: str
    remote: bool


class TestSettings(Strict):
    title: str
    location: str
    preview_rows: int = Field(gt=0)
    boards_per_provider: int = Field(gt=0)


class OnBlock(Strict):
    cooldown_minutes: float = Field(ge=0)
    give_up_after: int = Field(ge=1)


class Search(Strict):
    days: int = Field(gt=0)
    results_per_search: int = Field(gt=0)
    delay_seconds: Delay
    test: TestSettings
    on_block: OnBlock
    _delay = field_validator("delay_seconds")(_delay_ok)


class Titles(Strict):
    search: list[str] = Field(min_length=1)
    want: list[str] = Field(min_length=1)
    exclude: list[str]
    drop: list[str]
    level_codes: list[str]
    seniority_words: list[str]
    manager_allowed: list[str]
    needs_entry_signal: list[str]
    entry_signal: list[str]
    internship: str
    _patterns = field_validator("want", "exclude", "drop", "level_codes", "seniority_words", "manager_allowed",
                                "needs_entry_signal", "entry_signal")(_compiles)

    @field_validator("internship")
    @classmethod
    def _internship_compiles(cls, v: str) -> str:
        _compiles([v])
        return v


class Filters(Strict):
    max_min_years: int = Field(ge=0)
    experience_context_chars: int = Field(gt=0)


class Output(Strict):
    csv: str
    raw: str
    db: str
    rejected: str
    daily_dir: str                # a dated copy of the CSV per day: <daily_dir>/jobs_<date>.csv
    columns: dict[str, str] = Field(min_length=1)
    hide_statuses: list[str]


class Ats(Strict):
    timeout_seconds: float = Field(gt=0)
    endpoints: dict[str, str]
    location_match: list[str] = Field(min_length=1)
    detail_delay_seconds: Delay           # between per-posting requests (SmartRecruiters)
    max_details_per_board: int = Field(ge=0)
    _delay = field_validator("detail_delay_seconds")(_delay_ok)

    @field_validator("endpoints")
    @classmethod
    def _has_slug(cls, v: dict[str, str]) -> dict[str, str]:
        for provider, url in v.items():
            if "{slug}" not in url:
                raise ValueError(f"{provider}: endpoint must contain {{slug}}")
        return v


PROVIDERS = ("anthropic", "gemini", "openrouter", "xai", "openai-compatible")


class LlmProvider(Strict):
    base_url: str | None = None   # None: the SDK default (Anthropic) or required via LLM_BASE_URL
    model: str | None = None      # None: LLM_MODEL, else the first model the key can list
    fallback_models: list[str] = []   # tried in order, right away, when the model before them fails
    price_per_mtok: Annotated[list[Annotated[float, Field(ge=0)]], Field(min_length=2, max_length=2)] | None = None


class Llm(Strict):
    enabled: bool               # rescores the top list (ranking.top)
    max_tokens: int = Field(gt=0)
    timeout_seconds: float = Field(gt=0)   # per request; a model that takes longer counts as failed
    effort: Literal["low", "medium", "high", "xhigh", "max"]   # Anthropic only
    providers: dict[str, LlmProvider]

    @field_validator("providers")
    @classmethod
    def _known(cls, v: dict[str, LlmProvider]) -> dict[str, LlmProvider]:
        if set(v) != set(PROVIDERS):
            raise ValueError(f"need exactly these providers: {list(PROVIDERS)}")
        if not v["anthropic"].model:
            raise ValueError("anthropic needs a model")
        return v


class TopList(Strict):
    """Which jobs the LLM rescores and the email lists: every job with fit >= min_fit, at least min_jobs and
    at most max_jobs (resume reserves come on top, within max_jobs). The email adds a count of the others
    with fit >= more_fit."""
    min_fit: float = Field(ge=0, le=100)
    min_jobs: int = Field(ge=0)
    max_jobs: int = Field(ge=0)
    more_fit: float = Field(ge=0, le=100)

    @model_validator(mode="after")
    def _order(self) -> "TopList":
        if self.min_jobs > self.max_jobs:
            raise ValueError(f"min_jobs {self.min_jobs} is larger than max_jobs {self.max_jobs}")
        return self


class Ranking(Strict):
    profile: str
    skill_points: float = Field(ge=0)
    role_points: float = Field(ge=0)
    skill_saturation_min: float = Field(gt=0)
    skill_saturation_share: float = Field(gt=0, le=1)
    title_multiplier: float = Field(ge=1)
    top: TopList
    llm: Llm


class Email(Strict):
    enabled: bool               # credentials (EMAIL_FROM / EMAIL_APP_PASSWORD / EMAIL_TO) live in .env
    smtp_host: str
    smtp_port: int = Field(gt=0, lt=65536)
    timeout_seconds: float = Field(gt=0)


class Config(Strict):
    sources: dict[str, Annotated[JobspySource | AtsSource, Field(discriminator="type")]] = Field(min_length=1)
    locations: list[Location] = Field(min_length=1)
    search: Search
    titles: Titles
    filters: Filters
    output: Output
    ats: Ats
    ranking: Ranking
    email: Email

    @model_validator(mode="after")
    def _cross_checks(self) -> "Config":
        from jsearch.sources.ats import PROVIDERS
        for name, src in self.sources.items():
            if src.type == "ats" and (name not in self.ats.endpoints or name not in PROVIDERS):
                raise ValueError(f"sources.{name}: an ats source needs an ats.endpoints entry and a parser; "
                                 f"known: {sorted(PROVIDERS)}")
        if self.search.test.location not in {loc.label for loc in self.locations}:
            raise ValueError(f"search.test.location {self.search.test.location!r} is not a locations label")
        return self


def validate(raw: dict) -> None:
    """Raise ConfigError listing every problem in config.json, each with its location."""
    try:
        Config.model_validate(raw)
    except ValidationError as e:
        problems = [f"  {'.'.join(map(str, err['loc'])) or '(top level)'}: {err['msg']}" for err in e.errors()]
        raise ConfigError("config.json is invalid:\n" + "\n".join(problems)) from None


def load_env() -> dict:
    """How this machine runs: from .env (see .env.example), with variables already set in the shell winning."""
    load_dotenv(ROOT / ".env")
    data_dir = Path(os.getenv("JSEARCH_DATA_DIR") or "data")
    return {
        "data_dir": data_dir if data_dir.is_absolute() else ROOT / data_dir,
        "parallel": os.getenv("JSEARCH_PARALLEL", "true").strip().lower() not in OFF,
        "proxies": [p.strip() for p in os.getenv("JSEARCH_PROXIES", "").split(",") if p.strip()] or None,
    }


def load_config(path: Path = CONFIG_PATH) -> dict:
    """config.json (validated) plus the environment, output paths and the compiled title patterns."""
    try:
        cfg = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ConfigError(f"config.json is not valid JSON: {e}") from None
    validate(cfg)
    cfg["env"] = env = load_env()
    cfg["paths"] = {k: env["data_dir"] / cfg["output"][k] for k in ("csv", "raw", "db", "rejected")}
    cfg["paths"]["partial"] = env["data_dir"] / "partial"  # progress of an unfinished run (--resume)
    cfg["paths"]["daily"] = env["data_dir"] / cfg["output"]["daily_dir"]
    cfg["paths"]["profile"] = env["data_dir"] / cfg["ranking"]["profile"]  # whose fit jobs are ranked by
    t = cfg["titles"]
    cfg["rx"] = {
        "want": any_of(t["want"]),
        "exclude": any_of(t["exclude"]),
        "drop": re.compile(r"\b(" + "|".join(t["drop"]) + r")\b", re.I),
        "level_code": any_of(t["level_codes"]),
        "seniority": re.compile(r"\b(" + "|".join(t["seniority_words"]) + r")\b", re.I),
        "manager_ok": any_of(t["manager_allowed"]),
        "needs_entry_signal": any_of(t["needs_entry_signal"]),
        "entry_signal": any_of(t["entry_signal"]),
    }
    return cfg


def any_of(patterns: list[str]) -> re.Pattern:
    return re.compile("|".join(f"(?:{p})" for p in patterns), re.I)
