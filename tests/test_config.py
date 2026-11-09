"""config.json validation and .env loading."""
import copy
import json

import pytest

from jsearch.config import CONFIG_PATH, ConfigError, load_config, load_env, validate

REAL = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def broken(change) -> dict:
    """A copy of the real config with one change applied."""
    c = copy.deepcopy(REAL)
    change(c)
    return c


def problems(raw: dict) -> str:
    with pytest.raises(ConfigError) as e:
        validate(raw)
    return str(e.value)


# --- happy path -------------------------------------------------------------------------------------------

def test_the_real_config_is_valid():
    validate(REAL)


def test_load_config_adds_env_paths_and_patterns(monkeypatch, tmp_path):
    monkeypatch.setenv("JSEARCH_DATA_DIR", str(tmp_path))
    cfg = load_config()
    assert cfg["paths"]["csv"] == tmp_path / REAL["output"]["csv"]
    assert cfg["rx"]["want"].search("Data Engineer")


# --- failure modes: each problem is reported with its location --------------------------------------------

def test_bad_regex_names_list_index_and_pattern():
    msg = problems(broken(lambda c: c["titles"]["exclude"].append("foo(")))
    index = len(REAL["titles"]["exclude"])
    assert f"titles.exclude: Value error, [{index}] 'foo(': missing ), unterminated subpattern" in msg


def test_missing_key_is_reported():
    assert "search.days: Field required" in problems(broken(lambda c: c["search"].pop("days")))


def test_typo_in_a_key_is_reported():
    msg = problems(broken(lambda c: c["filters"].update(max_min_year=2)))
    assert "filters.max_min_year: Extra inputs are not permitted" in msg


def test_unknown_source_type_is_reported():
    msg = problems(broken(lambda c: c["sources"].update(feed={"type": "rss"})))
    assert "sources.feed" in msg and "'rss'" in msg


def test_delay_min_above_max_is_reported():
    msg = problems(broken(lambda c: c["sources"]["naukri"].update(delay_seconds=[20, 12])))
    assert "sources.naukri.jobspy.delay_seconds" in msg and "min 20.0 is larger than max 12.0" in msg


@pytest.mark.parametrize("value", [0, -5, "sixty"])
def test_invalid_max_age_days_is_reported(value):
    msg = problems(broken(lambda c: c["sources"]["greenhouse"].update(max_age_days=value)))
    assert "sources.greenhouse.ats.max_age_days" in msg


@pytest.mark.parametrize("value", [0, -1, "fast"])
def test_invalid_llm_timeout_is_reported(value):
    msg = problems(broken(lambda c: c["ranking"]["llm"].update(timeout_seconds=value)))
    assert "ranking.llm.timeout_seconds" in msg


def test_fallback_models_must_be_a_list_of_names():
    msg = problems(broken(lambda c: c["ranking"]["llm"]["providers"]["gemini"].update(fallback_models="m-2")))
    assert "ranking.llm.providers.gemini.fallback_models" in msg


def test_wrong_type_is_reported():
    msg = problems(broken(lambda c: c["search"].update(days="fourteen")))
    assert "search.days" in msg


def test_ats_source_without_endpoint_is_reported():
    msg = problems(broken(lambda c: c["sources"].update(teamtailor={"type": "ats", "boards": ["x"]})))
    assert "sources.teamtailor: an ats source needs an ats.endpoints entry" in msg


@pytest.mark.parametrize("change, where", [
    (lambda c: c["ranking"]["top"].update(min_jobs=70, max_jobs=60), "min_jobs 70 is larger than max_jobs 60"),
    (lambda c: c["ranking"]["top"].update(min_fit=120), "ranking.top.min_fit"),
    (lambda c: c["ranking"]["top"].pop("more_fit"), "ranking.top.more_fit: Field required"),
    (lambda c: c["ranking"]["llm"].update(top_n=30), "ranking.llm.top_n: Extra inputs are not permitted"),
])
def test_invalid_top_list_settings_are_reported(change, where):
    assert where in problems(broken(change))


@pytest.mark.parametrize("change, where", [
    (lambda c: c["ats"].update(detail_delay_seconds=[2, 1]), "ats.detail_delay_seconds"),
    (lambda c: c["ats"].update(max_details_per_board=-1), "ats.max_details_per_board"),
    (lambda c: c["ats"].pop("max_details_per_board"), "ats.max_details_per_board: Field required"),
])
def test_invalid_detail_limits_are_reported(change, where):
    assert where in problems(broken(change))


def test_test_location_must_exist():
    msg = problems(broken(lambda c: c["search"]["test"].update(location="Atlantis")))
    assert "search.test.location 'Atlantis' is not a locations label" in msg


def test_several_problems_are_all_listed():
    def two(c):
        c["search"].pop("days")
        c["filters"]["max_min_years"] = -1
    msg = problems(broken(two))
    assert "search.days" in msg and "filters.max_min_years" in msg


def test_invalid_json_is_a_config_error(tmp_path):
    bad = tmp_path / "config.json"
    bad.write_text('{"sources": ', encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid JSON"):
        load_config(bad)


# --- .env -------------------------------------------------------------------------------------------------

def test_env_defaults_and_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("JSEARCH_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("JSEARCH_PARALLEL", "No")
    monkeypatch.setenv("JSEARCH_PROXIES", " a:1, ,b:2 ")
    env = load_env()
    assert env == {"data_dir": tmp_path, "parallel": False, "proxies": ["a:1", "b:2"]}


def test_relative_data_dir_is_under_the_project(monkeypatch):
    monkeypatch.setenv("JSEARCH_DATA_DIR", "somewhere")
    monkeypatch.setenv("JSEARCH_PROXIES", "")
    env = load_env()
    assert env["data_dir"] == CONFIG_PATH.parent / "somewhere" and env["proxies"] is None
