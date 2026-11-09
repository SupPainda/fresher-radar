"""The shared worker loop (run_source), orchestration (scrape) and the jobspy source, with fakes."""
import logging

import pandas as pd
import pytest

from jsearch import runner
from jsearch.sources import Result, Unit, jobspy_site, source_type
from tests.helpers import job_frame

BLOCK_MSG = "response status code 406 (recaptcha required): rate limited"


class FakeStop:
    """A stop event that records waits instead of sleeping."""

    def __init__(self, set_after_waits: int | None = None):
        self.waits: list[float] = []
        self._set = False
        self.set_after_waits = set_after_waits

    def is_set(self) -> bool:
        return self._set

    def set(self) -> None:
        self._set = True

    def wait(self, seconds: float) -> bool:
        self.waits.append(seconds)
        if self.set_after_waits is not None and len(self.waits) >= self.set_after_waits:
            self._set = True
        return self._set


def units(n: int, tags: bool = True) -> list[Unit]:
    return [Unit(key=f"s|T|L{i}", label=f"T / L{i}", args={"i": i},
                 tags={"search_title": "T", "search_location": f"L{i}"} if tags else {}) for i in range(n)]


def scripted_fetch(script: list[tuple[str, int]], calls: list):
    """A fake fetch returning ("ok"|"blocked"|"error", rows) per unit, in order."""
    it = iter(script)

    def fetch(name, unit, src, cfg):
        kind, n = next(it)
        calls.append(unit.key)
        if kind == "error":
            raise ConnectionError("boom")
        return Result(df=job_frame(n, site=name, prefix=unit.label),
                      refusal=BLOCK_MSG if kind == "blocked" else None)
    return fetch


def run(cfg: dict, script, n: int, stop=None, src: dict | None = None):
    calls: list = []
    out = runner.run_source("naukri", units(n), scripted_fetch(script, calls), src or {}, cfg, stop or FakeStop())
    return out, calls


# --- run_source: happy path -------------------------------------------------------------------------------

def test_counts_rows_and_tags_each_unit(cfg):
    (frames, stats, errors), _ = run(cfg, [("ok", 2), ("ok", 0)], 2)
    assert stats == {"naukri": {"ok": 1, "empty": 1, "error": 0, "blocked": 0, "rows": 2}}
    assert errors == []
    tagged = pd.concat(frames)
    assert set(zip(tagged["search_title"], tagged["search_location"])) == {("T", "L0")}


def test_untagged_units_add_no_columns(cfg):
    frames, _, _ = runner.run_source("greenhouse", units(1, tags=False), scripted_fetch([("ok", 1)], []),
                                     {}, cfg, FakeStop())
    assert "search_title" not in frames[0]


def test_source_delay_overrides_the_default(cfg):
    stop = FakeStop()
    cfg["search"]["delay_seconds"] = [4, 4]
    run(cfg, [("ok", 1)] * 2, 2, stop=stop)
    run(cfg, [("ok", 1)] * 2, 2, stop=(stop2 := FakeStop()), src={"delay_seconds": [12, 12]})
    assert stop.waits == [4] and stop2.waits == [12]   # one delay between two units, none after the last


# --- run_source: blocks and failures -----------------------------------------------------------------------

def test_block_keeps_partial_rows_and_pauses_for_the_cooldown(cfg):
    cfg["search"]["on_block"]["cooldown_minutes"] = 5
    stop = FakeStop()
    (frames, stats, errors), _ = run(cfg, [("ok", 3), ("blocked", 2), ("ok", 3)], 3, stop=stop)
    assert stats["naukri"] == {"ok": 2, "empty": 0, "error": 0, "blocked": 1, "rows": 8}
    assert sum(len(f) for f in frames) == 8
    assert 300 in stop.waits
    assert errors == [f"naukri | T / L1: BLOCKED after 2 rows: {BLOCK_MSG}"]


def test_gives_up_after_three_blocks_in_a_row(cfg):
    script = [("blocked", 1), ("ok", 1), ("blocked", 1), ("blocked", 0), ("blocked", 0), ("ok", 1), ("ok", 1)]
    (_, stats, errors), calls = run(cfg, script, 7)
    assert len(calls) == 5                              # the last two units are skipped
    assert stats["naukri"]["blocked"] == 4 and stats["naukri"]["ok"] == 1
    assert len(errors) == 4


def test_fetch_error_is_recorded_and_the_worker_continues(cfg):
    (frames, stats, errors), _ = run(cfg, [("error", 0), ("ok", 2)], 2)
    assert stats["naukri"]["error"] == 1 and stats["naukri"]["ok"] == 1
    assert errors == ["naukri | T / L0: ConnectionError: boom"]
    assert sum(len(f) for f in frames) == 2


def test_stop_event_ends_the_worker_between_units(cfg):
    (_, stats, _), calls = run(cfg, [("ok", 1)] * 5, 5, stop=FakeStop(set_after_waits=2))
    assert len(calls) == 2 and stats["naukri"]["ok"] == 2


def test_already_stopped_worker_does_nothing(cfg):
    stop = FakeStop()
    stop.set()
    (frames, stats, _), calls = run(cfg, [("ok", 1)], 3, stop=stop)
    assert calls == [] and frames == [] and stats["naukri"]["rows"] == 0


def test_no_units_is_an_empty_outcome(cfg):
    (frames, stats, errors), _ = run(cfg, [], 0)
    assert frames == [] and errors == [] and stats["naukri"]["rows"] == 0


# --- source registry ---------------------------------------------------------------------------------------

def test_every_configured_source_has_a_known_type(cfg):
    for name, src in cfg["sources"].items():
        assert hasattr(source_type(name, src), "fetch")


def test_unknown_source_type_is_rejected():
    with pytest.raises(ValueError, match="unknown type 'rss'"):
        source_type("feed", {"type": "rss"})


# --- jobspy source -----------------------------------------------------------------------------------------

def test_jobspy_units_are_titles_times_locations(cfg):
    us = jobspy_site.units("indeed", cfg["sources"]["indeed"], cfg, test=False)
    assert len(us) == len(cfg["titles"]["search"]) * len(cfg["locations"])
    assert len({u.key for u in us}) == len(us)
    [one] = jobspy_site.units("indeed", cfg["sources"]["indeed"], cfg, test=True)
    assert one.args["title"] == cfg["search"]["test"]["title"]


def test_jobspy_options_override_defaults(cfg, monkeypatch):
    captured = {}
    monkeypatch.setattr(jobspy_site, "scrape_jobs", lambda **kw: captured.update(kw) or pd.DataFrame())
    unit = jobspy_site.units("naukri", cfg["sources"]["naukri"], cfg, test=True)[0]
    jobspy_site.fetch("naukri", unit, cfg["sources"]["naukri"], cfg)
    assert captured["results_wanted"] == cfg["sources"]["naukri"]["options"]["results_wanted"]
    jobspy_site.fetch("indeed", unit, cfg["sources"]["indeed"], cfg)
    assert captured["results_wanted"] == cfg["search"]["results_per_search"]
    assert captured["country_indeed"] == "India" and captured["hours_old"] == cfg["search"]["days"] * 24


def test_jobspy_refusal_is_read_from_frame_attrs(cfg, monkeypatch):
    def blocked(**kw):
        df = job_frame(2, site="naukri")
        df.attrs = {"errors": {"naukri": BLOCK_MSG}}
        return df
    monkeypatch.setattr(jobspy_site, "scrape_jobs", blocked)
    unit = jobspy_site.units("naukri", cfg["sources"]["naukri"], cfg, test=True)[0]
    result = jobspy_site.fetch("naukri", unit, cfg["sources"]["naukri"], cfg)
    assert result.refusal == BLOCK_MSG and len(result.df) == 2


# --- scrape(): orchestration -------------------------------------------------------------------------------

def fake_fetch(name, unit, src, cfg):
    return Result(df=job_frame(2, site=name, prefix=unit.label))


@pytest.fixture
def fake_sources(cfg, monkeypatch):
    """Every source type answers with 2 rows per unit; 2 titles x 2 locations; 2 boards per ATS source."""
    cfg["titles"]["search"] = ["AI Engineer", "Data Engineer"]
    cfg["locations"] = cfg["locations"][:2]
    for src in cfg["sources"].values():
        if "boards" in src:
            src["boards"] = src["boards"][:2]
    monkeypatch.setattr(jobspy_site, "fetch", fake_fetch)
    from jsearch.sources import ats
    monkeypatch.setattr(ats, "fetch", fake_fetch)
    return cfg


def test_parallel_and_sequential_give_the_same_rows(fake_sources):
    results = []
    for parallel in (True, False):
        fake_sources["env"]["parallel"] = parallel
        raw, _ = runner.scrape(fake_sources, ["naukri", "indeed", "greenhouse"], test=False)
        results.append(sorted(raw["job_url"]))
    assert results[0] == results[1] and len(results[0]) == 2 * (4 + 4 + 2)


def test_crashing_worker_does_not_lose_other_sources(fake_sources, monkeypatch, caplog):
    real = runner.run_source

    def crash_greenhouse(name, *a):
        if name == "greenhouse":
            raise ValueError("ats bug")
        return real(name, *a)
    monkeypatch.setattr(runner, "run_source", crash_greenhouse)
    with caplog.at_level(logging.INFO, logger="jsearch"):
        raw, _ = runner.scrape(fake_sources, ["indeed", "greenhouse"], test=False)
    assert set(raw["site"]) == {"indeed"} and len(raw) == 8
    assert "greenhouse: worker crashed: ValueError: ats bug" in caplog.text


def test_no_results_returns_none(fake_sources, monkeypatch):
    monkeypatch.setattr(jobspy_site, "fetch", lambda *a: Result(df=pd.DataFrame()))
    assert runner.scrape(fake_sources, ["indeed"], test=False) == (None, False)


def test_merge_with_saved_replaces_only_fetched_sources(tmp_path):
    saved = pd.concat([job_frame(2, "indeed"), job_frame(1, "naukri")])
    path = tmp_path / "raw.pkl"
    saved.to_pickle(path)
    fresh = job_frame(3, "naukri", prefix="Fresh")
    merged = runner.merge_with_saved(fresh, path)
    assert sorted(merged["site"].value_counts().items()) == [("indeed", 2), ("naukri", 3)]
    assert runner.merge_with_saved(fresh, tmp_path / "absent.pkl") is fresh
