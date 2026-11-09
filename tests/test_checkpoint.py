"""Saving progress during a run, and --resume / --fresh."""
from argparse import Namespace

import pandas as pd
import pytest

import fetch_jobs
from jsearch import runner
from jsearch.runner import Checkpoint
from jsearch.sources import Result, jobspy_site
from tests.helpers import job_frame
from tests.test_runner import FakeStop, units


# --- Checkpoint itself ------------------------------------------------------------------------------------

def test_add_and_load_round_trip(tmp_path):
    cp = Checkpoint.start(tmp_path / "partial", ["--sources", "indeed"])
    cp.add("indeed|A|L0", job_frame(2))
    cp.add("indeed|A|L1", pd.DataFrame())          # empty: done, but no file
    loaded, frames = Checkpoint.load(tmp_path / "partial")
    assert loaded.done == {"indeed|A|L0", "indeed|A|L1"}
    assert [len(f) for f in frames] == [2]
    assert Checkpoint.started_info(tmp_path / "partial")["args"] == ["--sources", "indeed"]


def test_new_files_continue_numbering_after_a_resume(tmp_path):
    folder = tmp_path / "partial"
    Checkpoint.start(folder, []).add("a", job_frame(1))
    cp, _ = Checkpoint.load(folder)
    cp.add("b", job_frame(1, prefix="Other"))
    _, frames = Checkpoint.load(folder)
    assert len(frames) == 2                          # "b" did not overwrite "a"'s file


def test_a_line_cut_short_by_a_crash_is_skipped(tmp_path):
    folder = tmp_path / "partial"
    Checkpoint.start(folder, []).add("a", job_frame(1))
    with open(folder / Checkpoint.DONE, "a", encoding="utf-8") as f:
        f.write('{"key": "b", "fi')                   # crash mid-write
    cp, frames = Checkpoint.load(folder)
    assert cp.done == {"a"} and len(frames) == 1


def test_finish_removes_the_folder(tmp_path):
    cp = Checkpoint.start(tmp_path / "partial", [])
    cp.finish()
    assert not (tmp_path / "partial").exists()


def test_started_info_records_the_date(tmp_path):
    Checkpoint.start(tmp_path / "partial", [])
    assert Checkpoint.started_info(tmp_path / "partial")["date"] == str(pd.Timestamp.now(tz="Asia/Kolkata").date())


def test_started_info_survives_a_missing_file(tmp_path):
    assert Checkpoint.started_info(tmp_path / "nowhere")["at"] == "an unknown time"


# --- run_source marks only clean units as done ------------------------------------------------------------

def test_only_ok_and_empty_units_are_marked_done(cfg, tmp_path):
    from tests.test_runner import scripted_fetch
    cp = Checkpoint.start(tmp_path / "partial", [])
    script = [("ok", 2), ("ok", 0), ("blocked", 1), ("error", 0)]
    runner.run_source("naukri", units(4), scripted_fetch(script, []), {}, cfg, FakeStop(), cp)
    assert cp.done == {"s|T|L0", "s|T|L1"}           # blocked L2 and failed L3 will be retried


# --- scrape: interrupt, then resume -----------------------------------------------------------------------

@pytest.fixture
def counting_sources(cfg, monkeypatch):
    """indeed only: 2 titles x 2 locations, 1 row per unit; records every unit fetched."""
    cfg["titles"]["search"] = ["AI Engineer", "Data Engineer"]
    cfg["locations"] = cfg["locations"][:2]
    cfg["env"]["parallel"] = False
    fetched: list[str] = []
    state = {"interrupt_at": None}

    def fetch(name, unit, src, c):
        if len(fetched) == state["interrupt_at"]:
            state["interrupt_at"] = None
            raise KeyboardInterrupt
        fetched.append(unit.key)
        return Result(df=job_frame(1, site=name, prefix=unit.label))
    monkeypatch.setattr(jobspy_site, "fetch", fetch)
    return cfg, fetched, state


def test_interrupted_run_resumes_without_repeating_units(counting_sources, tmp_path):
    cfg, fetched, state = counting_sources
    folder = tmp_path / "partial"
    state["interrupt_at"] = 2                         # Ctrl+C during the third unit
    raw1, complete1 = runner.scrape(cfg, ["indeed"], False, Checkpoint.start(folder, []))
    assert len(raw1) == 2 and complete1 is False

    cp, resumed = Checkpoint.load(folder)
    raw2, complete2 = runner.scrape(cfg, ["indeed"], False, cp, resumed)
    assert complete2 is True
    assert len(fetched) == 4 and len(set(fetched)) == 4    # each unit fetched exactly once overall
    assert sorted(raw2["job_url"]) == sorted(set(raw2["job_url"])) and len(raw2) == 4


def test_blocked_unit_is_retried_on_resume(cfg, monkeypatch, tmp_path):
    cfg["titles"]["search"], cfg["locations"] = ["AI Engineer"], cfg["locations"][:2]
    calls: list[str] = []
    blocked_once = {"done": False}

    def fetch(name, unit, src, c):
        calls.append(unit.key)
        if unit.key.endswith(cfg["locations"][1]["label"]) and not blocked_once["done"]:
            blocked_once["done"] = True
            return Result(df=pd.DataFrame(), refusal="406")
        return Result(df=job_frame(1, site=name, prefix=unit.label))
    monkeypatch.setattr(jobspy_site, "fetch", fetch)
    folder = tmp_path / "partial"
    _, complete = runner.scrape(cfg, ["indeed"], False, Checkpoint.start(folder, []))
    assert complete is False
    cp, resumed = Checkpoint.load(folder)
    _, complete = runner.scrape(cfg, ["indeed"], False, cp, resumed)
    assert complete is True and calls.count(calls[1]) == 2 and calls.count(calls[0]) == 1


# --- fetch_jobs: the guard and --fresh / --resume ---------------------------------------------------------

def args(**kw) -> Namespace:
    return Namespace(**({"test": False, "refilter": False, "sources": ["indeed"], "resume": False,
                         "fresh": False} | kw))


def test_unfinished_run_blocks_a_plain_run(cfg, tmp_path):
    Checkpoint.start(cfg["paths"]["partial"], ["--sources", "indeed"])
    with pytest.raises(fetch_jobs.UnfinishedRun, match="--resume to continue it, or --fresh"):
        fetch_jobs.open_checkpoint(cfg["paths"]["partial"], args())


def test_fresh_discards_and_resume_keeps(cfg):
    folder = cfg["paths"]["partial"]
    Checkpoint.start(folder, []).add("indeed|x", job_frame(1))
    cp, frames = fetch_jobs.open_checkpoint(folder, args(resume=True))
    assert cp.done == {"indeed|x"} and len(frames) == 1
    cp, frames = fetch_jobs.open_checkpoint(folder, args(fresh=True))
    assert cp.done == set() and frames == [] and not (folder / Checkpoint.DONE).exists()


def test_resume_without_progress_starts_normally(cfg):
    cp, frames = fetch_jobs.open_checkpoint(cfg["paths"]["partial"], args(resume=True))
    assert cp.done == set() and frames == [] and cfg["paths"]["partial"].exists()


def test_complete_run_removes_the_progress_folder(counting_sources):
    cfg, fetched, _ = counting_sources
    fetch_jobs.run(args(), cfg)
    assert len(fetched) == 4 and not cfg["paths"]["partial"].exists()
    assert len(pd.read_pickle(cfg["paths"]["raw"])) == 4


def test_incomplete_run_keeps_progress_for_resume(counting_sources):
    cfg, fetched, state = counting_sources
    state["interrupt_at"] = 1
    fetch_jobs.run(args(), cfg)
    assert cfg["paths"]["partial"].exists()
    with pytest.raises(fetch_jobs.UnfinishedRun):
        fetch_jobs.run(args(), cfg)
    fetch_jobs.run(args(resume=True), cfg)
    assert len(fetched) == 4 and not cfg["paths"]["partial"].exists()
