"""Unattended runs (--scheduled): which unfinished run to continue, carrying on past a CSV open in Excel, the
day's dated copy, and the email note."""
import json
from argparse import Namespace
from pathlib import Path

import pandas as pd
import pytest

import fetch_jobs
from jsearch import mailer
from jsearch.outputs import OutputLocked
from jsearch.runner import Checkpoint
from tests.helpers import job_frame
from tests.test_checkpoint import counting_sources  # noqa: F401  (fixture)

TODAY = str(pd.Timestamp.now(tz="Asia/Kolkata").date())


def args(**kw) -> Namespace:
    return Namespace(**({"test": False, "refilter": False, "sources": ["indeed"], "resume": False,
                         "fresh": False, "scheduled": True} | kw))


def started(folder: Path, date: str | None) -> None:
    info = {"at": f"{date} 07:00 PM IST", "args": []} | ({"date": date} if date else {})
    (folder / Checkpoint.STARTED).write_text(json.dumps(info), encoding="utf-8")


# --- which unfinished run to continue ----------------------------------------------------------------------

def test_scheduled_continues_an_unfinished_run_from_today(cfg):
    folder = cfg["paths"]["partial"]
    Checkpoint.start(folder, []).add("indeed|x", job_frame(1))
    cp, frames = fetch_jobs.open_checkpoint(folder, args())
    assert cp.done == {"indeed|x"} and len(frames) == 1


@pytest.mark.parametrize("date", ["2026-01-01", None])  # None: a checkpoint written before dates were stored
def test_scheduled_discards_an_older_unfinished_run(cfg, caplog, date):
    folder = cfg["paths"]["partial"]
    Checkpoint.start(folder, []).add("indeed|x", job_frame(1))
    started(folder, date)
    cp, frames = fetch_jobs.open_checkpoint(folder, args())
    assert cp.done == set() and frames == [] and not (folder / Checkpoint.DONE).exists()
    assert "is not from today" in caplog.text


def test_scheduled_without_progress_starts_normally(cfg):
    cp, frames = fetch_jobs.open_checkpoint(cfg["paths"]["partial"], args())
    assert cp.done == set() and frames == []


def test_scheduled_run_finishes_an_interrupted_one_without_repeating_units(counting_sources):  # noqa: F811
    cfg, fetched, state = counting_sources
    state["interrupt_at"] = 1
    fetch_jobs.run(args(scheduled=False), cfg)
    fetch_jobs.run(args(), cfg)                       # same day: continues it
    assert len(fetched) == 4 and len(set(fetched)) == 4 and not cfg["paths"]["partial"].exists()


# --- jobs.csv open in Excel --------------------------------------------------------------------------------

@pytest.fixture
def refilter_cfg(cfg):
    cfg["paths"]["raw"].parent.mkdir(parents=True, exist_ok=True)
    job_frame(3).to_pickle(cfg["paths"]["raw"])
    return cfg


def locked(*paths):
    raise OutputLocked("jobs.csv is open in another program (Excel?). Close it and run again.")


def test_a_locked_csv_stops_a_normal_run_but_not_a_scheduled_one(refilter_cfg, monkeypatch, caplog):
    monkeypatch.setattr(fetch_jobs, "check_writable", locked)
    with pytest.raises(OutputLocked):
        fetch_jobs.run(args(refilter=True, scheduled=False), refilter_cfg)
    fetch_jobs.run(args(refilter=True), refilter_cfg)
    assert "Scheduled run: carrying on" in caplog.text and refilter_cfg["paths"]["csv"].exists()


def test_the_email_says_when_the_list_went_to_a_copy(refilter_cfg, monkeypatch):
    real_save = fetch_jobs.save_csv
    csv = refilter_cfg["paths"]["csv"]

    def save(out, path):  # jobs.csv locked at save time: save_csv falls back to a timestamped copy
        return real_save(out, path.with_name("jobs_2026-11-09_19-05-00.csv") if path == csv else path)
    notes = []
    monkeypatch.setattr(fetch_jobs, "save_csv", save)
    monkeypatch.setattr(mailer, "send_top", lambda df, cfg, when, note="", **kw: notes.append(note))
    fetch_jobs.run(args(refilter=True, email=True), refilter_cfg)
    assert notes == ["jobs.csv was open in Excel, so this run's list was saved to jobs_2026-11-09_19-05-00.csv; "
                     "statuses typed into that copy are not read back."]


def test_no_note_when_jobs_csv_was_written(refilter_cfg, monkeypatch):
    notes = []
    monkeypatch.setattr(mailer, "send_top", lambda df, cfg, when, note="", **kw: notes.append(note))
    fetch_jobs.run(args(refilter=True, email=True), refilter_cfg)
    assert notes == [""]


# --- the day's copy ----------------------------------------------------------------------------------------

def test_each_run_writes_a_dated_copy_and_a_second_run_that_day_replaces_it(refilter_cfg):
    run = args(refilter=True, scheduled=False)
    fetch_jobs.run(run, refilter_cfg)
    daily = refilter_cfg["paths"]["daily"] / f"jobs_{TODAY}.csv"
    first = pd.read_csv(daily, encoding="utf-8-sig")
    assert first.equals(pd.read_csv(refilter_cfg["paths"]["csv"], encoding="utf-8-sig")) and len(first) == 3
    # titles ending 3+ would read as level codes ("Engineer 3"), so the two extra jobs get their own prefix
    pd.concat([job_frame(3), job_frame(2, prefix="AI Engineer")], ignore_index=True).to_pickle(
        refilter_cfg["paths"]["raw"])
    fetch_jobs.run(run, refilter_cfg)
    assert len(pd.read_csv(daily, encoding="utf-8-sig")) == 5
    assert [p.name for p in refilter_cfg["paths"]["daily"].iterdir()] == [f"jobs_{TODAY}.csv"]


def test_test_runs_write_no_dated_copy(cfg, monkeypatch):
    monkeypatch.setattr(fetch_jobs, "scrape", lambda *a: (job_frame(2), True))
    fetch_jobs.run(args(test=True, scheduled=False), cfg)
    assert not cfg["paths"]["daily"].exists()
