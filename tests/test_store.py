"""SQLite memory: first/last seen, statuses typed into the CSV, hidden statuses."""
from argparse import Namespace

import pandas as pd
import pytest

import fetch_jobs as fj
from jsearch import store
from tests.helpers import job_frame


def filtered(n: int = 2) -> pd.DataFrame:
    """A frame shaped like clean() output."""
    return job_frame(n).assign(min_years=1.0, internship=False)


@pytest.fixture
def con(tmp_path):
    c = store.connect(tmp_path / "jobs.db")
    yield c
    c.close()


# --- record() ---------------------------------------------------------------------------------------------

def test_record_adds_rows_as_new_with_first_seen(con):
    out = store.record(con, filtered(2), "2026-01-01")
    assert out["status"].tolist() == ["new", "new"]
    assert out["first_seen"].tolist() == ["2026-01-01", "2026-01-01"]
    assert store.status_counts(con) == {"new": 2}


def test_touch_updates_last_seen_and_refilter_does_not(con):
    store.record(con, filtered(1), "2026-01-01")
    store.record(con, filtered(1), "2026-01-02", touch=False)
    assert con.execute("SELECT first_seen, last_seen FROM jobs").fetchone() == ("2026-01-01", "2026-01-01")
    store.record(con, filtered(1), "2026-01-03")
    assert con.execute("SELECT first_seen, last_seen FROM jobs").fetchone() == ("2026-01-01", "2026-01-03")


def test_repost_under_new_url_keeps_status(con):
    store.record(con, filtered(1), "2026-01-01")
    con.execute("UPDATE jobs SET status = 'applied'")
    repost = filtered(1).assign(job_url="https://example.com/new-url")
    out = store.record(con, repost, "2026-01-05")
    assert out["status"].tolist() == ["applied"]
    assert con.execute("SELECT job_url, COUNT(*) FROM jobs").fetchone() == ("https://example.com/new-url", 1)


def test_job_key_ignores_case_and_punctuation():
    a = store.job_key(pd.Series(["Data Engineer (Python)"]), pd.Series(["ACME Pvt. Ltd"]))
    b = store.job_key(pd.Series(["data engineer python"]), pd.Series(["acme pvt ltd"]))
    assert a.tolist() == b.tolist()


# --- import_csv_edits() -----------------------------------------------------------------------------------

def write_csv(path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def test_csv_status_and_notes_are_read_back_normalised(con, tmp_path):
    store.record(con, filtered(2), "2026-01-01")
    csv = tmp_path / "jobs.csv"
    write_csv(csv, [{"title": "Data Engineer 0", "company": "Co0", "status": " Applied ", "notes": "ping"},
                    {"title": "Data Engineer 1", "company": "Co1", "status": "", "notes": None}])
    assert store.import_csv_edits(con, csv, "title", "company") == 1
    rows = dict((r[0], r[1:]) for r in con.execute("SELECT title, status, notes FROM jobs"))
    assert rows == {"Data Engineer 0": ("applied", "ping"), "Data Engineer 1": ("new", None)}


def test_unchanged_csv_reports_no_edits(con, tmp_path):
    store.record(con, filtered(1), "2026-01-01")
    csv = tmp_path / "jobs.csv"
    write_csv(csv, [{"title": "Data Engineer 0", "company": "Co0", "status": "new", "notes": None}])
    assert store.import_csv_edits(con, csv, "title", "company") == 0


def test_missing_csv_or_csv_without_status_column_is_ignored(con, tmp_path):
    assert store.import_csv_edits(con, tmp_path / "absent.csv", "title", "company") == 0
    csv = tmp_path / "old.csv"
    write_csv(csv, [{"title": "Data Engineer 0", "company": "Co0"}])
    assert store.import_csv_edits(con, csv, "title", "company") == 0


# --- end to end through fetch_jobs.run (refilter, offline) ------------------------------------------------

def test_refilter_run_writes_csv_and_hides_applied_jobs(cfg):
    paths = cfg["paths"]
    paths["raw"].parent.mkdir(parents=True, exist_ok=True)
    job_frame(3).to_pickle(paths["raw"])
    args = Namespace(test=False, refilter=True, sources=[])

    fj.run(args, cfg)
    first = pd.read_csv(paths["csv"], encoding="utf-8-sig", dtype=str)  # text, like an Excel edit
    assert len(first) == 3 and set(first["new"]) == {"NEW"} and set(first["status"]) == {"new"}

    first.loc[0, "status"] = "applied"
    first.loc[1, "notes"] = "follow up"
    first.to_csv(paths["csv"], index=False, encoding="utf-8-sig")

    fj.run(args, cfg)
    second = pd.read_csv(paths["csv"], encoding="utf-8-sig", dtype=str)
    assert len(second) == 2
    assert first.loc[0, "title"] not in second["title"].tolist()
    assert second.set_index("title").loc[first.loc[1, "title"], "notes"] == "follow up"


# --- statuses from the .csv and the .xlsx: the one edited last wins ---------------------------------------

def edit(path, title: str, status: str, mtime: float) -> None:
    """Change one row's status in the .csv or .xlsx job list, as Excel would, and set its modified time."""
    import os
    if path.suffix == ".xlsx":
        table = pd.read_excel(path, dtype=str)
        table.loc[table["title"] == title, "status"] = status
        table.to_excel(path, index=False, sheet_name="jobs")
    else:
        table = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
        table.loc[table["title"] == title, "status"] = status
        table.to_csv(path, index=False, encoding="utf-8-sig")
    os.utime(path, (mtime, mtime))


@pytest.mark.parametrize("newer", ["excel", "csv"])
def test_the_list_edited_last_wins(cfg, newer):
    paths = cfg["paths"]
    paths["raw"].parent.mkdir(parents=True, exist_ok=True)
    job_frame(2).to_pickle(paths["raw"])
    args = Namespace(test=False, refilter=True, sources=[])
    fj.run(args, cfg)
    assert paths["excel"].exists() and paths["csv"].exists()
    title = pd.read_csv(paths["csv"], encoding="utf-8-sig")["title"].iloc[0]
    older = "csv" if newer == "excel" else "excel"
    edit(paths[older], title, "interested", 1_000_000)   # an earlier edit in the other file...
    edit(paths[newer], title, "applied", 2_000_000)      # ...and the latest one, which must win
    fj.run(args, cfg)
    assert con_status(paths["db"], title) == "applied"
    assert title not in pd.read_excel(paths["excel"])["title"].tolist()   # applied: hidden from the list


def con_status(db, title: str) -> str:
    c = store.connect(db)
    try:
        return c.execute("SELECT status FROM jobs WHERE title = ?", (title,)).fetchone()[0]
    finally:
        c.close()
