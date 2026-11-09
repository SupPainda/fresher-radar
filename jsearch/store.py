"""SQLite memory of every job that ever passed the filters: when it was first/last seen and its status.

The CSV stays the place to work from; the database only remembers. Statuses typed into the CSV's
"status" column (e.g. applied, rejected, interested) are read back at the start of the next run,
so marking jobs in Excel is enough.
"""
import sqlite3
from pathlib import Path

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    key         TEXT PRIMARY KEY,   -- normalised "title|company": a repost with a new URL is the same job
    job_url     TEXT,
    title       TEXT,
    company     TEXT,
    location    TEXT,
    site        TEXT,
    date_posted TEXT,
    min_years   REAL,
    is_remote   INTEGER,
    internship  INTEGER,
    first_seen  TEXT NOT NULL,
    last_seen   TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'new',
    notes       TEXT
);
CREATE TABLE IF NOT EXISTS llm_scores (       -- Claude's fit scores, so a job is only paid for once per profile
    key          TEXT NOT NULL,
    profile_hash TEXT NOT NULL,
    score        INTEGER NOT NULL,
    reason       TEXT,
    best_resume  TEXT,
    model        TEXT,
    scored_at    TEXT,
    PRIMARY KEY (key, profile_hash)
);
"""
FIELDS = ["job_url", "title", "company", "location", "site", "date_posted", "min_years", "is_remote", "internship"]


def norm(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.lower().str.replace(r"\W+", " ", regex=True).str.strip()


def job_key(title: pd.Series, company: pd.Series) -> pd.Series:
    """Same key for the frame and the CSV, so CSV edits find their database row."""
    return norm(title) + "|" + norm(company)


def connect(path: Path | str) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    return con


def llm_scores(con: sqlite3.Connection, profile_hash: str) -> dict[str, tuple[int, str, str, str]]:
    """{job key: (score, reason, best_resume, model)} scored for this profile."""
    rows = con.execute("SELECT key, score, reason, best_resume, model FROM llm_scores WHERE profile_hash = ?",
                       (profile_hash,)).fetchall()
    return {k: (s, r, b, m) for k, s, r, b, m in rows}


def save_llm_scores(con: sqlite3.Connection, profile_hash: str, scored_at: str,
                    scores: dict[str, tuple[int, str, str, str]]) -> None:
    """scores: {job key: (score, reason, best_resume, model)}."""
    con.executemany("INSERT OR REPLACE INTO llm_scores VALUES (?, ?, ?, ?, ?, ?, ?)",
                    [(k, profile_hash, s, r, b, m, scored_at) for k, (s, r, b, m) in scores.items()])
    con.commit()


def import_csv_edits(con: sqlite3.Connection, csv_path: Path | str, title_col: str, company_col: str) -> int:
    """Copy status/notes typed into the job list (.csv, or the .xlsx copy) back into the database.
    Returns how many rows changed."""
    if not Path(csv_path).exists():
        return 0
    if Path(csv_path).suffix.lower() == ".xlsx":
        csv = pd.read_excel(csv_path, dtype=str)
    else:
        csv = pd.read_csv(csv_path, encoding="utf-8-sig", dtype=str)
    if "status" not in csv:
        return 0  # a CSV written before the database existed
    csv["key"] = job_key(csv[title_col], csv[company_col])
    csv["status"] = csv["status"].fillna("new").str.strip().str.lower().replace("", "new")
    csv["notes"] = csv["notes"].where(csv["notes"].notna(), None) if "notes" in csv else None
    before = con.total_changes
    con.executemany("UPDATE jobs SET status = ?, notes = ? WHERE key = ? AND (status != ? OR notes IS NOT ?)",
                    [(r.status, r.notes, r.key, r.status, r.notes) for r in csv.itertuples()])
    con.commit()
    return con.total_changes - before


def record(con: sqlite3.Connection, df: pd.DataFrame, today: str, touch: bool = True) -> pd.DataFrame:
    """Upsert the filtered frame. `touch` updates last_seen (a real scrape); a refilter only adds rows.

    Returns df with first_seen, status and notes from the database.
    """
    df = df.assign(key=job_key(df["title"], df["company"]))
    rows = df[["key"] + FIELDS].copy()
    rows["date_posted"] = rows["date_posted"].map(lambda d: None if pd.isna(d) else str(pd.Timestamp(d).date()))
    rows["min_years"] = rows["min_years"].astype(float).where(rows["min_years"].notna(), None)
    rows = rows.astype(object).where(rows.notna(), None)
    set_seen = ", last_seen = excluded.last_seen" if touch else ""
    con.executemany(
        f"""INSERT INTO jobs (key, {", ".join(FIELDS)}, first_seen, last_seen)
            VALUES ({", ".join("?" * (len(FIELDS) + 3))})
            ON CONFLICT(key) DO UPDATE SET {", ".join(f"{f} = excluded.{f}" for f in FIELDS)}{set_seen}""",
        [tuple(r) + (today, today) for r in rows.itertuples(index=False)])
    con.commit()
    memo = pd.read_sql("SELECT key, first_seen, status, notes FROM jobs", con)
    return df.merge(memo, on="key", how="left").drop(columns="key")


def status_counts(con: sqlite3.Connection) -> dict[str, int]:
    return dict(con.execute("SELECT status, COUNT(*) FROM jobs GROUP BY status ORDER BY 2 DESC").fetchall())
