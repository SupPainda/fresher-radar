"""Why a posting was rejected: reject_reason, rejected rows from clean(), --explain, rejected.csv."""
import sys
from argparse import Namespace

import pandas as pd
import pytest

import fetch_jobs
from jsearch import filters
from tests.helpers import job_frame


def reason(cfg: dict, title: str, years: float | None = None) -> str | None:
    return filters.reject_reason({"title": title, "min_years": years}, cfg)


@pytest.mark.parametrize("title, years, expected", [
    ("Sales Executive", None, "not a wanted title"),
    ("Salesforce Technical Consultant", None, "excluded word: Salesforce"),
    ("Presales Analyst", None, "excluded word: Analyst"),
    ("SDE3 Backend", None, "level code: SDE3"),
    ("AI Engineer - L3", None, "level code: L3"),
    ("Senior Data Engineer", None, "seniority word: senior"),
    ("Lead Data Engineer II", None, "seniority word: ii, lead"),
    ("Account Manager", None, "manager title not in titles.manager_allowed"),   # wanted, but only TAM is allowed
    ("Solution Architect", None, '"Solution Architect" without an entry-level word or stated experience'),
    ("Data Engineer", 5, "asks for 5+ years (limit 2)"),
    ("Data Engineer", 2.5, "asks for 2.5+ years (limit 2)"),
])
def test_each_rejection_names_its_reason(cfg, title, years, expected):
    assert reason(cfg, title, years) == expected


@pytest.mark.parametrize("title, years", [
    ("Data Engineer", None), ("Data Engineer", 2), ("Technical Account Manager", None),
    ("Associate Solutions Architect", None), ("AI/ML Intern", 0),
])
def test_kept_postings_have_no_reason(cfg, title, years):
    assert reason(cfg, title, years) is None
    assert filters.keep({"title": title, "min_years": years}, cfg)


def test_clean_returns_rejected_rows_with_reasons(cfg):
    raw = job_frame(3)
    raw.loc[0, "title"] = "Senior Data Engineer"
    raw.loc[1, "date_posted"] = pd.Timestamp("2026-01-01")
    kept, counts, rejected = filters.clean(raw, cfg)
    assert kept["title"].tolist() == ["Data Engineer 2"]
    assert dict(zip(rejected["title"], rejected["reason"])) == {
        "Data Engineer 1": f"posted 2026-01-01, older than {cfg['search']['days']} days",
        "Senior Data Engineer": "seniority word: senior",
    }
    assert counts["raw"] == len(kept) + len(rejected)


def test_clean_with_nothing_in_range_has_only_date_rejections(cfg):
    raw = job_frame(2).assign(date_posted=pd.Timestamp("2026-01-01"))
    kept, counts, rejected = filters.clean(raw, cfg)
    assert kept.empty and len(rejected) == 2 and counts["entry-level"] == 0


# --- --explain --------------------------------------------------------------------------------------------

def test_explain_reads_years_from_the_title(cfg):
    assert fetch_jobs.explain("Data Engineer (3-5 years)", None, cfg) == \
        "'Data Engineer (3-5 years)' (3 years): rejected: asks for 3+ years (limit 2)"
    assert fetch_jobs.explain("AI Engineer", None, cfg) == "'AI Engineer' (no experience stated): kept"
    assert fetch_jobs.explain("AI Engineer", 1, cfg) == "'AI Engineer' (1 years): kept"


def test_explain_cli_prints_and_writes_nothing(cfg, monkeypatch, capsys):
    monkeypatch.setattr(fetch_jobs, "load_config", lambda: cfg)
    monkeypatch.setattr(sys, "argv", ["fetch_jobs.py", "--explain", "SDE3 Backend"])
    fetch_jobs.main()
    assert capsys.readouterr().out.strip() == "'SDE3 Backend' (no experience stated): rejected: level code: SDE3"
    assert not any(cfg["env"]["data_dir"].iterdir())   # no log, no files


# --- rejected.csv -----------------------------------------------------------------------------------------

def test_run_writes_rejected_csv_sorted_by_reason(cfg):
    raw = job_frame(3)
    raw.loc[0, "title"] = "Senior Data Engineer"
    raw.loc[1, "title"] = "Sales Executive"
    cfg["paths"]["raw"].parent.mkdir(parents=True, exist_ok=True)
    raw.to_pickle(cfg["paths"]["raw"])
    fetch_jobs.run(Namespace(test=False, refilter=True, sources=[]), cfg)
    rej = pd.read_csv(cfg["paths"]["rejected"], encoding="utf-8-sig")
    assert rej["reason"].tolist() == ["not a wanted title", "seniority word: senior"]
    assert list(rej.columns[:3]) == ["reason", "title", "company"]
