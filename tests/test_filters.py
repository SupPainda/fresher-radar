"""Title and level filters (keep) and the clean() pipeline."""
import pandas as pd
import pytest

from jsearch import filters
from tests.helpers import job_frame


def kept(cfg: dict, title: str, years: float | None = None) -> bool:
    return filters.keep({"title": title, "min_years": years}, cfg)


# --- happy path: wanted titles pass -----------------------------------------------------------------------

@pytest.mark.parametrize("title", [
    "Data Engineer", "Data Engineer I", "L1 Data Engineer ( For Ops & Support)", "AI Engineer",
    "LLM Engineer", "Agentic AI Developer", "AI Developer", "Prompt Engineer", "Applied AI Engineer",
    "RAG Engineer", "Generative AI Engineer", "NLP Engineer", "Forward Deployed Engineer",
    "Technical Product Manager", "Technical Account Manager", "Junior Solutions Architect",
    "Graduate Engineer Trainee (GET)", "Web3 Backend Engineer", "Tech Consultant - AWS S3 Migration",
    "Cloud Engineer EC2",
])
def test_wanted_titles_pass(cfg, title):
    assert kept(cfg, title)


@pytest.mark.parametrize("title", [
    "AI/ML Intern", "GenAI Intern", "Data Engineering Intern", "Software Engineering Internship",
    "Python Developer Intern", "Intern - Data Science",
])
def test_tech_internships_pass(cfg, title):
    assert kept(cfg, title)


# --- failure modes: unwanted titles are dropped -----------------------------------------------------------

@pytest.mark.parametrize("title", ["HR Intern", "Marketing Internship", "Content Writing Intern",
                                   "Internal Audit Associate"])
def test_non_tech_internships_and_lookalikes_are_dropped(cfg, title):
    assert not kept(cfg, title)


@pytest.mark.parametrize("title", [
    "Senior LLM Engineer", "Lead Data Engineer", "Sr. Data Engineer", "Principal AI Engineer",
    "Data Engineer II", "Group Product Manager", "Engineering Manager", "Senior Technical Account Manager",
    "SSE - Backend +AI", "Intermediate Backend Engineer, Authorization",
])
def test_senior_titles_are_dropped(cfg, title):
    assert not kept(cfg, title)


@pytest.mark.parametrize("title", ["AI Engineer 4A", "Solution Architect - HR 3", "AI Engineer - L3",
                                   "SDE3 Backend"])
def test_level_codes_are_dropped(cfg, title):
    assert not kept(cfg, title)


@pytest.mark.parametrize("title", [
    "Salesforce Business Analyst", "Oracle VBCS Technical Consultant", "ELV System Pre-Sales Engineer",
    "D365 Technical Consultant F&O", "Business Central / Navision Technical Consultant",
    "SAP Technical Consultant", "Mechanical Design Engineer",
])
def test_excluded_domains_are_dropped(cfg, title):
    assert not kept(cfg, title)


@pytest.mark.parametrize("title", ["Business Analyst", "Technical Business Analyst", "Associate Business Analyst",
                                   "Data Analyst", "Business Analyst - Tech Background", "Presales Analysts"])
def test_analyst_roles_are_dropped(cfg, title):
    assert not kept(cfg, title)


def test_unwanted_title_is_dropped(cfg):
    assert not kept(cfg, "Sales Executive")


# --- edge cases: experience boundary and the Solution Architect rule --------------------------------------

def test_experience_at_the_limit_passes_and_above_it_fails(cfg):
    limit = cfg["filters"]["max_min_years"]
    assert kept(cfg, "Data Engineer", limit)
    assert not kept(cfg, "Data Engineer", limit + 1)


def test_plain_solution_architect_needs_entry_signal_or_stated_experience(cfg):
    assert not kept(cfg, "Solution Architect")              # no years, no entry word
    assert kept(cfg, "Solution Architect", 0)               # stated, within the limit
    assert kept(cfg, "Associate Solutions Architect")       # entry word
    assert kept(cfg, "AI Solution Architect I")             # trailing grade I


# --- clean(): date window, dedupe, internship flag --------------------------------------------------------

def test_clean_drops_old_postings_keeps_undated_and_dedupes(cfg):
    raw = job_frame(3)
    raw.loc[0, "date_posted"] = pd.Timestamp.now().normalize() - pd.Timedelta(days=cfg["search"]["days"] + 1)
    raw.loc[1, "date_posted"] = None                                      # undated: kept
    dup_url = raw.iloc[[2]]                                               # same URL again
    repost = raw.iloc[[2]].assign(job_url="https://example.com/repost")    # same title+company, new URL
    out, counts, rejected = filters.clean(pd.concat([raw, dup_url, repost], ignore_index=True), cfg)
    assert sorted(out["job_url"]) == sorted(raw["job_url"].iloc[1:])
    assert counts == {"raw": 5, "in date range": 4, "entry-level": 4, "deduped": 2}


def test_company_boards_use_their_own_age_window(cfg):
    days_ago = lambda n: pd.Timestamp.now().normalize() - pd.Timedelta(days=n)
    board_window = cfg["sources"]["greenhouse"]["max_age_days"]
    assert board_window > cfg["search"]["days"]
    raw = pd.concat([  # wanted titles, so only the date rule decides
        job_frame(1, "greenhouse", "AI Engineer").assign(date_posted=days_ago(board_window)),     # on the edge: kept
        job_frame(1, "greenhouse", "Data Engineer").assign(date_posted=days_ago(board_window + 1)),
        job_frame(1, "indeed", "Cloud Engineer").assign(date_posted=days_ago(board_window)),      # 14-day rule
    ], ignore_index=True)
    kept, _, rejected = filters.clean(raw, cfg)
    assert kept["title"].tolist() == ["AI Engineer 0"]
    reasons = dict(zip(rejected["title"], rejected["reason"]))
    assert reasons["Data Engineer 0"].endswith(f"older than {board_window} days")
    assert reasons["Cloud Engineer 0"].endswith(f"older than {cfg['search']['days']} days")


def test_sites_without_a_source_entry_use_the_default_window(cfg):
    raw = job_frame(1, "oldsource").assign(date_posted=pd.Timestamp.now().normalize() - pd.Timedelta(days=30))
    kept, _, rejected = filters.clean(raw, cfg)
    assert kept.empty and rejected["reason"].str.endswith(f"older than {cfg['search']['days']} days").all()


def test_clean_of_an_empty_frame(cfg):
    kept, counts, rejected = filters.clean(job_frame(0), cfg)
    assert kept.empty and rejected.empty and counts["raw"] == 0


def test_clean_flags_internships(cfg):
    raw = job_frame(1, prefix="AI/ML Intern")
    out, _, _ = filters.clean(raw, cfg)
    assert out["internship"].tolist() == [True]


def test_clean_does_not_modify_its_input(cfg):
    raw = job_frame(2)
    before = raw.copy()
    filters.clean(raw, cfg)
    pd.testing.assert_frame_equal(raw, before)
