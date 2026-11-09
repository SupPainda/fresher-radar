"""Offline fit score, and ranking inside a full (refilter) run."""
from argparse import Namespace

import pandas as pd
import pytest

import fetch_jobs
from jsearch.config import ROOT
from jsearch.profile import Profile, Resume
from jsearch.ranking import reserves, saturation, score, top_picks
from tests.helpers import job_frame


@pytest.fixture
def profile() -> Profile:
    return Profile(name="T", experience_years=0, target_roles=["AI Engineer", "Product Manager"],
                   skills={"python": 3, "rag": 3, "fastapi": 2, "sql": 1},
                   aliases={"rag": ["retrieval augmented generation"]},
                   resumes=[Resume(label="GenAI", focus=["llm", "rag", "langgraph"]),
                            Resume(label="PM", focus=["product", "roadmap", "stakeholder"])])


def jobs(*rows: tuple[str, str], skills: list | None = None) -> pd.DataFrame:
    df = pd.DataFrame({"title": [t for t, _ in rows], "description": [d for _, d in rows], "company": "Co"})
    return df.assign(skills=skills) if skills is not None else df


def one(df, profile, cfg) -> pd.Series:
    return score(df, profile, cfg).iloc[0]


# --- the score ---------------------------------------------------------------------------------------------

def test_title_match_counts_double_and_role_overlap_adds(cfg, profile):
    r = cfg["ranking"]
    row = one(jobs(("RAG Engineer", "")), profile, cfg)
    skill = min(1, 3 * r["title_multiplier"] / saturation(profile, cfg)) * r["skill_points"]
    role = 0.5 * r["role_points"]                      # "engineer" of {"ai", "engineer"}
    assert row["fit"] == round(skill + role)
    assert row["why"] == "rag | role: AI Engineer" and row["scored_by"] == "offline"


def test_alias_in_description_matches(cfg, profile):
    row = one(jobs(("Developer", "Experience with retrieval augmented generation pipelines")), profile, cfg)
    assert row["why"] == "rag" and row["fit"] > 0


def test_skills_match_whole_words_only(cfg, profile):
    row = one(jobs(("Storage Admin", "Manage storage arrays; nosql databases")), profile, cfg)
    assert row["fit"] == 0 and row["why"] == "no profile skills mentioned"


def test_skill_part_saturates_and_full_role_match_gives_100(cfg, profile):
    cfg["ranking"].update(skill_saturation_min=9, skill_saturation_share=0.1)   # cap = 9 = all weights
    row = one(jobs(("AI Engineer", "python rag fastapi sql " * 3)), profile, cfg)
    assert row["fit"] == 100
    assert row["why"].startswith("python, rag, fastapi, sql")       # heaviest first


def test_saturation_grows_with_the_profile(cfg, profile):
    cfg["ranking"].update(skill_saturation_min=12, skill_saturation_share=0.4)
    assert saturation(profile, cfg) == 12                            # 0.4 x 9 is below the minimum
    big = profile.model_copy(update={"skills": {f"s{i}": 3 for i in range(20)}})
    assert saturation(big, cfg) == pytest.approx(24)                 # 0.4 x 60


def test_title_focus_terms_outweigh_passing_mentions(cfg, profile):
    # body: two GenAI focus terms (2 points); title: two PM focus terms, counted double (4 points)
    out = score(jobs(("Product Roadmap Associate", "LLM and RAG features")), profile, cfg)
    assert out["send"].tolist() == ["PM"]


def test_naukri_skill_tags_are_used(cfg, profile):
    row = one(jobs(("Software Engineer", ""), skills=["Python, FastAPI, Docker"]), profile, cfg)
    assert row["why"].startswith("python, fastapi")


def test_send_picks_the_resume_with_most_focus_terms(cfg, profile):
    out = score(jobs(("Product Manager I", "Own the roadmap with stakeholder input; LLM features"),
                     ("LLM Engineer", "Build RAG with LangGraph")), profile, cfg)
    assert out["send"].tolist() == ["PM", "GenAI"]


def test_send_tie_goes_to_the_first_resume(cfg, profile):
    assert one(jobs(("Data Engineer", "SQL pipelines")), profile, cfg)["send"] == "GenAI"


def test_no_resumes_means_no_send(cfg, profile):
    assert one(jobs(("AI Engineer", "")), profile.model_copy(update={"resumes": []}), cfg)["send"] == ""


def test_score_does_not_modify_input_and_handles_empty(cfg, profile):
    df = jobs(("AI Engineer", "python"))
    before = df.copy()
    score(df, profile, cfg)
    pd.testing.assert_frame_equal(df, before)
    assert score(df.iloc[0:0], profile, cfg).empty


def test_missing_text_columns_are_treated_as_empty(cfg, profile):
    row = one(pd.DataFrame({"title": ["AI Engineer"], "company": ["Co"]}), profile, cfg)
    assert row["why"] == "no profile skills mentioned | role: AI Engineer"


# --- the top list: everything above the bar (within min/max), plus each resume's reserved slots ------------

def fixed(n: int) -> dict:
    """A top list of exactly n: a bar no job clears, min = max = n."""
    return {"min_fit": 101, "min_jobs": n, "max_jobs": n, "more_fit": 101}


def bar(min_fit: float, min_jobs: int = 0, max_jobs: int = 100) -> dict:
    return {"min_fit": min_fit, "min_jobs": min_jobs, "max_jobs": max_jobs, "more_fit": 0}


def scored(*rows: tuple[str, int, str]) -> pd.DataFrame:
    """(title, fit, send) rows, newest first by date so ties are predictable."""
    dates = pd.date_range(end="2026-11-09", periods=len(rows))[::-1]
    return pd.DataFrame({"title": [t for t, _, _ in rows], "fit": [f for _, f, _ in rows],
                         "send": [s for _, _, s in rows], "date_posted": dates})


SIX = scored(("AI 1", 95, "GenAI"), ("AI 2", 90, "GenAI"), ("AI 3", 85, "GenAI"), ("AI 4", 80, "GenAI"),
             ("PM 1", 50, "PM"), ("PM 2", 40, "PM"))


def test_without_reserves_the_top_list_is_the_best_fit():
    assert top_picks(SIX, fixed(3))["title"].tolist() == ["AI 1", "AI 2", "AI 3"]


def test_reserved_jobs_take_slots_from_the_bottom_of_the_rest():
    picks = top_picks(SIX, fixed(4), {"PM": (1, [])})
    assert picks["title"].tolist() == ["AI 1", "AI 2", "AI 3", "PM 1"]   # PM 1 replaces AI 4, list stays 4 long


def test_a_reserve_larger_than_the_jobs_available_takes_what_there_is():
    assert top_picks(SIX, fixed(4), {"PM": (5, [])})["title"].tolist() == ["AI 1", "AI 2", "PM 1", "PM 2"]


def test_reserves_never_make_the_list_longer_than_n():
    assert top_picks(SIX, fixed(2), {"PM": (5, [])})["title"].tolist() == ["PM 1", "PM 2"]


def test_jobs_already_in_the_top_count_towards_their_reserve():
    assert top_picks(SIX, fixed(3), {"GenAI": (2, [])})["title"].tolist() == ["AI 1", "AI 2", "AI 3"]


def test_a_reserved_slot_needs_a_focus_term_in_the_title():
    # "Data and AI Engineer" was sent to PM because its description talks product; it doesn't get a PM slot
    df = scored(("AI 1", 95, "GenAI"), ("AI 2", 90, "GenAI"), ("Data and AI Engineer", 62, "PM"),
                ("Associate Product Manager", 57, "PM"))
    assert top_picks(df, fixed(3), {"PM": (1, ["product", "roadmap"])})["title"].tolist() == \
        ["AI 1", "AI 2", "Associate Product Manager"]
    assert top_picks(df, fixed(3), {"PM": (1, [])})["title"].tolist() == ["AI 1", "AI 2", "Data and AI Engineer"]


def test_ties_go_to_the_newer_posting():
    tied = scored(("new", 70, "PM"), ("old", 70, "PM"))
    assert top_picks(tied, fixed(1))["title"].tolist() == ["new"]


@pytest.mark.parametrize("df, top, reserve", [
    (SIX.drop(columns="fit"), fixed(3), {}),    # not ranked (no profile)
    (SIX, fixed(0), {"PM": (2, [])}),           # max_jobs 0
    (SIX.iloc[0:0], bar(0), {}),                # no jobs
])
def test_no_fit_or_no_room_gives_an_empty_list(df, top, reserve):
    assert top_picks(df, top, reserve).empty


def test_every_job_above_the_bar_is_on_the_list():
    assert top_picks(SIX, bar(80))["title"].tolist() == ["AI 1", "AI 2", "AI 3", "AI 4"]   # 80 itself counts


def test_a_quiet_day_is_topped_up_to_min_jobs():
    assert top_picks(SIX, bar(90, min_jobs=4))["title"].tolist() == ["AI 1", "AI 2", "AI 3", "AI 4"]


def test_a_busy_day_is_capped_at_max_jobs():
    assert top_picks(SIX, bar(0, max_jobs=3))["title"].tolist() == ["AI 1", "AI 2", "AI 3"]


def test_reserved_jobs_come_on_top_of_the_bar():
    picks = top_picks(SIX, bar(80), {"PM": (1, [])})
    assert picks["title"].tolist() == ["AI 1", "AI 2", "AI 3", "AI 4", "PM 1"]   # nobody above the bar drops


def test_at_max_jobs_reserved_jobs_keep_their_place_and_the_lowest_of_the_rest_go():
    picks = top_picks(SIX, bar(80, max_jobs=4), {"PM": (1, [])})
    assert picks["title"].tolist() == ["AI 1", "AI 2", "AI 3", "PM 1"]


def test_an_unknown_label_or_no_send_column_changes_nothing():
    assert top_picks(SIX, fixed(3), {"DA": (2, [])})["title"].tolist() == ["AI 1", "AI 2", "AI 3"]
    assert top_picks(SIX.drop(columns="send"), fixed(3), {"PM": (2, [])})["title"].tolist() == ["AI 1", "AI 2", "AI 3"]


def test_top_picks_does_not_modify_its_input():
    before = SIX.copy()
    top_picks(SIX, fixed(4), {"PM": (2, ["pm"])})
    pd.testing.assert_frame_equal(SIX, before)


def test_reserves_come_from_the_profile(profile):
    assert reserves(profile) == {}
    profile.resumes[1].reserve = 4
    assert reserves(profile) == {"PM": (4, ["product", "roadmap", "stakeholder"])} and reserves(None) == {}


# --- inside a run ------------------------------------------------------------------------------------------

def refilter_run(cfg, raw: pd.DataFrame) -> pd.DataFrame:
    cfg["paths"]["raw"].parent.mkdir(parents=True, exist_ok=True)
    raw.to_pickle(cfg["paths"]["raw"])
    fetch_jobs.run(Namespace(test=False, refilter=True, sources=[]), cfg)
    return pd.read_csv(cfg["paths"]["csv"], encoding="utf-8-sig")


def three_jobs() -> pd.DataFrame:
    # only "AI Engineer" is a full target-role match in profile.example.yaml; the others half-match "engineer"
    return pd.concat([job_frame(1, prefix="Platform Engineer"), job_frame(1, prefix="AI Engineer"),
                      job_frame(1, prefix="Cloud Engineer")], ignore_index=True)


def test_run_ranks_by_fit_with_a_profile(cfg):
    cfg["ranking"]["llm"]["enabled"] = False
    (cfg["paths"]["profile"]).write_text((ROOT / "profile.example.yaml").read_text(encoding="utf-8"),
                                         encoding="utf-8")
    csv = refilter_run(cfg, three_jobs())
    assert csv["title"].iloc[0] == "AI Engineer 0"                  # the only target role
    assert csv["fit"].is_monotonic_decreasing
    assert set(csv["scored by"]) == {"offline"} and csv["send resume"].notna().all()


def test_run_without_a_profile_leaves_fit_empty(cfg, caplog):
    csv = refilter_run(cfg, three_jobs())
    assert len(csv) == 3 and csv["fit"].isna().all()
    assert "no profile at" in caplog.text


def test_broken_profile_stops_the_run_before_anything_else(cfg):
    cfg["paths"]["profile"].write_text("skills: [", encoding="utf-8")
    with pytest.raises(fetch_jobs.ProfileError):
        fetch_jobs.run(Namespace(test=False, refilter=True, sources=[]), cfg)
    assert not cfg["paths"]["csv"].exists()
