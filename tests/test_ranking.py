"""Offline fit score, and ranking inside a full (refilter) run."""
from argparse import Namespace

import pandas as pd
import pytest

import fetch_jobs
from jsearch.config import ROOT
from jsearch.profile import Profile, Resume
from jsearch.ranking import saturation, score
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
