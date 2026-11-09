"""Reading the minimum years of experience from a posting."""
import pytest

from jsearch import filters

WINDOW = 60  # config.json filters.experience_context_chars


def years(title: str = "", desc: str = "", rng=None) -> int | None:
    return filters.min_years({"title": title, "description": desc, "experience_range": rng}, WINDOW)


# --- happy path -------------------------------------------------------------------------------------------

@pytest.mark.parametrize("title, expected", [
    ("Data Engineer ( 2 to 5 Years)", 2),
    ("AWS Cloud Engineer(Exp 5 yrs)", 5),
    ("Backend Engineer 1-3 years", 1),
])
def test_years_in_title(title, expected):
    assert years(title=title) == expected


@pytest.mark.parametrize("desc, expected", [
    ("You need 3+ years of experience in Python.", 3),
    ("Requires 10+ year experience leading teams", 10),
    ("Experience: 1 yr minimum", 1),
    ("Experience required: 1.5 years", 1),
    ("2 to 4 years of relevant experience", 2),
])
def test_years_in_description(desc, expected):
    assert years(desc=desc) == expected


def test_naukri_experience_range_wins_over_title_and_description():
    assert years(title="Engineer 5 years", desc="8 years experience", rng="1-6 Yrs") == 1


def test_lowest_requirement_wins_when_several_are_stated():
    desc = "3+ years experience with Python. Overall 5+ years of experience required."
    assert years(desc=desc) == 3


# --- edge cases -------------------------------------------------------------------------------------------

def test_figures_without_experience_context_are_ignored():
    assert years(desc="We were founded 25 years ago and serve 40 countries.") is None


def test_context_check_picks_the_experience_figure_only():
    desc = "Founded 25 years ago. " + "x" * 100 + " You need 3+ years of experience in Python."
    assert years(desc=desc) == 3


def test_calendar_years_are_not_years_of_experience():
    assert years(desc="Experience: celebrating 2024 years of service") is None


def test_text_beyond_the_old_3000_char_cutoff_is_scanned():
    assert years(desc="x" * 5000 + " minimum 4 years of relevant experience") == 4


@pytest.mark.parametrize("desc", ["", None])
def test_empty_description_gives_none(desc):
    assert years(desc=desc) is None


def test_naukri_range_without_unit_uses_first_number():
    assert years(rng="0") == 0


def test_window_controls_how_far_the_context_word_may_be():
    desc = "4 years" + " " * 80 + "experience"
    assert filters.years_in_text(desc, window=60) is None
    assert filters.years_in_text(desc, window=100) == 4
