"""Test helpers shared by conftest and the test modules (one import path, so one copy of each class)."""
import pandas as pd


class NetworkBlocked(RuntimeError):
    pass


def job_frame(n: int, site: str = "indeed", prefix: str = "Data Engineer") -> pd.DataFrame:
    """n distinct postings in the jobspy column layout."""
    return pd.DataFrame({
        "title": [f"{prefix} {i}" for i in range(n)],
        "company": [f"Co{i}" for i in range(n)],
        "location": "Pune, India",
        "job_url": [f"https://example.com/{site}/{prefix.replace(' ', '-')}/{i}" for i in range(n)],
        "date_posted": pd.Timestamp.now().normalize(),
        "site": site,
        "description": "",
        "is_remote": False,
        "experience_range": None,
    })
