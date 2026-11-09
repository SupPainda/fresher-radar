"""What do the city searches add on top of the all-India + Remote searches?

Reads the raw dump (needs the search_title/search_location tags the runner adds) and, per job site,
counts the entry-level matches that only a city search found. If that share is small, the city
searches can be dropped for that site, which cuts its requests by 75%.

  python overlap.py
"""
import pandas as pd

from jsearch import filters
from jsearch.config import load_config

WIDE = {"India", "Remote"}  # location labels that cover the whole country


def main() -> None:
    cfg = load_config()
    raw = pd.read_pickle(cfg["paths"]["raw"])
    if "search_location" not in raw:
        print("This raw dump predates search tagging; run fetch_jobs.py once more first.")
        return
    raw = raw[raw["search_location"].notna()]  # job-site rows only (company boards are not searches)

    # same date and level filters as the real run, but no dedupe: duplicates are the point here
    raw["date_posted"] = pd.to_datetime(raw["date_posted"], errors="coerce")
    cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=cfg["search"]["days"])
    df = raw[raw["date_posted"].isna() | (raw["date_posted"] >= cutoff)].copy()
    df["min_years"] = df.apply(filters.min_years, axis=1, window=cfg["filters"]["experience_context_chars"])
    df = df[df.apply(filters.keep, axis=1, cfg=cfg)]

    for site, g in df.groupby("site"):
        wide = set(g.loc[g["search_location"].isin(WIDE), "job_url"])
        city = set(g.loc[~g["search_location"].isin(WIDE), "job_url"])
        only_city = city - wide
        total = len(wide | city)
        searches = raw[raw["site"] == site].groupby(["search_title", "search_location"]).ngroups
        print(f"\n{site}: {total} entry-level matches from {searches} searches with results")
        print(f"  found by India/Remote searches: {len(wide)}")
        print(f"  found ONLY by city searches:    {len(only_city)} ({len(only_city) / max(total, 1):.0%}) "
              f"<- lost if city searches are dropped")
        per_city = (g[g["job_url"].isin(only_city)].drop_duplicates("job_url")["search_location"]
                    .value_counts().to_dict())
        print(f"  city-only matches by city: {per_city}")


if __name__ == "__main__":
    main()
