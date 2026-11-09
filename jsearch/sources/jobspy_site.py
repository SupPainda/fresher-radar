"""Job sites scraped through python-jobspy (Naukri, Indeed, ...): one unit per title x location search."""
import pandas as pd
from jobspy import scrape_jobs

from jsearch.sources import Result, Unit


def units(name: str, src: dict, cfg: dict, test: bool) -> list[Unit]:
    s = cfg["search"]
    if test:
        titles = [s["test"]["title"]]
        locations = [loc for loc in cfg["locations"] if loc["label"] == s["test"]["location"]]
    else:
        titles, locations = cfg["titles"]["search"], cfg["locations"]
    return [Unit(key=f"{name}|{title}|{loc['label']}", label=f"{title} / {loc['label']}",
                 args={"title": title, "location": loc},
                 tags={"search_title": title, "search_location": loc["label"]})  # for overlap.py
            for title in titles for loc in locations]


def fetch(name: str, unit: Unit, src: dict, cfg: dict) -> Result:
    s, loc = cfg["search"], unit.args["location"]
    defaults = {"results_wanted": s["results_per_search"], "hours_old": s["days"] * 24}
    # a source's own options override the defaults (e.g. naukri results_wanted: 20 = one page)
    df = scrape_jobs(site_name=[name], search_term=unit.args["title"], location=loc["query"],
                     is_remote=loc["remote"], verbose=0, proxies=cfg["env"]["proxies"],
                     **(defaults | src.get("options", {})))
    if df is None:
        df = pd.DataFrame()
    # jobspy does not raise when a site refuses it (e.g. Naukri 406 = rate limited): it logs,
    # stops paging and returns what it had, noting the reason in df.attrs["errors"]
    return Result(df=df, refusal=(df.attrs.get("errors") or {}).get(name))
