"""Public job-board APIs of Greenhouse, Lever, Ashby, Workable and SmartRecruiters, normalised to the jobspy
column layout.

Each provider returns every open posting of a company; location filtering to the wanted
geography happens here, title/level filtering happens in jsearch.filters like for any source.
One config source per provider (type "ats"); each board is one unit of work.

SmartRecruiters lists postings without their text, so each one needs its own request; those are made only
for postings that are recent and pass the title rules, at most ats.max_details_per_board per board, paced
by ats.detail_delay_seconds.
"""
import html
import random
import re
import time
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from typing import Iterator

import pandas as pd
import requests

from jsearch.filters import title_reject_reason
from jsearch.sources import Result, Unit

COLUMNS = ["title", "company", "location", "job_url", "date_posted", "site",
           "description", "is_remote", "experience_range"]
TAGS = re.compile(r"<[^>]+>")


def to_text(markup: str | None) -> str:
    """HTML (possibly entity-escaped) -> plain text, for the experience scan."""
    return re.sub(r"\s+", " ", html.unescape(TAGS.sub(" ", html.unescape(markup or "")))).strip()


def to_date(value: str | int | float | None) -> date | None:
    """ISO string or epoch-millis -> date, None when missing/unparseable."""
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value / 1000, tz=timezone.utc).date()
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    return None if pd.isna(ts) else ts.date()


def _row(site: str, company: str, title: str, locations: list, url: str, posted: date | None,
         description: str, remote: bool) -> dict:
    return {"title": title, "company": company, "location": "; ".join(dict.fromkeys(l for l in locations if l)),
            "job_url": url, "date_posted": posted, "site": site, "description": description,
            "is_remote": bool(remote), "experience_range": None}


def parse_greenhouse(slug: str, payload: dict) -> Iterator[dict]:
    for j in payload.get("jobs", []):
        locs = [(j.get("location") or {}).get("name")] + [o.get("location") or o.get("name") for o in j.get("offices", [])]
        yield _row("greenhouse", j.get("company_name") or slug, j.get("title"), locs, j.get("absolute_url"),
                   to_date(j.get("first_published") or j.get("updated_at")), to_text(j.get("content")),
                   "remote" in " ".join(l for l in locs if l).lower())


def parse_lever(slug: str, payload: list) -> Iterator[dict]:
    for j in payload:
        cat = j.get("categories") or {}
        locs = [cat.get("location")] + (cat.get("allLocations") or [])
        text = " ".join(filter(None, [j.get("descriptionPlain"), j.get("additionalPlain")]))
        yield _row("lever", slug, j.get("text"), locs, j.get("hostedUrl"), to_date(j.get("createdAt")),
                   text, j.get("workplaceType") == "remote")


def parse_ashby(slug: str, payload: dict) -> Iterator[dict]:
    for j in payload.get("jobs", []):
        if j.get("isListed") is False:
            continue
        locs = [j.get("location")] + [s.get("location") for s in j.get("secondaryLocations", [])]
        yield _row("ashby", slug, j.get("title"), locs, j.get("jobUrl"), to_date(j.get("publishedAt")),
                   j.get("descriptionPlain") or to_text(j.get("descriptionHtml")),
                   j.get("isRemote") or j.get("workplaceType") == "Remote")


def parse_workable(slug: str, payload: dict) -> Iterator[dict]:
    company = payload.get("name") or slug
    for j in payload.get("jobs", []):
        locs = [", ".join(filter(None, [l.get("city"), l.get("region"), l.get("country")]))
                for l in j.get("locations") or [] if not l.get("hidden")]
        locs = locs or [", ".join(filter(None, [j.get("city"), j.get("state"), j.get("country")]))]
        yield _row("workable", company, j.get("title"), locs, j.get("url") or j.get("shortlink"),
                   to_date(j.get("published_on") or j.get("created_at")), to_text(j.get("description")),
                   j.get("telecommuting"))


def smartrecruiters_row(slug: str, posting: dict, detail: dict | None = None) -> dict:
    """One SmartRecruiters posting from its list entry; the detail (when fetched) adds the text and URL."""
    loc = posting.get("location") or {}
    sections = ((detail or {}).get("jobAd") or {}).get("sections") or {}
    text = " ".join(to_text(s.get("text")) for s in sections.values() if isinstance(s, dict))
    url = (detail or {}).get("postingUrl") or f"https://jobs.smartrecruiters.com/{slug}/{posting.get('id')}"
    return _row("smartrecruiters", (posting.get("company") or {}).get("name") or slug, posting.get("name"),
                [loc.get("fullLocation") or ", ".join(filter(None, [loc.get("city"), loc.get("country")]))],
                url, to_date(posting.get("releasedDate")), text, loc.get("remote"))


PARSERS = {"greenhouse": parse_greenhouse, "lever": parse_lever, "ashby": parse_ashby, "workable": parse_workable}
PROVIDERS = {*PARSERS, "smartrecruiters"}   # smartrecruiters: fetch_smartrecruiters (paged, with details)


def _get_json(url: str, ats_cfg: dict) -> dict | list:
    resp = requests.get(url, timeout=ats_cfg["timeout_seconds"])
    resp.raise_for_status()
    return resp.json()


def fetch_board(provider: str, slug: str, ats_cfg: dict) -> list[dict]:
    url = ats_cfg["endpoints"][provider].format(slug=slug)
    return list(PARSERS[provider](slug, _get_json(url, ats_cfg)))


def fetch_smartrecruiters(slug: str, src: dict, cfg: dict) -> tuple[list[dict], int]:
    """Every listed posting (paged), with details for the promising ones. Returns (rows, details fetched)."""
    ats_cfg = cfg["ats"]
    base = ats_cfg["endpoints"]["smartrecruiters"].format(slug=slug)
    listed: list[dict] = []
    while True:
        page = _get_json(f"{base}{'&' if '?' in base else '?'}offset={len(listed)}", ats_cfg)
        content = page.get("content") or []
        listed += content
        if not content or len(listed) >= page.get("totalFound", 0):
            break
    cutoff = date.today() - timedelta(days=src.get("max_age_days") or cfg["search"]["days"])
    match = _location_rx(tuple(ats_cfg["location_match"]))
    rows, fetched = [], 0
    for p in listed:
        row = smartrecruiters_row(slug, p)
        promising = ((row["date_posted"] is None or row["date_posted"] >= cutoff)
                     and match.search(row["location"] or "") and title_reject_reason(str(row["title"]), cfg) is None)
        if promising and fetched < ats_cfg["max_details_per_board"]:
            if fetched:
                time.sleep(random.uniform(*ats_cfg["detail_delay_seconds"]))
            ref = p.get("ref") or f"{base.split('?')[0]}/{p.get('id')}"
            row = smartrecruiters_row(slug, p, _get_json(ref, ats_cfg))
            fetched += 1
        rows.append(row)
    return rows, fetched


@lru_cache(maxsize=None)
def _location_rx(terms: tuple[str, ...]) -> re.Pattern:
    return re.compile("|".join(re.escape(t) for t in terms), re.I)


# --- source contract ---------------------------------------------------------------------------------------

def units(name: str, src: dict, cfg: dict, test: bool) -> list[Unit]:
    """One unit per company board; --test takes only the first few."""
    boards = src["boards"][:cfg["search"]["test"]["boards_per_provider"]] if test else src["boards"]
    return [Unit(key=f"{name}|{slug}", label=slug, args={"slug": slug}) for slug in boards]


def fetch(name: str, unit: Unit, src: dict, cfg: dict) -> Result:
    """Every open posting of one board, kept only when a location matches ats.location_match."""
    note = ""
    if name == "smartrecruiters":
        postings, fetched = fetch_smartrecruiters(unit.args["slug"], src, cfg)
        note = f", {fetched} with details"
    else:
        postings = fetch_board(name, unit.args["slug"], cfg["ats"])
    match = _location_rx(tuple(cfg["ats"]["location_match"]))
    local = [r for r in postings if match.search(r["location"] or "")]
    return Result(df=pd.DataFrame(local, columns=COLUMNS),
                  note=f"{len(postings)} postings, {len(local)} in location{note}")
