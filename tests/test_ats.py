"""Company-board parsers (Greenhouse, Lever, Ashby, Workable, SmartRecruiters) and fetch_all, with small
synthetic payloads."""
from datetime import date, timedelta

import pytest

from jsearch.sources import ats

GREENHOUSE = {"jobs": [{
    "title": "Customer Solutions Engineer", "company_name": "Cloudflare",
    "location": {"name": "Bengaluru, India"}, "offices": [{"name": "Remote"}],
    "absolute_url": "https://boards.greenhouse.io/cf/1", "first_published": "2026-11-01T10:00:00Z",
    "content": "&lt;p&gt;2+ years of experience&lt;/p&gt;",
}]}
LEVER = [{
    "text": "Backend Engineer", "categories": {"location": "Bangalore", "allLocations": ["Bangalore", "Pune"]},
    "hostedUrl": "https://jobs.lever.co/zeta/1", "createdAt": 1793664000000,
    "descriptionPlain": "Build APIs.", "additionalPlain": "1 year experience", "workplaceType": "remote",
}]
ASHBY = {"jobs": [
    {"title": "AI Engineer", "location": "Mumbai", "secondaryLocations": [{"location": "Remote - India"}],
     "jobUrl": "https://jobs.ashbyhq.com/x/1", "publishedAt": "2026-11-02", "descriptionPlain": "LLMs",
     "isRemote": False, "workplaceType": "Remote", "isListed": True},
    {"title": "Hidden", "location": "Mumbai", "jobUrl": "u", "isListed": False},
]}


WORKABLE = {"name": "Hugging Face", "jobs": [
    {"title": "ML Engineer Intern", "url": "https://apply.workable.com/j/AB12", "published_on": "2026-11-04",
     "telecommuting": False, "city": "Bengaluru", "state": "Karnataka", "country": "India",
     "locations": [{"city": "Bengaluru", "region": "Karnataka", "country": "India", "hidden": False},
                   {"city": "Secret", "country": "India", "hidden": True}],
     "description": "<p>0-1 years of experience</p>"},
    {"title": "Remote Engineer", "shortlink": "https://apply.workable.com/j/CD34", "created_at": "2026-11-05",
     "telecommuting": True, "city": "Pune", "state": "", "country": "India"},
]}


def test_greenhouse_parsing():
    [row] = ats.parse_greenhouse("cloudflare", GREENHOUSE)
    assert row["company"] == "Cloudflare" and row["site"] == "greenhouse"
    assert row["location"] == "Bengaluru, India; Remote"
    assert row["date_posted"] == date(2026, 11, 1)
    assert row["description"] == "2+ years of experience"   # double-escaped HTML -> text
    assert row["is_remote"] is True


def test_lever_parsing_dedupes_locations_and_reads_epoch_millis():
    [row] = ats.parse_lever("zeta", LEVER)
    assert row["company"] == "zeta" and row["location"] == "Bangalore; Pune"
    assert row["date_posted"] == date(2026, 11, 3)
    assert row["description"] == "Build APIs. 1 year experience"
    assert row["is_remote"] is True


def test_ashby_skips_unlisted_and_reads_remote():
    rows = list(ats.parse_ashby("x", ASHBY))
    assert [r["title"] for r in rows] == ["AI Engineer"]
    assert rows[0]["location"] == "Mumbai; Remote - India" and rows[0]["is_remote"] is True


def test_workable_uses_the_account_name_and_skips_hidden_locations():
    first, second = ats.parse_workable("huggingface", WORKABLE)
    assert first["company"] == "Hugging Face" and first["site"] == "workable"
    assert first["location"] == "Bengaluru, Karnataka, India" and first["is_remote"] is False
    assert first["date_posted"] == date(2026, 11, 4) and first["description"] == "0-1 years of experience"
    assert first["job_url"] == "https://apply.workable.com/j/AB12"
    # no locations list: the posting's own city/country; no url: the shortlink
    assert second["location"] == "Pune, India" and second["job_url"] == "https://apply.workable.com/j/CD34"
    assert second["is_remote"] is True and second["date_posted"] == date(2026, 11, 5)


@pytest.mark.parametrize("value, expected", [
    (None, None), ("", None), ("not a date", None), (1793664000000, date(2026, 11, 3)),
    ("2026-11-01T23:30:00Z", date(2026, 11, 1)),
])
def test_to_date(value, expected):
    assert ats.to_date(value) == expected


def test_empty_payloads_give_no_rows():
    assert list(ats.parse_greenhouse("s", {})) == []
    assert list(ats.parse_lever("s", [])) == []
    assert list(ats.parse_ashby("s", {"jobs": []})) == []
    assert list(ats.parse_workable("s", {})) == []


# --- SmartRecruiters: paged list, details only for promising postings --------------------------------------

def sr_posting(pid: str, title: str, days_ago: int = 1, location: str = "Bengaluru, KA, India") -> dict:
    return {"id": pid, "name": title, "company": {"name": "Bosch Group"},
            "releasedDate": f"{date.today() - timedelta(days=days_ago)}T10:00:00.000Z",
            "location": {"fullLocation": location, "remote": False},
            "ref": f"https://api.smartrecruiters.com/v1/companies/BoschGroup/postings/{pid}"}


def sr_detail(pid: str) -> dict:
    return {"postingUrl": f"https://jobs.smartrecruiters.com/BoschGroup/{pid}-nice-title",
            "jobAd": {"sections": {"jobDescription": {"title": "Job", "text": "<p>Build <b>LLM</b> apps</p>"},
                                   "qualifications": {"title": "Q", "text": "0-1 years of experience"}}}}


class FakeHTTP:
    """requests.get by exact URL; records every URL asked for."""

    def __init__(self, pages: dict):
        self.pages, self.urls = pages, []

    def get(self, url, timeout):
        self.urls.append(url)
        payload = self.pages.get(url)

        class Resp:
            def raise_for_status(self):
                if payload is None:
                    raise ats.requests.HTTPError(f"404 for {url}")

            def json(self):
                return payload
        return Resp()


def sr_pages(cfg, postings: list[dict], page_size: int = 2) -> dict:
    base = cfg["ats"]["endpoints"]["smartrecruiters"].format(slug="BoschGroup")
    pages = {f"{base}&offset={i}": {"totalFound": len(postings), "content": postings[i:i + page_size]}
             for i in range(0, max(len(postings), 1), page_size)}
    pages.update({p["ref"]: sr_detail(p["id"]) for p in postings})
    return pages


def test_smartrecruiters_pages_and_fetches_details_only_for_promising_postings(cfg, monkeypatch):
    postings = [sr_posting("1", "AI Engineer"), sr_posting("2", "Accountant"),
                sr_posting("3", "AI Engineer", days_ago=400), sr_posting("4", "Data Engineer")]
    http = FakeHTTP(sr_pages(cfg, postings))
    monkeypatch.setattr(ats.requests, "get", http.get)
    rows, fetched = ats.fetch_smartrecruiters("BoschGroup", cfg["sources"]["smartrecruiters"], cfg)
    base = cfg["ats"]["endpoints"]["smartrecruiters"].format(slug="BoschGroup")
    assert http.urls[:2] == [f"{base}&offset=0", f"{base}&offset=2"]          # two pages of two
    assert fetched == 2 and [u for u in http.urls[2:]] == [postings[0]["ref"], postings[3]["ref"]]
    by_id = {r["job_url"].rsplit("/", 1)[-1]: r for r in rows}
    first = by_id["1-nice-title"]
    assert first["description"] == "Build LLM apps 0-1 years of experience" and first["company"] == "Bosch Group"
    assert first["location"] == "Bengaluru, KA, India" and first["site"] == "smartrecruiters"
    # not fetched: kept with the list fields (title, date, location) and a constructed URL, so filters see them
    assert by_id["2"]["description"] == "" and by_id["2"]["job_url"] == "https://jobs.smartrecruiters.com/BoschGroup/2"
    assert len(rows) == 4


def test_smartrecruiters_details_are_capped_per_board(cfg, monkeypatch):
    cfg["ats"]["max_details_per_board"] = 1
    postings = [sr_posting("1", "AI Engineer"), sr_posting("2", "Data Engineer")]
    monkeypatch.setattr(ats.requests, "get", FakeHTTP(sr_pages(cfg, postings)).get)
    rows, fetched = ats.fetch_smartrecruiters("BoschGroup", cfg["sources"]["smartrecruiters"], cfg)
    assert fetched == 1 and [r["description"] != "" for r in rows] == [True, False]


def test_smartrecruiters_empty_board_is_one_request(cfg, monkeypatch):
    http = FakeHTTP(sr_pages(cfg, []))
    monkeypatch.setattr(ats.requests, "get", http.get)
    assert ats.fetch_smartrecruiters("BoschGroup", cfg["sources"]["smartrecruiters"], cfg) == ([], 0)
    assert len(http.urls) == 1


def test_smartrecruiters_fetch_reports_details_and_errors_propagate(cfg, monkeypatch):
    pages = sr_pages(cfg, [sr_posting("1", "AI Engineer", location="Berlin, Germany"), sr_posting("2", "AI Engineer")])
    monkeypatch.setattr(ats.requests, "get", FakeHTTP(pages).get)
    [unit] = ats.units("smartrecruiters", {"boards": ["BoschGroup"]}, cfg, test=False)
    result = ats.fetch("smartrecruiters", unit, cfg["sources"]["smartrecruiters"], cfg)
    assert result.note == "2 postings, 1 in location, 1 with details"   # Berlin: no detail request, dropped
    monkeypatch.setattr(ats.requests, "get", FakeHTTP({}).get)
    with pytest.raises(ats.requests.HTTPError):  # the runner records it as that board's error
        ats.fetch("smartrecruiters", unit, cfg["sources"]["smartrecruiters"], cfg)


# --- source contract: units() and fetch() ------------------------------------------------------------------

def fake_fetch_board(provider, slug, ats_cfg):
    if slug == "broken":
        raise ConnectionError("404")
    payload = {"greenhouse": GREENHOUSE, "lever": LEVER}[provider]
    rows = list(ats.PARSERS[provider](slug, payload))
    rows.append(rows[0] | {"location": "Berlin, Germany", "job_url": "elsewhere"})  # outside India
    return rows


def test_units_are_one_per_board_and_test_mode_takes_a_few(cfg):
    src = {"type": "ats", "boards": ["a", "b", "c", "d"]}
    assert [u.key for u in ats.units("greenhouse", src, cfg, test=False)] == [f"greenhouse|{s}" for s in "abcd"]
    limit = cfg["search"]["test"]["boards_per_provider"]
    assert len(ats.units("greenhouse", src, cfg, test=True)) == limit
    assert ats.units("greenhouse", {"type": "ats", "boards": []}, cfg, test=False) == []


def test_fetch_keeps_only_wanted_locations(cfg, monkeypatch):
    monkeypatch.setattr(ats, "fetch_board", fake_fetch_board)
    [unit] = ats.units("greenhouse", {"boards": ["cloudflare"]}, cfg, test=False)
    result = ats.fetch("greenhouse", unit, cfg["sources"]["greenhouse"], cfg)
    assert result.df["job_url"].tolist() == ["https://boards.greenhouse.io/cf/1"]
    assert list(result.df.columns) == ats.COLUMNS
    assert result.note == "2 postings, 1 in location" and result.refusal is None


def test_board_errors_are_recorded_by_the_runner(cfg, monkeypatch):
    from jsearch import runner
    from tests.test_runner import FakeStop
    monkeypatch.setattr(ats, "fetch_board", fake_fetch_board)
    src = {"type": "ats", "boards": ["cloudflare", "broken"]}
    frames, stats, errors = runner.run_source("greenhouse", ats.units("greenhouse", src, cfg, False),
                                              ats.fetch, src, cfg, FakeStop())
    assert stats["greenhouse"] == {"ok": 1, "empty": 0, "error": 1, "blocked": 0, "rows": 1}
    assert errors == ["greenhouse | broken: ConnectionError: 404"]


def test_fetch_board_uses_the_configured_endpoint(cfg, monkeypatch):
    seen = {}

    class Resp:
        def raise_for_status(self): pass
        def json(self): return LEVER

    monkeypatch.setattr(ats.requests, "get", lambda url, timeout: seen.update(url=url, timeout=timeout) or Resp())
    rows = ats.fetch_board("lever", "zeta", cfg["ats"])
    assert seen == {"url": cfg["ats"]["endpoints"]["lever"].format(slug="zeta"),
                    "timeout": cfg["ats"]["timeout_seconds"]}
    assert [r["title"] for r in rows] == ["Backend Engineer"]
