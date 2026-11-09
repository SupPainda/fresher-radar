"""The suite's own safety net: no test may reach a job site."""
import pytest
import requests

from jsearch.sources import ats, jobspy_site
from tests.helpers import NetworkBlocked


def test_http_requests_are_refused():
    with pytest.raises(NetworkBlocked):
        requests.get("https://www.naukri.com")


def test_jobspy_scrapes_are_refused(cfg):
    [unit, *_] = jobspy_site.units("naukri", cfg["sources"]["naukri"], cfg, test=True)
    with pytest.raises(NetworkBlocked):
        jobspy_site.fetch("naukri", unit, cfg["sources"]["naukri"], cfg)


def test_company_board_fetches_are_refused(cfg):
    [unit, *_] = ats.units("greenhouse", cfg["sources"]["greenhouse"], cfg, test=True)
    with pytest.raises(NetworkBlocked):
        ats.fetch("greenhouse", unit, cfg["sources"]["greenhouse"], cfg)
