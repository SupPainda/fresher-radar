"""Shared fixtures. Every test runs offline and writes only under tmp_path."""
import smtplib

import pytest
import requests

from jsearch import llm
from jsearch.config import load_config
from jsearch.sources import jobspy_site
from tests.helpers import NetworkBlocked


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Any real HTTP request or jobspy scrape fails the test loudly."""
    def refuse(*args, **kwargs):
        raise NetworkBlocked("tests must not touch the network")
    monkeypatch.setattr(requests.sessions.Session, "request", refuse)
    monkeypatch.setattr(jobspy_site, "scrape_jobs", refuse)
    monkeypatch.setattr(llm, "make_backend", refuse)         # any LLM: tests pass fake backends/clients instead
    monkeypatch.setattr(smtplib, "SMTP_SSL", refuse)         # any email: tests pass a fake SMTP instead
    monkeypatch.setattr(smtplib, "SMTP", refuse)


@pytest.fixture
def cfg(monkeypatch: pytest.MonkeyPatch, tmp_path) -> dict:
    """The real config.json, with outputs in tmp_path, zero delays and no proxies."""
    monkeypatch.setenv("JSEARCH_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("JSEARCH_PARALLEL", "true")
    monkeypatch.setenv("JSEARCH_PROXIES", "")
    c = load_config()
    c["search"]["delay_seconds"] = [0, 0]
    for src in c["sources"].values():
        src["delay_seconds"] = [0, 0]
    c["search"]["on_block"] = {"cooldown_minutes": 0, "give_up_after": 3}
    c["ats"]["detail_delay_seconds"] = [0, 0]
    c["email"]["enabled"] = False  # .env may hold real credentials; mailer tests turn it on with a fake SMTP
    return c
