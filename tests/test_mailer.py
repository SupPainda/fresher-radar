"""The top-jobs email: settings, which jobs, the message, sending, and when a run sends. SMTP is always fake."""
import logging
import smtplib
from argparse import Namespace
from pathlib import Path

import pandas as pd
import pytest

import fetch_jobs
from jsearch import mailer
from tests.helpers import job_frame

ROOT = Path(__file__).resolve().parent.parent
ENV = {"EMAIL_FROM": "me@gmail.com", "EMAIL_APP_PASSWORD": "abcd efgh ijkl mnop", "EMAIL_TO": "me@gmail.com"}


class FakeSMTP:
    """smtplib.SMTP_SSL stand-in: records the connection, login and messages; can fail at login or send."""
    instances: list["FakeSMTP"] = []

    def __init__(self, host, port, timeout=None, context=None, fail_login=None, fail_send=None):
        self.host, self.port, self.timeout, self.context = host, port, timeout, context
        self.fail_login, self.fail_send = fail_login, fail_send
        self.logins, self.sent = [], []
        FakeSMTP.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def login(self, user, password):
        if self.fail_login:
            raise self.fail_login
        self.logins.append((user, password))

    def send_message(self, msg, to_addrs=None):
        if self.fail_send:
            raise self.fail_send
        self.sent.append((msg, to_addrs))


def factory(**fail):
    FakeSMTP.instances = []
    return lambda *a, **kw: FakeSMTP(*a, **kw, **fail)


@pytest.fixture
def ecfg(cfg):
    cfg["email"]["enabled"] = True
    return cfg


def ranked(n: int = 4) -> pd.DataFrame:
    df = job_frame(n)
    return df.assign(fit=[50 + 10 * i for i in range(n)], why=[f"reason {i}" for i in range(n)],
                     send=["GenAI"] * n, new=[i % 2 == 0 for i in range(n)])


# --- settings ----------------------------------------------------------------------------------------------

def test_settings_strip_the_app_password_spaces():
    s = mailer.settings(ENV)
    assert (s.sender, s.password, s.to) == ("me@gmail.com", "abcdefghijklmnop", ["me@gmail.com"])


def test_recipients_default_to_the_sender_and_accept_a_list():
    assert mailer.settings(ENV | {"EMAIL_TO": ""}).to == ["me@gmail.com"]
    assert mailer.settings(ENV | {"EMAIL_TO": "a@x.com, b@y.com"}).to == ["a@x.com", "b@y.com"]


@pytest.mark.parametrize("missing", ["EMAIL_FROM", "EMAIL_APP_PASSWORD"])
def test_incomplete_settings_say_what_is_missing(missing):
    with pytest.raises(mailer.MailUnavailable, match=missing):
        mailer.settings({k: v for k, v in ENV.items() if k != missing})


def test_settings_never_show_the_password():
    assert "abcdefghijklmnop" not in repr(mailer.settings(ENV))


# --- which jobs, and the message ---------------------------------------------------------------------------

def test_top_jobs_are_the_best_fit_first():
    top = mailer.top_jobs(ranked(4), 2)
    assert top["fit"].tolist() == [80, 70]


def test_no_fit_column_or_zero_n_means_no_jobs():
    assert mailer.top_jobs(job_frame(3), 30).empty
    assert mailer.top_jobs(ranked(3), 0).empty


def test_message_lists_each_job_with_its_link_and_marks_new_ones():
    top = mailer.top_jobs(ranked(4), 4)
    msg = mailer.build_message(top, "me@gmail.com", ["me@gmail.com"], "9 Nov")
    assert msg["Subject"] == "JSearch: top 4 jobs, 2 new (9 Nov)"
    text = msg.get_body(("plain",)).get_content()
    page = msg.get_body(("html",)).get_content()
    for _, r in top.iterrows():
        assert r["job_url"] in text and f'href="{r["job_url"]}"' in page
    assert text.index("[80]") < text.index("[50]")
    assert page.count(">NEW</b>") == 2 and "NEW [70]" in text


def test_message_escapes_html_in_job_text():
    df = ranked(1).assign(title=["<script>x</script> & Co"])
    page = mailer.build_message(df, "a@b", ["a@b"], "d").get_body(("html",)).get_content()
    assert "<script>" not in page and "&lt;script&gt;" in page


def test_message_without_new_jobs_has_a_plain_subject():
    msg = mailer.build_message(ranked(2).assign(new=False), "a@b", ["a@b"], "d")
    assert msg["Subject"] == "JSearch: top 2 jobs (d)"


# --- sending -----------------------------------------------------------------------------------------------

def test_send_logs_in_over_tls_and_sends(ecfg):
    s = mailer.settings(ENV)
    msg = mailer.build_message(ranked(1), s.sender, s.to, "d")
    mailer.send(msg, s, ecfg, factory())
    [smtp] = FakeSMTP.instances
    assert (smtp.host, smtp.port, smtp.timeout) == ("smtp.gmail.com", 465, ecfg["email"]["timeout_seconds"])
    assert smtp.context is not None and smtp.logins == [("me@gmail.com", "abcdefghijklmnop")]
    assert smtp.sent == [(msg, ["me@gmail.com"])]


@pytest.mark.parametrize("fail, message", [
    ({"fail_login": smtplib.SMTPAuthenticationError(535, b"bad")}, "app password"),
    ({"fail_send": smtplib.SMTPRecipientsRefused({})}, "refused the email"),
    ({"fail_login": TimeoutError("timed out")}, "could not reach smtp.gmail.com:465"),
])
def test_smtp_problems_become_mail_unavailable(ecfg, fail, message):
    s = mailer.settings(ENV)
    with pytest.raises(mailer.MailUnavailable, match=message):
        mailer.send(mailer.build_message(ranked(1), s.sender, s.to, "d"), s, ecfg, factory(**fail))


def test_send_top_sends_the_top_n(ecfg, caplog):
    ecfg["ranking"]["llm"]["top_n"] = 3
    with caplog.at_level(logging.INFO, logger="jsearch"):
        assert mailer.send_top(ranked(5), ecfg, "d", ENV, factory()) is True
    [smtp] = FakeSMTP.instances
    assert smtp.sent[0][0]["Subject"].startswith("JSearch: top 3 jobs")
    assert "emailed the top 3 jobs to me@gmail.com" in caplog.text


def test_send_top_includes_reserved_jobs(ecfg):
    ecfg["ranking"]["llm"]["top_n"] = 2
    df = ranked(4).assign(send=["GenAI", "GenAI", "GenAI", "PM"], fit=[90, 80, 70, 30])
    mailer.send_top(df, ecfg, "d", ENV, factory(), reserve={"PM": (1, [])})
    text = FakeSMTP.instances[0].sent[0][0].get_body(("plain",)).get_content()
    assert "[90]" in text and "[30]" in text and "[80]" not in text


@pytest.mark.parametrize("env, df, why", [
    ({}, ranked(2), "EMAIL_FROM and EMAIL_APP_PASSWORD not set"),
    (ENV, job_frame(2), "no ranked jobs"),
])
def test_send_top_skips_with_one_line_and_never_raises(ecfg, caplog, env, df, why):
    with caplog.at_level(logging.INFO, logger="jsearch"):
        assert mailer.send_top(df, ecfg, "d", env, factory()) is False
    assert why in caplog.text and FakeSMTP.instances == []


def test_smtp_failure_is_logged_not_raised(ecfg, caplog):
    with caplog.at_level(logging.INFO, logger="jsearch"):
        assert mailer.send_top(ranked(2), ecfg, "d", ENV,
                               factory(fail_login=smtplib.SMTPAuthenticationError(535, b"no"))) is False
    assert "email skipped" in caplog.text and "app password" in caplog.text


def test_disabled_in_config_sends_nothing(cfg):
    assert mailer.send_top(ranked(2), cfg, "d", ENV, factory()) is False and FakeSMTP.instances == []


# --- when a run sends --------------------------------------------------------------------------------------

@pytest.fixture
def sends(ecfg, monkeypatch):
    """Runs with a profile and the LLM off; records send_top calls instead of sending."""
    ecfg["ranking"]["llm"]["enabled"] = False
    ecfg["paths"]["profile"].write_text((ROOT / "profile.example.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    ecfg["paths"]["raw"].parent.mkdir(parents=True, exist_ok=True)
    job_frame(3, prefix="AI Engineer").to_pickle(ecfg["paths"]["raw"])
    calls = []
    monkeypatch.setattr(mailer, "send_top", lambda df, cfg, when, **kw: calls.append((df, kw)) or True)
    return calls


def test_refilter_does_not_email_unless_asked(ecfg, sends):
    fetch_jobs.run(Namespace(test=False, refilter=True, sources=[]), ecfg)
    assert sends == []
    fetch_jobs.run(Namespace(test=False, refilter=True, sources=[], email=True), ecfg)
    [(df, kw)] = sends
    assert len(df) == 3 and df["fit"].notna().all() and "job_url" in df
    # from profile.example.yaml's PM resume: 3 slots, and its focus terms for the title check
    assert kw["reserve"] == {"PM": (3, ["product", "roadmap", "stakeholder", "user research", "metrics"])}
