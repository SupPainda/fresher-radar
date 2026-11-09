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

def fixed(n: int) -> dict:
    """A top list of exactly n: a bar no job clears, min = max = n, no "more" line."""
    return {"min_fit": 101, "min_jobs": n, "max_jobs": n, "more_fit": 101}


def test_top_jobs_are_the_best_fit_first():
    top = mailer.top_jobs(ranked(4), fixed(2))
    assert top["fit"].tolist() == [80, 70]


def test_no_fit_column_or_no_room_means_no_jobs():
    assert mailer.top_jobs(job_frame(3), fixed(30)).empty
    assert mailer.top_jobs(ranked(3), fixed(0)).empty


def test_more_good_fits_counts_the_rest_above_more_fit():
    df = ranked(5)                                     # fits 50, 60, 70, 80, 90
    top = mailer.top_jobs(df, fixed(2))                # 90, 80
    assert mailer.more_good_fits(df, top, 60) == 2     # 70 and 60; 50 is below
    assert mailer.more_good_fits(df, top, 95) == 0
    assert mailer.more_good_fits(job_frame(2), top, 0) == 0   # unranked


def test_new_jobs_come_first_then_the_rest_each_with_its_link():
    top = mailer.top_jobs(ranked(4), fixed(4))          # fit 80, 70 (new), 60, 50 (new)
    msg = mailer.build_message(top, "me@gmail.com", ["me@gmail.com"], "9 Nov")
    assert msg["Subject"] == "fresher-radar: 2 new, top 4 jobs (9 Nov)"
    text = msg.get_body(("plain",)).get_content()
    page = msg.get_body(("html",)).get_content()
    for _, r in top.iterrows():
        assert r["job_url"] in text and f'href="{r["job_url"]}"' in page
    assert (text.index("New today (2)") < text.index("1. [70]") < text.index("2. [50]")
            < text.index("Also on your top list (2)") < text.index("3. [80]") < text.index("4. [60]"))
    assert "New today (2)</h3>" in page and "Also on your top list (2)</h3>" in page


def test_all_new_has_no_also_section():
    msg = mailer.build_message(ranked(2).assign(new=True), "a@b", ["a@b"], "d")
    assert "Also on your top list" not in msg.get_body(("plain",)).get_content()


def test_message_escapes_html_in_job_text():
    df = ranked(1).assign(title=["<script>x</script> & Co"])
    page = mailer.build_message(df, "a@b", ["a@b"], "d").get_body(("html",)).get_content()
    assert "<script>" not in page and "&lt;script&gt;" in page


def test_a_quiet_day_is_one_line_not_the_same_list_again():
    top = ranked(2).assign(new=False)
    msg = mailer.build_message(top, "a@b", ["a@b"], "d")
    assert msg["Subject"] == "fresher-radar: nothing new today (d)"
    text = msg.get_body(("plain",)).get_content()
    assert text.strip() == "No new jobs made your top list today. Your top list (2 jobs) is unchanged in jobs.xlsx."
    assert top["job_url"].iloc[0] not in text


@pytest.mark.parametrize("below, says", [(1, "; 1 new one scored lower."), (3, "; 3 new ones scored lower.")])
def test_a_quiet_day_says_how_many_new_jobs_scored_lower(below, says):
    msg = mailer.build_message(ranked(2).assign(new=False), "a@b", ["a@b"], "d", new_below=below)
    assert says in msg.get_body(("plain",)).get_content()


def test_a_quiet_day_still_carries_the_note():
    msg = mailer.build_message(ranked(1).assign(new=False), "a@b", ["a@b"], "d", note="jobs.xlsx was open")
    assert msg.get_body(("plain",)).get_content().startswith("Note: jobs.xlsx was open")


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


def test_send_top_sends_everything_above_the_bar_and_counts_the_rest(ecfg, caplog):
    ecfg["ranking"]["top"] = {"min_fit": 80, "min_jobs": 0, "max_jobs": 60, "more_fit": 60}
    with caplog.at_level(logging.INFO, logger="jsearch"):
        assert mailer.send_top(ranked(5), ecfg, "d", ENV, factory()) is True   # fits 50..90
    msg = FakeSMTP.instances[0].sent[0][0]
    text, page = msg.get_body(("plain",)).get_content(), msg.get_body(("html",)).get_content()
    assert msg["Subject"] == "fresher-radar: 1 new, top 2 jobs (d)" and "[90]" in text and "[70]" not in text
    assert "...and 2 more with fit 60+ in jobs.xlsx." in text and "...and 2 more with fit 60+" in page
    assert "emailed 1 new, top 2 jobs (d) to me@gmail.com" in caplog.text


def test_send_top_counts_new_jobs_below_the_list_on_a_quiet_day(ecfg):
    ecfg["ranking"]["top"] = {"min_fit": 80, "min_jobs": 0, "max_jobs": 60, "more_fit": 60}
    df = ranked(5).assign(new=[True, False, False, False, False])   # the only new job has fit 50
    mailer.send_top(df, ecfg, "d", ENV, factory())
    msg = FakeSMTP.instances[0].sent[0][0]
    assert msg["Subject"] == "fresher-radar: nothing new today (d)"
    assert "; 1 new one scored lower." in msg.get_body(("plain",)).get_content()


def test_no_more_line_when_nothing_else_is_good(ecfg):
    ecfg["ranking"]["top"] = {"min_fit": 50, "min_jobs": 0, "max_jobs": 60, "more_fit": 60}
    mailer.send_top(ranked(5), ecfg, "d", ENV, factory())
    assert "more with fit" not in FakeSMTP.instances[0].sent[0][0].get_body(("plain",)).get_content()


def test_send_top_includes_reserved_jobs(ecfg):
    ecfg["ranking"]["top"] = fixed(2)
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


# --- the jobs.xlsx attachment --------------------------------------------------------------------------------

XLSX_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


@pytest.fixture
def xlsx(tmp_path) -> Path:
    path = tmp_path / "jobs.xlsx"
    path.write_bytes(b"PK\x03\x04 fake workbook")
    return path


def attachments(msg) -> list[tuple[str, str, bytes]]:
    return [(a.get_filename(), a.get_content_type(), a.get_content()) for a in msg.iter_attachments()]


def test_the_excel_file_is_attached_with_a_note_that_it_is_a_copy(ecfg, xlsx, caplog):
    with caplog.at_level(logging.INFO, logger="jsearch"):
        assert mailer.send_top(ranked(4), ecfg, "d", ENV, factory(), attachment=xlsx) is True
    msg = FakeSMTP.instances[0].sent[0][0]
    assert attachments(msg) == [("jobs.xlsx", XLSX_TYPE, b"PK\x03\x04 fake workbook")]
    for kind in ("plain", "html"):
        assert "Attached: jobs.xlsx, a copy of the full list." in msg.get_body((kind,)).get_content()
    assert "without the attachment" not in caplog.text


def test_a_quiet_day_email_carries_the_attachment_too(ecfg, xlsx):
    mailer.send_top(ranked(3).assign(new=False), ecfg, "d", ENV, factory(), attachment=xlsx)
    msg = FakeSMTP.instances[0].sent[0][0]
    assert msg["Subject"] == "fresher-radar: nothing new today (d)"
    assert [a[0] for a in attachments(msg)] == ["jobs.xlsx"]
    assert "Attached: jobs.xlsx" in msg.get_body(("plain",)).get_content()


def test_a_fallback_copy_is_attached_under_its_own_name(ecfg, tmp_path):
    copy = tmp_path / "jobs_2026-11-09_20-18-54.xlsx"   # jobs.xlsx was open in Excel
    copy.write_bytes(b"workbook")
    mailer.send_top(ranked(2), ecfg, "d", ENV, factory(), attachment=copy)
    assert [a[0] for a in attachments(FakeSMTP.instances[0].sent[0][0])] == [copy.name]


def test_attaching_off_or_no_file_sends_the_plain_email(ecfg, xlsx, caplog):
    ecfg["email"]["attach_excel"] = False
    with caplog.at_level(logging.INFO, logger="jsearch"):
        mailer.send_top(ranked(2), ecfg, "d", ENV, factory(), attachment=xlsx)
        msg = FakeSMTP.instances[0].sent[0][0]
        ecfg["email"]["attach_excel"] = True
        mailer.send_top(ranked(2), ecfg, "d", ENV, factory())          # no file given (e.g. older callers)
        msg2 = FakeSMTP.instances[0].sent[0][0]
    for m in (msg, msg2):
        assert attachments(m) == [] and "Attached:" not in m.get_body(("plain",)).get_content()
    assert "without the attachment" not in caplog.text   # nothing went wrong, so nothing to say


@pytest.mark.parametrize("make, why", [
    (lambda p: p.unlink(), "could not read"),
    (lambda p: p.write_bytes(b"x" * 1_500_001), "jobs.xlsx is 1.5 MB, over email.attach_max_mb 1"),
])
def test_a_missing_or_too_big_file_is_logged_and_the_email_still_goes(ecfg, xlsx, caplog, make, why):
    ecfg["email"]["attach_max_mb"] = 1
    make(xlsx)
    with caplog.at_level(logging.INFO, logger="jsearch"):
        assert mailer.send_top(ranked(2), ecfg, "d", ENV, factory(), attachment=xlsx) is True
    msg = FakeSMTP.instances[0].sent[0][0]
    assert attachments(msg) == [] and "Attached:" not in msg.get_body(("plain",)).get_content()
    assert "email goes without the attachment" in caplog.text and why in caplog.text


def test_a_file_exactly_at_the_cap_is_attached(ecfg, xlsx):
    ecfg["email"]["attach_max_mb"] = 1
    xlsx.write_bytes(b"x" * 1_000_000)
    mailer.send_top(ranked(2), ecfg, "d", ENV, factory(), attachment=xlsx)
    assert [a[0] for a in attachments(FakeSMTP.instances[0].sent[0][0])] == ["jobs.xlsx"]


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
    assert kw["attachment"] == ecfg["paths"]["excel"] and kw["attachment"].is_file()   # this run's workbook
    # from profile.example.yaml's PM resume: 3 slots, and its focus terms for the title check
    assert kw["reserve"] == {"PM": (3, ["product", "roadmap", "stakeholder", "user research", "metrics"])}
