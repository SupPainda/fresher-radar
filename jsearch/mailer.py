"""Email the top jobs after a run: the top list (config "ranking.top", by the final fit, so after any LLM
rescoring), best fit first, new ones marked, each with its link; then a count of the other good fits.

Credentials come from .env (EMAIL_FROM, EMAIL_APP_PASSWORD, EMAIL_TO; for Gmail an app password, see
README); the SMTP server from config.json "email". Optional by design: without credentials, or on any SMTP
problem, send_top() logs one line and the run carries on.
"""
import html
import logging
import os
import smtplib
import ssl
from dataclasses import dataclass
from email.message import EmailMessage
from typing import Callable, Mapping

import pandas as pd

from jsearch.ranking import Reserve, top_picks

logger = logging.getLogger("jsearch.mailer")


class MailUnavailable(RuntimeError):
    """The email couldn't be sent (no credentials, login refused, network); the message says why."""


@dataclass(frozen=True)
class MailSettings:
    sender: str
    password: str
    to: list[str]

    def __repr__(self) -> str:  # never print the password
        return f"MailSettings(from {self.sender} to {self.to})"


def settings(env: Mapping[str, str] = os.environ) -> MailSettings:
    """Sender, app password and recipients from the environment. Raises MailUnavailable when incomplete."""
    sender = env.get("EMAIL_FROM", "").strip()
    password = env.get("EMAIL_APP_PASSWORD", "").replace(" ", "").strip()  # Gmail shows it in groups of four
    to = [a.strip() for a in env.get("EMAIL_TO", "").split(",") if a.strip()] or ([sender] if sender else [])
    missing = [name for name, value in (("EMAIL_FROM", sender), ("EMAIL_APP_PASSWORD", password)) if not value]
    if missing:
        raise MailUnavailable(f"{' and '.join(missing)} not set in .env (see .env.example)")
    return MailSettings(sender, password, to)


def top_jobs(df: pd.DataFrame, top: Mapping, reserve: Mapping[str, Reserve] | None = None) -> pd.DataFrame:
    """The top list, best first, with each resume's reserved slots (see ranking.top_picks)."""
    return top_picks(df, top, reserve)


def more_good_fits(df: pd.DataFrame, top_list: pd.DataFrame, more_fit: float) -> int:
    """Jobs not on the top list with fit >= more_fit: the email points to them in the CSV."""
    if "fit" not in df or "job_url" not in df:
        return 0
    rest = df[~df["job_url"].isin(top_list["job_url"])]
    return int((rest["fit"] >= more_fit).sum())


def _cell(row: pd.Series, col: str) -> str:
    value = row.get(col)
    return "" if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)) else str(value)


SUBJECT = "fresher-radar"
TABLE_HEAD = ('<table cellpadding="6" style="border-collapse:collapse" border="1">'
              "<tr><th>#</th><th>Fit</th><th>Job</th><th>Why</th><th>Send resume</th></tr>")


def _is_new(r: pd.Series) -> bool:
    return pd.notna(r.get("new")) and bool(r.get("new"))


def _section(jobs: pd.DataFrame, start: int) -> tuple[list[str], list[str]]:
    """Plain-text entries and HTML table rows for jobs, numbered from start."""
    lines, rows = [], []
    for i, (_, r) in enumerate(jobs.iterrows(), start):
        extra = " | ".join(x for x in (_cell(r, "location"), f"send: {_cell(r, 'send')}" if _cell(r, "send") else "")
                           if x)
        lines.append(f"{i}. [{_cell(r, 'fit')}] {_cell(r, 'title')} at {_cell(r, 'company')}\n"
                     f"   {_cell(r, 'why')}\n   {extra}\n   {_cell(r, 'job_url')}")
        e = {c: html.escape(_cell(r, c)) for c in ("title", "company", "location", "why", "send", "job_url", "fit")}
        rows.append(f'<tr><td>{i}</td><td align="right"><b>{e["fit"]}</b></td>'
                    f'<td><a href="{e["job_url"]}">{e["title"]}</a><br><small>{e["company"]} · '
                    f'{e["location"]}</small></td><td><small>{e["why"]}</small></td><td>{e["send"]}</td></tr>')
    return lines, rows


def build_message(top: pd.DataFrame, sender: str, to: list[str], when: str, note: str = "",
                  more: str = "", new_below: int = 0) -> EmailMessage:
    """Plain-text and HTML versions of the same list, the jobs new today first.

    note (e.g. where the list went) goes above it, more (how many other good fits jobs.xlsx has) below it.
    With no new job on the list, the email is one line instead (new_below: new jobs that scored lower).
    """
    is_new = top.apply(_is_new, axis=1).astype(bool) if len(top) else pd.Series(dtype=bool)
    fresh, rest = top[is_new], top[~is_new]
    msg = EmailMessage()
    msg["From"], msg["To"] = sender, ", ".join(to)
    head_text = f"Note: {note}\n\n" if note else ""
    head_html = f'<p style="color:#c0392b"><b>Note:</b> {html.escape(note)}</p>' if note else ""

    if not len(fresh):  # quiet day: one line, not the same list again
        msg["Subject"] = f"{SUBJECT}: nothing new today ({when})"
        line = (f"No new jobs made your top list today"
                + (f"; {new_below} new {'one' if new_below == 1 else 'ones'} scored lower" if new_below else "")
                + f". Your top list ({len(top)} jobs) is unchanged in jobs.xlsx.")
        msg.set_content(head_text + line + "\n")
        msg.add_alternative(f'<html><body style="font-family:sans-serif">{head_html}<p>{html.escape(line)}</p>'
                            "</body></html>", subtype="html")
        return msg

    msg["Subject"] = f"{SUBJECT}: {len(fresh)} new, top {len(top)} jobs ({when})"
    new_lines, new_rows = _section(fresh, 1)
    old_lines, old_rows = _section(rest, len(fresh) + 1)
    text = head_text + f"New today ({len(fresh)}):\n\n" + "\n\n".join(new_lines) + "\n"
    page = (f'<html><body style="font-family:sans-serif">{head_html}'
            f'<h3 style="color:#c0392b">New today ({len(fresh)})</h3>' + TABLE_HEAD + "".join(new_rows) + "</table>")
    if len(rest):
        text += f"\nAlso on your top list ({len(rest)}):\n\n" + "\n\n".join(old_lines) + "\n"
        page += f"<h3>Also on your top list ({len(rest)})</h3>" + TABLE_HEAD + "".join(old_rows) + "</table>"
    text += f"\n{more}\n" if more else ""
    page += (f"<p>{html.escape(more)}</p>" if more else "") + "</body></html>"
    msg.set_content(text)
    msg.add_alternative(page, subtype="html")
    return msg


def send(msg: EmailMessage, s: MailSettings, cfg: dict,
         smtp_factory: Callable[..., smtplib.SMTP] | None = None) -> None:
    """Send over SMTP with TLS. Raises MailUnavailable."""
    e = cfg["email"]
    factory = smtp_factory or smtplib.SMTP_SSL  # looked up per call, so tests can block it
    try:
        with factory(e["smtp_host"], e["smtp_port"], timeout=e["timeout_seconds"],
                          context=ssl.create_default_context()) as smtp:
            smtp.login(s.sender, s.password)
            smtp.send_message(msg, to_addrs=s.to)
    except smtplib.SMTPAuthenticationError:
        raise MailUnavailable(f"{e['smtp_host']} refused the login for {s.sender}; check EMAIL_APP_PASSWORD "
                              f"(for Gmail an app password, not your normal one)") from None
    except smtplib.SMTPException as err:
        raise MailUnavailable(f"{e['smtp_host']} refused the email: {err}") from None
    except OSError as err:  # DNS, refused connection, timeout, TLS
        raise MailUnavailable(f"could not reach {e['smtp_host']}:{e['smtp_port']} ({err})") from None


def send_top(df: pd.DataFrame, cfg: dict, when: str, env: Mapping[str, str] = os.environ,
             smtp_factory: Callable[..., smtplib.SMTP] | None = None, note: str = "",
             reserve: Mapping[str, Reserve] | None = None) -> bool:
    """Email the top list (ranking.top). Never raises: problems are logged. True when sent."""
    if not cfg["email"]["enabled"]:
        return False
    top_cfg = cfg["ranking"]["top"]
    top = top_jobs(df, top_cfg, reserve)
    if not len(top):
        logger.info("email skipped: no ranked jobs to send (ranking needs a profile)")
        return False
    extra = more_good_fits(df, top, top_cfg["more_fit"])
    more = f"...and {extra} more with fit {top_cfg['more_fit']:g}+ in jobs.xlsx." if extra else ""
    on_list = set(top["job_url"]) if "job_url" in top else set()
    new_below = int(sum(_is_new(r) for _, r in df.iterrows() if r.get("job_url") not in on_list))
    try:
        s = settings(env)
        msg = build_message(top, s.sender, s.to, when, note, more, new_below)
        send(msg, s, cfg, smtp_factory)
    except MailUnavailable as e:
        logger.info(f"email skipped ({e})")
        return False
    logger.info(f"emailed {msg['Subject'].split(': ', 1)[1]} to {', '.join(s.to)}")
    return True
