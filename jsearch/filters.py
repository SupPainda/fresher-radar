"""Entry-level filtering: experience parsing, title rules, and the clean() pipeline."""
import re

import pandas as pd

from jsearch import store

# "3 years", "2+ yrs", "1-3 years", "2 to 5 year", "1.5 years"; the lower bound is group 1.
# (?<![\d.]) keeps "2024 years" or "v1.5 years" from matching their tail.
YEARS = re.compile(r"(?<![\d.])(\d{1,2})(?:\.\d+)?\s*\+?\s*(?:(?:-|–|to)\s*\d{1,2}\s*\+?\s*)?(?:years?|yrs?)\b", re.I)
EXP_CONTEXT = re.compile(r"\bexp", re.I)  # experience / experienced / exp.


def years_in_text(text: str, window: int) -> int | None:
    """Lowest years figure that has an "exp..." word within `window` chars of it, or None.

    The context check skips figures like "founded 25 years ago"; taking the lowest of several
    requirements ("3+ years Python ... 5+ years overall") errs towards keeping a posting.
    """
    found = [int(m.group(1)) for m in YEARS.finditer(text)
             if EXP_CONTEXT.search(text[max(0, m.start() - window):m.end() + window])]
    return min(found) if found else None


def min_years(row: pd.Series | dict, window: int) -> int | None:
    """Minimum years of experience: Naukri's experience_range, else the title, else the description."""
    rng = row.get("experience_range")
    text = str(rng) if pd.notna(rng) and rng else ""
    m = YEARS.search(text) or re.search(r"(\d+)", text)
    if m:
        return int(m.group(1))
    m = YEARS.search(str(row.get("title") or ""))  # "Data Engineer (2 to 5 Years)": no context needed
    if m:
        return int(m.group(1))
    return years_in_text(str(row.get("description") or ""), window)


def title_reject_reason(title: str, cfg: dict) -> str | None:
    """The rules of reject_reason that need only the title (wanted, excluded, level code, seniority).
    Sources use it to skip fetching details of postings that would be rejected anyway."""
    rx = cfg["rx"]
    if not rx["want"].search(title):
        return "not a wanted title"
    if m := rx["exclude"].search(title):
        return f"excluded word: {m.group(0)}"
    if m := rx["level_code"].search(title):
        return f"level code: {m.group(0).strip()}"
    dropped = {w.lower() for w in rx["drop"].findall(title)}
    if dropped == {"manager"}:  # a bare "manager" is allowed only for wanted manager titles with no seniority word
        if not rx["manager_ok"].search(title):
            return "manager title not in titles.manager_allowed"
        if m := rx["seniority"].search(title):
            return f"seniority word: {m.group(0).lower()}"
    elif dropped:
        return f"seniority word: {', '.join(sorted(dropped))}"
    return None


def reject_reason(row: pd.Series | dict, cfg: dict) -> str | None:
    """Why a posting is not entry-level / not wanted, or None when it is kept."""
    rx = cfg["rx"]
    title = str(row["title"])
    if reason := title_reject_reason(title, cfg):
        return reason
    y = row["min_years"]
    if pd.isna(y):
        # titles that are usually senior pass only with an explicit entry-level word
        if (m := rx["needs_entry_signal"].search(title)) and not rx["entry_signal"].search(title):
            return f'"{m.group(0)}" without an entry-level word or stated experience'
        return None
    limit = cfg["filters"]["max_min_years"]
    return f"asks for {y:g}+ years (limit {limit})" if y > limit else None


def keep(row: pd.Series | dict, cfg: dict) -> bool:
    return reject_reason(row, cfg) is None


def clean(raw: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Apply date, level and title filters, then dedupe.

    Returns (kept frame, counts per stage, rejected frame with a "reason" column, sorted by reason).
    """
    df = raw.copy()
    df["date_posted"] = pd.to_datetime(df["date_posted"], errors="coerce")
    # each source's own max_age_days, else search.days (company boards keep open jobs for months)
    max_age = df["site"].map(lambda s: cfg["sources"].get(s, {}).get("max_age_days") or cfg["search"]["days"])
    cutoff = pd.Timestamp.now().normalize() - pd.to_timedelta(max_age, unit="D")
    in_range = df["date_posted"].isna() | (df["date_posted"] >= cutoff)
    old = df[~in_range]
    old = old.assign(reason="posted " + old["date_posted"].dt.strftime("%Y-%m-%d") + ", older than "
                     + max_age[~in_range].astype(str) + " days")
    df = df[in_range]
    counts = {"raw": len(raw), "in date range": len(df)}
    window = cfg["filters"]["experience_context_chars"]
    df["min_years"] = df.apply(min_years, axis=1, window=window) if len(df) else pd.Series(dtype=float)
    reasons = df.apply(reject_reason, axis=1, cfg=cfg) if len(df) else pd.Series(dtype=object)
    rejected = pd.concat([old, df[reasons.notna()].assign(reason=reasons[reasons.notna()])], ignore_index=True)
    rejected = rejected.sort_values(["reason", "title"], kind="stable").reset_index(drop=True)
    df = df[reasons.isna()]
    counts["entry-level"] = len(df)
    df = (df.assign(_key=store.job_key(df["title"], df["company"]))
            .drop_duplicates("job_url").drop_duplicates("_key").drop(columns="_key"))
    df["internship"] = (df["title"].fillna("").astype(str)  # astype: an empty column may not be text-typed
                        .str.contains(cfg["titles"]["internship"], case=False, regex=True))
    counts["deduped"] = len(df)
    return df, counts, rejected
