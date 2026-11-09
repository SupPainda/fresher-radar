"""Offline fit score: how well each job matches a profile, from skill and role overlap. Instant, no API.

fit = skill part + role part, both set in config.json "ranking":
  skill part: summed weights of profile skills found in the job (a title match counts title_multiplier
              times), saturating, scaled to skill_points. The saturation grows with the profile, so a long
              skill list doesn't put every job at 100: max(skill_saturation_min, skill_saturation_share x
              the profile's total skill weight);
  role part:  best word overlap between the job title and a target role, scaled to role_points.
"""
import re

import pandas as pd

from jsearch.profile import Profile

STOP = {"a", "an", "and", "the", "of", "for", "in", "with", "to", "at"}
MAX_WHY_SKILLS = 8


def term_rx(terms: list[str]) -> re.Pattern:
    """Any of the terms as a whole word or phrase ('rag' not in 'storage'; 'c++' and 'node.js' work)."""
    alts = "|".join(re.escape(t.strip()) for t in sorted(terms, key=len, reverse=True) if t.strip())
    return re.compile(rf"(?<!\w)(?:{alts})(?!\w)", re.I)


def words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower())) - STOP


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    return df[name].fillna("").astype(str) if name in df else pd.Series("", index=df.index)


def saturation(profile: Profile, cfg: dict) -> float:
    r = cfg["ranking"]
    return max(r["skill_saturation_min"], r["skill_saturation_share"] * sum(profile.skills.values()))


def score(df: pd.DataFrame, profile: Profile, cfg: dict) -> pd.DataFrame:
    """df plus fit (0-100), why, send (resume label) and scored_by="offline". Does not modify df."""
    r = cfg["ranking"]
    cap = saturation(profile, cfg)
    skills = [(s, w, term_rx([s] + profile.aliases.get(s, []))) for s, w in profile.skills.items()]
    roles = [(role, words(role)) for role in profile.target_roles]
    focus = [(res.label, [term_rx([t]) for t in res.focus]) for res in profile.resumes]
    titles = _col(df, "title")
    bodies = _col(df, "description") + "\n" + _col(df, "skills")  # Naukri: skill tags, no description

    fits, whys, sends = [], [], []
    for title, body in zip(titles, bodies):
        points, matched = 0.0, []
        for s, w, rx in skills:
            if rx.search(title):
                points += w * r["title_multiplier"]
            elif rx.search(body):
                points += w
            else:
                continue
            matched.append((w, s))
        skill_part = min(1.0, points / cap) * r["skill_points"]

        tw = words(title)
        best_role, overlap = max(((role, len(rw & tw) / len(rw)) for role, rw in roles if rw),
                                 key=lambda x: x[1], default=("", 0.0))
        fits.append(round(skill_part + overlap * r["role_points"]))

        shown = ", ".join(s for _, s in sorted(matched, key=lambda m: (-m[0], m[1]))[:MAX_WHY_SKILLS])
        whys.append(" | ".join(filter(None, [shown or "no profile skills mentioned",
                                             f"role: {best_role}" if overlap else ""])))

        # focus terms in the title count like skills do, so "MLOps Engineer" beats a JD's passing "docker"
        counts = [(sum(r["title_multiplier"] if rx.search(title) else 1 if rx.search(body) else 0 for rx in rxs),
                   label) for label, rxs in focus]
        sends.append(max(counts, key=lambda c: c[0])[1] if counts else "")  # ties: first listed resume

    return df.assign(fit=fits, why=whys, send=sends, scored_by="offline")
