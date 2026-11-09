"""The source contract and the registry of source types.

A source type is a module with two functions:

  units(name, src, cfg, test) -> list[Unit]     the work to do (one search, one company board, ...)
  fetch(name, unit, src, cfg) -> Result         do one unit; raise on failure, set `refusal` when blocked

`name` is the source's key in config.json "sources", `src` its entry. The runner (jsearch.runner)
supplies everything else: delays, block cooldown/give-up, the stop event, stats, logging and tags.
Adding a source = a module here + a "type" in TYPES + an entry in config.json.
"""
from dataclasses import dataclass, field
from types import ModuleType

import pandas as pd


@dataclass(frozen=True)
class Unit:
    key: str                                   # stable id, e.g. "naukri|AI Engineer|Pune", "greenhouse|stripe"
    label: str                                 # for the log
    args: dict = field(default_factory=dict)   # what fetch() needs
    tags: dict = field(default_factory=dict)   # columns added to every row (e.g. search_title)


@dataclass
class Result:
    df: pd.DataFrame
    refusal: str | None = None   # the site refused (rate limit etc.); df may still hold the first pages
    note: str = ""               # extra log text


def source_type(name: str, src: dict) -> ModuleType:
    """The module implementing src["type"]."""
    from jsearch.sources import ats, jobspy_site  # imported here: those modules import Unit/Result from us
    types = {"jobspy": jobspy_site, "ats": ats}
    try:
        return types[src["type"]]
    except KeyError:
        raise ValueError(f"source {name!r}: unknown type {src.get('type')!r}, expected one of {sorted(types)}")
