"""Output files: check they can be written before scraping, and fall back when one is locked at save time."""
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from jsearch.runlog import IST

logger = logging.getLogger("jsearch.outputs")


class OutputLocked(RuntimeError):
    """An output file is open in another program (on Windows, usually Excel), so it can't be written."""


def check_writable(*paths: Path) -> None:
    """Fail before any scraping if an existing output is locked. Opening for append changes nothing."""
    for path in paths:
        if not path.exists():
            continue
        try:
            with open(path, "a"):
                pass
        except PermissionError:
            raise OutputLocked(f"{path} is open in another program (Excel?). Close it and run again.") from None


def save_csv(out: pd.DataFrame, path: Path) -> Path:
    """Write the CSV; if it was locked meanwhile, write a timestamped copy next to it instead."""
    try:
        out.to_csv(path, index=False, encoding="utf-8-sig")
        return path
    except PermissionError:
        fallback = path.with_name(f"{path.stem}_{datetime.now(IST):%Y-%m-%d_%H-%M-%S}{path.suffix}")
        out.to_csv(fallback, index=False, encoding="utf-8-sig")
        logger.warning(f"{path} was locked (open in Excel?), saved to {fallback} instead. Statuses typed into "
                       f"that copy are NOT read back: close {path.name} and run again, or copy them over.")
        return fallback
