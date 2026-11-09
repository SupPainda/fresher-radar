"""Run logging: every line stamped in IST (12-hour clock), to the console and to one file per run.

jobspy logs through the standard `logging` module too, so its messages (e.g. Naukri's 406) land in
the same stamped output, prefixed with their logger name.
"""
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

IST = timezone(timedelta(hours=5, minutes=30), "IST")  # India has no daylight saving
logger = logging.getLogger("jsearch")


class ISTFormatter(logging.Formatter):
    """'06:34:12 PM message'; other libraries' records also show level and logger name; blank stays blank."""

    def format(self, record: logging.LogRecord) -> str:
        msg = record.getMessage()
        if not msg:
            return ""
        if not record.name.startswith("jsearch"):
            msg = f"{record.levelname} {record.name}: {msg}"
        if record.exc_info:
            msg += "\n" + self.formatException(record.exc_info)
        return datetime.fromtimestamp(record.created, IST).strftime("%I:%M:%S %p ") + msg


def start_log(log_dir: Path | None) -> Path | None:
    """Send logging to the console and, with log_dir, to log_dir/run_<date>_<time>.log. Returns the file."""
    root = logging.getLogger()
    for h in list(root.handlers):  # idempotent: a second call (tests) does not double every line
        root.removeHandler(h)
        h.close()
    root.setLevel(logging.WARNING)          # libraries: warnings and errors only
    logger.setLevel(logging.INFO)           # ours: everything
    fmt = ISTFormatter()
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    root.addHandler(console)
    path = None
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        path = log_dir / f"run_{datetime.now(IST):%Y-%m-%d_%H-%M-%S}.log"  # 24h here so files sort by time
        file = logging.FileHandler(path, encoding="utf-8")  # flushes every record: readable while running
        file.setFormatter(fmt)
        root.addHandler(file)
    logging.captureWarnings(True)           # pandas/python warnings go to the log too
    return path


def now_ist() -> str:
    return f"{datetime.now(IST):%Y-%m-%d %I:%M:%S %p} IST"
