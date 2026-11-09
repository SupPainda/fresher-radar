"""Run logging: IST 12-hour stamps, one file per run, library messages included."""
import logging
import re
from datetime import datetime, timezone

import pytest

from jsearch import runlog

STAMP = r"\d{2}:\d{2}:\d{2} (AM|PM) "


@pytest.fixture(autouse=True)
def restore_logging():
    """start_log reconfigures the root logger; put pytest's handlers back afterwards."""
    root = logging.getLogger()
    saved, level = list(root.handlers), root.level
    yield
    for h in list(root.handlers):
        root.removeHandler(h)
        h.close()
    for h in saved:
        root.addHandler(h)
    root.setLevel(level)
    logging.captureWarnings(False)


def record(msg: str, name: str = "jsearch.runner", level: int = logging.INFO, created: float | None = None):
    r = logging.LogRecord(name, level, __file__, 1, msg, None, None)
    if created is not None:
        r.created = created
    return r


def test_stamp_is_ist_12_hour_clock():
    utc_noon = datetime(2026, 11, 8, 12, 0, 0, tzinfo=timezone.utc).timestamp()
    out = runlog.ISTFormatter().format(record("hello", created=utc_noon))
    assert out == "05:30:00 PM hello"            # 12:00 UTC = 17:30 IST


def test_blank_message_stays_blank():
    assert runlog.ISTFormatter().format(record("")) == ""


def test_library_records_show_level_and_logger():
    out = runlog.ISTFormatter().format(record("status 406", name="jobspy.naukri", level=logging.ERROR))
    assert re.fullmatch(STAMP + "ERROR jobspy.naukri: status 406", out)


def test_start_log_writes_ours_and_library_errors_to_the_run_file(tmp_path):
    path = runlog.start_log(tmp_path / "logs")
    logging.getLogger("jsearch.runner").info("indeed [1/2] AI Engineer / Pune: 30 rows")
    logging.getLogger("jobspy.naukri").error("response status code 406")
    logging.getLogger("urllib3").info("library chatter")          # below WARNING: not logged
    lines = path.read_text(encoding="utf-8").splitlines()
    assert path.parent == tmp_path / "logs" and re.fullmatch(r"run_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.log", path.name)
    assert len(lines) == 2
    assert re.fullmatch(STAMP + r"indeed \[1/2\] AI Engineer / Pune: 30 rows", lines[0])
    assert re.fullmatch(STAMP + "ERROR jobspy.naukri: response status code 406", lines[1])


def test_start_log_without_dir_writes_no_file_and_is_idempotent(tmp_path):
    assert runlog.start_log(None) is None
    runlog.start_log(None)
    assert len(logging.getLogger().handlers) == 1   # a second call does not double every line
