"""fetch_jobs.main: clean exits for a bad config and a locked output."""
import logging
import sys

import pytest

import fetch_jobs
from jsearch.config import ConfigError
from jsearch.outputs import OutputLocked


@pytest.fixture(autouse=True)
def keep_pytest_logging():
    root = logging.getLogger()
    saved = list(root.handlers)
    yield
    for h in list(root.handlers):
        root.removeHandler(h)
    for h in saved:
        root.addHandler(h)
    logging.captureWarnings(False)


def test_bad_config_exits_2_with_the_message_and_no_traceback(monkeypatch, capsys):
    def bad():
        raise ConfigError("config.json is invalid:\n  search.days: Field required")
    monkeypatch.setattr(fetch_jobs, "load_config", bad)
    monkeypatch.setattr(sys, "argv", ["fetch_jobs.py"])
    with pytest.raises(SystemExit) as e:
        fetch_jobs.main()
    err = capsys.readouterr().err
    assert e.value.code == 2 and "search.days: Field required" in err and "Traceback" not in err


def test_locked_output_exits_1_with_the_message(cfg, monkeypatch, capsys):
    monkeypatch.setattr(fetch_jobs, "load_config", lambda: cfg)
    monkeypatch.setattr(sys, "argv", ["fetch_jobs.py", "--refilter"])

    def locked(args, cfg):
        raise OutputLocked("data\\jobs.csv is open in another program (Excel?). Close it and run again.")
    monkeypatch.setattr(fetch_jobs, "run", locked)
    with pytest.raises(SystemExit) as e:
        fetch_jobs.main()
    out = capsys.readouterr().out
    assert e.value.code == 1 and "Close it and run again" in out and "run ended" in out
