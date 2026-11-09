"""Output files: the up-front lock check and the save fallback."""
import ctypes
import logging
import sys
from contextlib import contextmanager

import pandas as pd
import pytest

from jsearch.outputs import OutputLocked, check_writable, save_csv

windows_only = pytest.mark.skipif(sys.platform != "win32", reason="Excel-style share locks are a Windows thing")


@contextmanager
def locked_like_excel(path):
    """Hold the file open allowing others to read but not write, as Excel does."""
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateFileW.restype = ctypes.c_void_p
    GENERIC_READ, FILE_SHARE_READ, OPEN_EXISTING = 0x80000000, 0x1, 3
    handle = k32.CreateFileW(str(path), GENERIC_READ, FILE_SHARE_READ, None, OPEN_EXISTING, 0, None)
    assert handle not in (None, ctypes.c_void_p(-1).value), ctypes.get_last_error()
    try:
        yield
    finally:
        k32.CloseHandle(ctypes.c_void_p(handle))


def frame() -> pd.DataFrame:
    return pd.DataFrame({"title": ["Data Engineer"], "status": ["new"]})


def test_missing_and_free_files_pass(tmp_path):
    free = tmp_path / "jobs.csv"
    free.write_text("x", encoding="utf-8")
    check_writable(tmp_path / "absent.csv", free)
    assert free.read_text(encoding="utf-8") == "x"     # the check changes nothing


@windows_only
def test_locked_file_fails_before_scraping(tmp_path):
    path = tmp_path / "jobs.csv"
    path.write_text("x", encoding="utf-8")
    with locked_like_excel(path), pytest.raises(OutputLocked, match="open in another program"):
        check_writable(path)


def test_save_writes_the_csv(tmp_path):
    path = tmp_path / "jobs.csv"
    assert save_csv(frame(), path) == path
    assert pd.read_csv(path, encoding="utf-8-sig")["title"].tolist() == ["Data Engineer"]


@windows_only
def test_save_falls_back_to_a_timestamped_copy_when_locked(tmp_path, caplog):
    path = tmp_path / "jobs.csv"
    path.write_text("old", encoding="utf-8")
    with locked_like_excel(path), caplog.at_level(logging.WARNING, logger="jsearch"):
        saved = save_csv(frame(), path)
    assert saved != path and saved.name.startswith("jobs_") and saved.suffix == ".csv"
    assert pd.read_csv(saved, encoding="utf-8-sig")["title"].tolist() == ["Data Engineer"]
    assert path.read_text(encoding="utf-8") == "old"
    assert "NOT read back" in caplog.text
