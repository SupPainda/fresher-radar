"""Output files: the up-front lock check and the save fallback."""
import ctypes
import logging
import sys
from contextlib import contextmanager

import pandas as pd
import pytest

from openpyxl import load_workbook

from jsearch import store
from jsearch.outputs import OutputLocked, check_writable, save_csv, save_xlsx

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


# --- the Excel copy --------------------------------------------------------------------------------------------

CHOICES = ["new", "interested", "applied", "rejected"]


def job_list() -> pd.DataFrame:
    """Two rows as the CSV has them: a NEW one and an old one."""
    return pd.DataFrame({"new": ["NEW", ""], "title": ["AI Engineer", "Data Engineer"], "company": ["A", "B"],
                         "fit": [88, 52], "why": ["python, rag", "sql"],
                         "URL": ["https://example.com/a", "https://example.com/b"],
                         "status": ["new", "interested"], "notes": [None, "call back"]})


def test_excel_copy_holds_the_same_rows(tmp_path):
    path = tmp_path / "jobs.xlsx"
    assert save_xlsx(job_list(), path, "URL", CHOICES) == path
    back = pd.read_excel(path, dtype=str)
    assert back["title"].tolist() == ["AI Engineer", "Data Engineer"]
    assert back["URL"].tolist() == ["https://example.com/a", "https://example.com/b"]
    assert back["notes"].fillna("").tolist() == ["", "call back"]


def test_excel_copy_is_formatted_for_daily_use(tmp_path):
    path = tmp_path / "jobs.xlsx"
    save_xlsx(job_list(), path, "URL", CHOICES)
    ws = load_workbook(path)["jobs"]
    assert ws.freeze_panes == "B2" and ws.auto_filter.ref == "A1:H3"
    assert ws["A1"].font.bold and ws.column_dimensions["B"].width == 45            # title
    assert ws["F2"].hyperlink.target == "https://example.com/a"                    # URL column, clickable
    assert ws["B2"].fill.fgColor.rgb.endswith("E2F0D9") and ws["B3"].fill.fgColor.rgb in (None, "00000000")
    assert ws["E2"].alignment.wrap_text                                             # why wraps
    [dv] = ws.data_validations.dataValidation
    assert dv.formula1 == '"new,interested,applied,rejected"' and str(dv.sqref) == "G2:G3"
    assert dv.showErrorMessage is False                                             # free text still allowed
    assert any("D2:D3" in str(rng.sqref) for rng in ws.conditional_formatting)     # fit colour scale


def test_an_empty_job_list_still_writes_a_header(tmp_path):
    path = tmp_path / "jobs.xlsx"
    save_xlsx(job_list().iloc[0:0], path, "URL", CHOICES)
    assert list(pd.read_excel(path).columns) == list(job_list().columns)


def test_a_locked_excel_copy_falls_back_to_a_timestamped_one(tmp_path, monkeypatch, caplog):
    path = tmp_path / "jobs.xlsx"
    real = pd.ExcelWriter

    def writer(target, **kw):  # Excel holding jobs.xlsx open: writing it fails, any other name works
        if target == path:
            raise PermissionError(13, "in use")
        return real(target, **kw)
    monkeypatch.setattr(pd, "ExcelWriter", writer)
    with caplog.at_level(logging.WARNING, logger="jsearch"):
        saved = save_xlsx(job_list(), path, "URL", CHOICES)
    assert saved != path and saved.name.startswith("jobs_") and saved.suffix == ".xlsx" and not path.exists()
    assert pd.read_excel(saved)["title"].tolist() == ["AI Engineer", "Data Engineer"]
    assert "NOT read back" in caplog.text


def test_statuses_typed_into_the_excel_copy_are_read_back(tmp_path):
    con = store.connect(tmp_path / "jobs.db")
    df = pd.DataFrame({"title": ["AI Engineer"], "company": ["A"], "location": "", "job_url": "u", "site": "s",
                       "date_posted": pd.Timestamp("2026-11-09"), "min_years": None, "is_remote": False,
                       "internship": False})
    store.record(con, df, "2026-11-09")
    edited = job_list().iloc[[0]].assign(status="applied", notes="sent GenAI resume")
    save_xlsx(edited, tmp_path / "jobs.xlsx", "URL", CHOICES)
    assert store.import_csv_edits(con, tmp_path / "jobs.xlsx", "title", "company") == 1
    assert con.execute("SELECT status, notes FROM jobs").fetchone() == ("applied", "sent GenAI resume")
