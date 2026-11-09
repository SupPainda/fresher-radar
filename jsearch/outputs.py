"""Output files: check they can be written before scraping, and fall back when one is locked at save time.

The Excel copy of the job list is formatted for daily use: links you can click, a frozen header row with
filters, NEW rows highlighted, fit coloured red to green, and a status drop-down.
"""
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.worksheet.worksheet import Worksheet

from jsearch.runlog import IST

SHEET = "jobs"
WIDTHS = {"new": 6, "title": 45, "company": 24, "fit": 6, "why": 60, "location": 22, "URL": 14, "notes": 30,
          "status": 12}       # by displayed column name; others get DEFAULT_WIDTH
DEFAULT_WIDTH = 13
WRAP = {"why", "notes"}
HEADER_FILL = PatternFill("solid", fgColor="DDDDDD")
NEW_FILL = PatternFill("solid", fgColor="E2F0D9")
LINK_FONT = Font(color="0563C1", underline="single")

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


def _format(ws: Worksheet, out: pd.DataFrame, url_col: str, status_choices: list[str]) -> None:
    cols = {name: i for i, name in enumerate(out.columns, 1)}
    last_row, last_col = len(out) + 1, len(out.columns)
    for name, i in cols.items():
        letter = get_column_letter(i)
        ws.column_dimensions[letter].width = WIDTHS.get(name, DEFAULT_WIDTH)
        head = ws.cell(row=1, column=i)
        head.font, head.fill = Font(bold=True), HEADER_FILL
    ws.freeze_panes = "B2"                       # header row and the NEW column stay put
    ws.auto_filter.ref = f"A1:{get_column_letter(last_col)}{last_row}"
    for r in range(2, last_row + 1):
        for name in WRAP & cols.keys():
            ws.cell(row=r, column=cols[name]).alignment = Alignment(wrap_text=True, vertical="top")
        if url_col in cols:
            cell = ws.cell(row=r, column=cols[url_col])
            if isinstance(cell.value, str) and cell.value.startswith("http"):
                cell.hyperlink, cell.font = cell.value, LINK_FONT
        if "new" in cols and ws.cell(row=r, column=cols["new"]).value == "NEW":
            for c in range(1, last_col + 1):
                ws.cell(row=r, column=c).fill = NEW_FILL
    if "fit" in cols and last_row > 1:
        letter = get_column_letter(cols["fit"])
        ws.conditional_formatting.add(f"{letter}2:{letter}{last_row}", ColorScaleRule(
            start_type="num", start_value=40, start_color="F8696B", mid_type="num", mid_value=65,
            mid_color="FFEB84", end_type="num", end_value=90, end_color="63BE7B"))
    if "status" in cols and last_row > 1:
        letter = get_column_letter(cols["status"])
        # a drop-down of the usual statuses; anything else typed in is still accepted
        dv = DataValidation(type="list", formula1='"' + ",".join(status_choices) + '"', showErrorMessage=False)
        dv.add(f"{letter}2:{letter}{last_row}")
        ws.add_data_validation(dv)


def save_xlsx(out: pd.DataFrame, path: Path, url_col: str, status_choices: list[str]) -> Path:
    """Write the formatted Excel copy; if it is locked (open in Excel), a timestamped copy next to it."""
    def write(target: Path) -> None:
        with pd.ExcelWriter(target, engine="openpyxl") as writer:
            out.to_excel(writer, sheet_name=SHEET, index=False)
            _format(writer.sheets[SHEET], out, url_col, status_choices)
    try:
        write(path)
        return path
    except PermissionError:
        fallback = path.with_name(f"{path.stem}_{datetime.now(IST):%Y-%m-%d_%H-%M-%S}{path.suffix}")
        write(fallback)
        logger.warning(f"{path} was open in Excel, saved to {fallback} instead. Statuses typed into that copy are "
                       f"NOT read back: close {path.name} and run again, or copy them over.")
        return fallback
