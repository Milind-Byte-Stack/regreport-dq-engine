"""Daily Excel exceptions report."""
from __future__ import annotations

from datetime import date
from io import BytesIO
from pathlib import Path

import pandas as pd
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter

HEADER_FILL = PatternFill("solid", fgColor="1F3864")
SEV_FILL = {"HIGH": "F8CBAD", "MEDIUM": "FFE699", "LOW": "E2EFDA"}


def _style(ws, widths: dict[int, int] | None = None):
    for cell in ws[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = HEADER_FILL
    ws.freeze_panes = "A2"
    for i, col in enumerate(ws.iter_cols(min_row=1, max_row=min(ws.max_row, 200)), start=1):
        width = max(len(str(c.value or "")) for c in col)
        ws.column_dimensions[get_column_letter(i)].width = min(max(10, width + 2), 70)
    ws.auto_filter.ref = ws.dimensions


def build_excel(results, triage_groups: pd.DataFrame | None = None,
                decisions: pd.DataFrame | None = None, path: str | Path | None = None) -> bytes:
    """Build the workbook. Returns bytes; also writes to `path` if given."""
    summary = pd.DataFrame([{
        "Regime": r.regime,
        "Records": len(r.data),
        "Records with exceptions": r.exceptions["record_id"].nunique(),
        "Exceptions": len(r.exceptions),
        "Rules run": r.rule_count,
        "DQ score %": r.overall,
    } for r in results])
    exceptions = pd.concat([r.exceptions for r in results], ignore_index=True)
    rules = pd.concat([r.rules for r in results], ignore_index=True)
    fields = pd.concat([r.fields.assign(regime=r.regime) for r in results], ignore_index=True)
    daily = pd.concat([r.daily.assign(regime=r.regime) for r in results], ignore_index=True)

    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        summary.to_excel(xw, sheet_name="Summary", index=False, startrow=3)
        ws = xw.sheets["Summary"]
        ws["A1"] = "RegReport DQ Engine - Daily Exceptions Report"
        ws["A1"].font = Font(bold=True, size=14)
        ws["A2"] = f"Run date: {date.today().isoformat()}  |  SYNTHETIC DATA ONLY"
        rules.to_excel(xw, sheet_name="Rule summary", index=False)
        exceptions.to_excel(xw, sheet_name="Exceptions", index=False)
        fields[["regime", "field", "failed_records", "dq_score"]].to_excel(xw, sheet_name="Field scores", index=False)
        daily[["regime", "report_date", "records", "failed_records", "dq_score"]].to_excel(
            xw, sheet_name="Daily scores", index=False)
        if triage_groups is not None and not triage_groups.empty:
            triage_groups.to_excel(xw, sheet_name="AI triage", index=False)
        if decisions is not None and not decisions.empty:
            decisions.to_excel(xw, sheet_name="Decision log", index=False)

        for name, sheet in xw.sheets.items():
            if name == "Summary":
                for cell in sheet[4]:
                    cell.font = Font(bold=True, color="FFFFFF")
                    cell.fill = HEADER_FILL
                for i in range(1, 7):
                    sheet.column_dimensions[get_column_letter(i)].width = 24
                continue
            _style(sheet)
        ex_ws = xw.sheets["Exceptions"]
        sev_col = list(exceptions.columns).index("severity") + 1 if not exceptions.empty else None
        if sev_col:
            for row in ex_ws.iter_rows(min_row=2, min_col=sev_col, max_col=sev_col):
                c = row[0]
                if c.value in SEV_FILL:
                    c.fill = PatternFill("solid", fgColor=SEV_FILL[c.value])

    data = buf.getvalue()
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(data)
    return data
