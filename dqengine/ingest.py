"""Ingest: read report files into a DataFrame of strings.

Everything is read as text so the validators see exactly what was reported
(e.g. "0012" stays "0012", blanks stay blank). CSV only for now; EMIR Refit
ISO 20022 XML ingestion is on the roadmap.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_report(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() != ".csv":
        raise ValueError(f"Unsupported file type '{p.suffix}'. CSV is supported today.")
    df = pd.read_csv(p, dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    if "record_id" not in df.columns:
        df.insert(0, "record_id", [f"R{i:05d}" for i in range(1, len(df) + 1)])
    return df
