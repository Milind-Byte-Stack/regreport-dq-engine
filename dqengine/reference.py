"""Reference data loaders.

The repo ships small starter lists so the demo runs offline. For full
coverage, drop the official files into data/reference/:
  - ISO 10383 MIC list  -> ISO10383_MIC.csv  (column "MIC")
  - ISO 4217 currencies -> list_one.csv      (column "Alphabetic Code" or "code")
They are picked up automatically if present.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

REF_DIR = Path(__file__).resolve().parent.parent / "data" / "reference"
GLEIF_URL = "https://api.gleif.org/api/v1/lei-records/{lei}"


def _first_existing(*names: str) -> Path | None:
    for n in names:
        p = REF_DIR / n
        if p.exists():
            return p
    return None


def _column(df: pd.DataFrame, *candidates: str) -> pd.Series:
    for c in candidates:
        if c in df.columns:
            return df[c]
    raise KeyError(f"None of {candidates} found in {list(df.columns)}")


@lru_cache(maxsize=1)
def load_reference() -> dict[str, set[str]]:
    mic_file = _first_existing("ISO10383_MIC.csv", "mic_codes.csv")
    ccy_file = _first_existing("list_one.csv", "currencies.csv")
    mics = _column(pd.read_csv(mic_file, dtype=str), "MIC", "mic")
    ccys = _column(pd.read_csv(ccy_file, dtype=str), "Alphabetic Code", "code")
    return {
        "mic": set(mics.dropna().str.strip().str.upper()),
        "currency": set(ccys.dropna().str.strip().str.upper()),
    }


def gleif_lookup(lei: str, timeout: float = 5.0) -> dict | None:
    """Optional live check against GLEIF's public API (no key needed).

    Returns {"lei", "legal_name", "status"} or None if not found.
    Not used in the default offline pipeline; synthetic LEIs will not exist.
    """
    import requests

    r = requests.get(GLEIF_URL.format(lei=lei), timeout=timeout)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    attrs = r.json()["data"]["attributes"]
    return {
        "lei": lei,
        "legal_name": attrs["entity"]["legalName"]["name"],
        "status": attrs["registration"]["status"],
    }
