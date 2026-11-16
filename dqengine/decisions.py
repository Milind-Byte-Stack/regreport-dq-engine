"""Human-in-the-loop decision log for AI triage suggestions."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

LOG_PATH = Path(__file__).resolve().parent.parent / "data" / "output" / "decisions_log.csv"
COLUMNS = ["timestamp_utc", "reviewer", "group_id", "regime", "title", "rule_ids",
           "record_count", "decision", "comment", "triage_source"]


def log_decision(group: dict, decision: str, reviewer: str, comment: str = "",
                 path: Path = LOG_PATH) -> dict:
    decision = decision.upper()
    if decision not in {"APPROVED", "REJECTED"}:
        raise ValueError("decision must be APPROVED or REJECTED")
    row = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "reviewer": reviewer or "unknown",
        "group_id": group["group_id"],
        "regime": group["regime"],
        "title": group["title"],
        "rule_ids": group["rule_ids"],
        "record_count": group["record_count"],
        "decision": decision,
        "comment": comment,
        "triage_source": group["source"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row], columns=COLUMNS).to_csv(path, mode="a", header=not path.exists(), index=False)
    return row


def load_decisions(path: Path = LOG_PATH) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(path, dtype=str, keep_default_na=False)
