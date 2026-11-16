"""Data-quality scoring from an exceptions table."""
from __future__ import annotations

import pandas as pd


def _explode_fields(exceptions: pd.DataFrame) -> pd.DataFrame:
    ex = exceptions.copy()
    ex["field"] = ex["fields"].str.split(", ")
    return ex.explode("field")


def overall_score(df: pd.DataFrame, exceptions: pd.DataFrame) -> float:
    """Share of records with zero exceptions, as a percentage."""
    if len(df) == 0:
        return 100.0
    bad = exceptions["record_id"].nunique()
    return round(100 * (1 - bad / len(df)), 2)


def field_scores(df: pd.DataFrame, exceptions: pd.DataFrame) -> pd.DataFrame:
    """Per field: share of records where no rule touching that field failed."""
    total = len(df)
    ex = _explode_fields(exceptions)
    bad = ex.groupby("field")["record_id"].nunique()
    out = pd.DataFrame({"field": df.columns})
    out = out[out["field"] != "record_id"]
    out["failed_records"] = out["field"].map(bad).fillna(0).astype(int)
    out["dq_score"] = (100 * (1 - out["failed_records"] / total)).round(2)
    return out.sort_values(["dq_score", "field"]).reset_index(drop=True)


def daily_scores(df: pd.DataFrame, exceptions: pd.DataFrame) -> pd.DataFrame:
    """Per report date: records, failing records, DQ score."""
    totals = df.groupby("report_date")["record_id"].nunique().rename("records")
    bad = exceptions.groupby("report_date")["record_id"].nunique().rename("failed_records")
    out = pd.concat([totals, bad], axis=1).fillna(0).astype(int).reset_index()
    out["dq_score"] = (100 * (1 - out["failed_records"] / out["records"])).round(2)
    return out


def rule_summary(exceptions: pd.DataFrame) -> pd.DataFrame:
    if exceptions.empty:
        return pd.DataFrame(columns=["regime", "rule_id", "rule_name", "severity", "exceptions"])
    return (exceptions.groupby(["regime", "rule_id", "rule_name", "severity"])["record_id"]
            .count().rename("exceptions").reset_index()
            .sort_values(["regime", "rule_id"]))
