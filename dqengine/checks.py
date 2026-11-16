"""Reusable check functions.

Every check takes (df, fields, params, ref) and returns a boolean Series
that is True where a record FAILS. Rules in the YAML packs point at these
checks by name, so new rules usually need no new Python.
"""
from __future__ import annotations

from typing import Callable

import pandas as pd

from . import identifiers as ids

CHECKS: dict[str, Callable] = {}


def check(name: str):
    def register(fn):
        CHECKS[name] = fn
        return fn
    return register


def is_blank(series: pd.Series) -> pd.Series:
    return series.isna() | (series.astype(str).str.strip() == "")


# ---------- generic checks ----------
@check("required")
def required(df, fields, params, ref):
    mask = pd.Series(False, index=df.index)
    for f in fields:
        mask |= is_blank(df[f])
    return mask


@check("unique")
def unique(df, fields, params, ref):
    f = fields[0]
    present = ~is_blank(df[f])
    dupes = df[f].duplicated(keep=False) & present
    return dupes | ~present


@check("allowed_values")
def allowed_values(df, fields, params, ref):
    allowed = set(params["values"])
    mask = pd.Series(False, index=df.index)
    for f in fields:
        mask |= ~df[f].astype(str).str.strip().isin(allowed)
    return mask


@check("fields_differ")
def fields_differ(df, fields, params, ref):
    a, b = fields
    return df[a].astype(str).str.upper() == df[b].astype(str).str.upper()


@check("conditional_required")
def conditional_required(df, fields, params, ref):
    """If `when_field` == `when_value`, `fields` must be filled; otherwise blank."""
    cond = df[params["when_field"]].astype(str).str.upper() == str(params["when_value"]).upper()
    mask = pd.Series(False, index=df.index)
    for f in fields:
        blank = is_blank(df[f])
        mask |= (cond & blank)
        if params.get("blank_otherwise", False):
            mask |= (~cond & ~blank)
    return mask


@check("positive_number")
def positive_number(df, fields, params, ref):
    mask = pd.Series(False, index=df.index)
    for f in fields:
        num = pd.to_numeric(df[f], errors="coerce")
        mask |= num.isna() | (num <= 0)
    return mask


@check("iso_date")
def iso_date(df, fields, params, ref):
    mask = pd.Series(False, index=df.index)
    for f in fields:
        parsed = pd.to_datetime(df[f], format="%Y-%m-%d", errors="coerce")
        mask |= parsed.isna()
    return mask


# ---------- identifier checks ----------
@check("lei")
def lei(df, fields, params, ref):
    mask = pd.Series(False, index=df.index)
    for f in fields:
        mask |= ~df[f].apply(ids.is_valid_lei)
    return mask


@check("isin")
def isin(df, fields, params, ref):
    mask = pd.Series(False, index=df.index)
    for f in fields:
        mask |= ~df[f].apply(ids.is_valid_isin)
    return mask


@check("utc_timestamp")
def utc_timestamp(df, fields, params, ref):
    mask = pd.Series(False, index=df.index)
    for f in fields:
        mask |= ~df[f].apply(ids.is_utc_timestamp)
    return mask


@check("uti_format")
def uti_format(df, fields, params, ref):
    """UTI: up to 52 uppercase alphanumerics, starting with the generator's LEI."""
    f = fields[0]
    s = df[f].astype(str).str.strip()
    ok = s.str.fullmatch(r"[A-Z0-9]{21,52}") & s.str[:20].apply(ids.is_valid_lei)
    return ~ok


# ---------- reference-data checks ----------
@check("in_reference")
def in_reference(df, fields, params, ref):
    """Value must be in a reference list (e.g. ISO 10383 MIC, ISO 4217)."""
    valid = set(ref[params["list"]]) | set(params.get("also_allowed", []))
    mask = pd.Series(False, index=df.index)
    for f in fields:
        mask |= ~df[f].astype(str).str.strip().str.upper().isin(valid)
    return mask


@check("id_matches_type")
def id_matches_type(df, fields, params, ref):
    """fields = [id_type_field, id_field]. The ID must match its declared type."""
    type_f, id_f = fields
    validators = {
        "LEI": ids.is_valid_lei,
        "CONCAT": ids.is_valid_concat,
        "NIDN": lambda v: isinstance(v, str) and 2 < len(v.strip()) <= 35 and v[:2].isalpha(),
        "CCPT": lambda v: isinstance(v, str) and 2 < len(v.strip()) <= 35 and v[:2].isalpha(),
    }

    def bad(row):
        fn = validators.get(str(row[type_f]).strip().upper())
        return fn is None or not fn(row[id_f])

    return df.apply(bad, axis=1)
