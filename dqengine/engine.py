"""Rule engine: loads a YAML rule pack and runs it over a DataFrame."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

from .checks import CHECKS
from .reference import load_reference

RULES_DIR = Path(__file__).resolve().parent.parent / "config" / "rules"
SEVERITIES = ("HIGH", "MEDIUM", "LOW")

EXCEPTION_COLUMNS = [
    "regime", "rule_id", "rule_name", "severity", "record_id", "report_date",
    "fields", "values", "message",
]


@dataclass
class Rule:
    id: str
    name: str
    severity: str
    message: str
    steps: list[dict]                      # [{check, fields, params}]
    applies_when: dict | None = None       # {field: [allowed values]}
    fields: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "Rule":
        if "checks" in d:
            steps = d["checks"]
        else:
            steps = [{"check": d["check"], "fields": d["fields"], "params": d.get("params", {})}]
        for s in steps:
            if s["check"] not in CHECKS:
                raise ValueError(f"Rule {d['id']}: unknown check '{s['check']}'")
        sev = d.get("severity", "MEDIUM").upper()
        if sev not in SEVERITIES:
            raise ValueError(f"Rule {d['id']}: severity must be one of {SEVERITIES}")
        all_fields = list(dict.fromkeys(f for s in steps for f in s["fields"]))
        return cls(d["id"], d["name"], sev, d["message"], steps, d.get("applies_when"), all_fields)


@dataclass
class RulePack:
    regime: str
    version: str
    description: str
    rules: list[Rule]

    @classmethod
    def load(cls, name_or_path: str | Path) -> "RulePack":
        p = Path(name_or_path)
        if not p.suffix:
            p = RULES_DIR / f"{name_or_path}.yaml"
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
        return cls(data["regime"], str(data.get("version", "1")), data.get("description", ""),
                   [Rule.from_dict(r) for r in data["rules"]])


def run_rule(rule: Rule, df: pd.DataFrame, ref: dict) -> pd.Series:
    """Return a boolean fail mask for one rule."""
    in_scope = pd.Series(True, index=df.index)
    if rule.applies_when:
        for f, allowed in rule.applies_when.items():
            in_scope &= df[f].astype(str).isin([str(a) for a in allowed])
    fail = pd.Series(False, index=df.index)
    for step in rule.steps:
        fail |= CHECKS[step["check"]](df, step["fields"], step.get("params", {}), ref).fillna(True)
    return fail & in_scope


def validate(df: pd.DataFrame, pack: RulePack) -> pd.DataFrame:
    """Run every rule in the pack. Returns one row per (rule, record) failure."""
    ref = load_reference()
    df = df.copy()
    rows = []
    for rule in pack.rules:
        missing = [f for f in rule.fields if f not in df.columns]
        if missing:
            raise KeyError(f"Rule {rule.id} needs columns not in the file: {missing}")
        fail = run_rule(rule, df, ref)
        for idx in df.index[fail]:
            rec = df.loc[idx]
            rows.append({
                "regime": pack.regime,
                "rule_id": rule.id,
                "rule_name": rule.name,
                "severity": rule.severity,
                "record_id": rec["record_id"],
                "report_date": rec.get("report_date", ""),
                "fields": ", ".join(rule.fields),
                "values": " | ".join(f"{f}={'' if pd.isna(rec[f]) else rec[f]}" for f in rule.fields),
                "message": rule.message,
            })
    return pd.DataFrame(rows, columns=EXCEPTION_COLUMNS)
