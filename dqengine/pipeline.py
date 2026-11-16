"""End-to-end pipeline: ingest -> validate -> score (-> triage -> report)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import scoring
from .engine import RulePack, validate
from .ingest import read_report

ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT / "data" / "input"
OUTPUT_DIR = ROOT / "data" / "output"

DEFAULT_FILES = {
    "emir": INPUT_DIR / "emir_trades.csv",
    "mifir": INPUT_DIR / "mifir_transactions.csv",
}
TRUTH_FILES = {
    "emir": INPUT_DIR / "emir_seeded_errors.csv",
    "mifir": INPUT_DIR / "mifir_seeded_errors.csv",
}


@dataclass
class RegimeResult:
    regime: str
    data: pd.DataFrame
    exceptions: pd.DataFrame
    overall: float
    fields: pd.DataFrame
    daily: pd.DataFrame
    rules: pd.DataFrame
    rule_count: int


def run_regime(regime: str, path: str | Path | None = None) -> RegimeResult:
    pack = RulePack.load(regime)
    df = read_report(path or DEFAULT_FILES[regime])
    ex = validate(df, pack)
    return RegimeResult(
        regime=pack.regime,
        data=df,
        exceptions=ex,
        overall=scoring.overall_score(df, ex),
        fields=scoring.field_scores(df, ex),
        daily=scoring.daily_scores(df, ex),
        rules=scoring.rule_summary(ex),
        rule_count=len(pack.rules),
    )


def detection_report(result: RegimeResult, truth_path: str | Path | None = None) -> pd.DataFrame | None:
    """Compare exceptions with the generator's seeded errors (synthetic data only).

    Returns one row per seeded error with whether its expected rule fired.
    """
    p = Path(truth_path or TRUTH_FILES[result.regime.lower()])
    if not p.exists():
        return None
    truth = pd.read_csv(p, dtype=str)
    hits = set(zip(result.exceptions["record_id"], result.exceptions["rule_id"]))
    truth["detected"] = [(r, rule) in hits for r, rule in zip(truth["record_id"], truth["expected_rule"])]
    return truth


def save_outputs(results: list[RegimeResult], out_dir: Path = OUTPUT_DIR) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    all_ex = pd.concat([r.exceptions for r in results], ignore_index=True)
    paths["exceptions"] = out_dir / "exceptions.csv"
    all_ex.to_csv(paths["exceptions"], index=False)
    for r in results:
        p = out_dir / f"{r.regime.lower()}_field_scores.csv"
        r.fields.to_csv(p, index=False)
        paths[f"{r.regime.lower()}_field_scores"] = p
    return paths
