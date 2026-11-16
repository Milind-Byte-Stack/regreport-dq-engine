"""Command-line run of the full pipeline.

    python run_pipeline.py                 # generate data if missing, validate, score, report
    python run_pipeline.py --regenerate    # fresh synthetic data first
    python run_pipeline.py --triage        # add AI triage (Claude if ANTHROPIC_API_KEY is set)
    python run_pipeline.py --emir path/to/file.csv --mifir path/to/file.csv
"""
from __future__ import annotations

import argparse
from datetime import date

import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from dqengine.pipeline import DEFAULT_FILES, OUTPUT_DIR, detection_report, run_regime, save_outputs
from dqengine.report import build_excel
from dqengine.triage import triage
from synthetic import generate


def main(argv=None):
    ap = argparse.ArgumentParser(description="RegReport DQ Engine")
    ap.add_argument("--emir", help="EMIR report CSV (default: synthetic file)")
    ap.add_argument("--mifir", help="MiFIR report CSV (default: synthetic file)")
    ap.add_argument("--regenerate", action="store_true", help="regenerate synthetic data first")
    ap.add_argument("--triage", action="store_true", help="run AI triage on the exceptions")
    a = ap.parse_args(argv)

    if a.regenerate or not all(p.exists() for p in DEFAULT_FILES.values()):
        generate.main([])

    results = [run_regime("emir", a.emir), run_regime("mifir", a.mifir)]
    paths = save_outputs(results)

    print("\n=== Data quality ===")
    for r in results:
        print(f"{r.regime:6} {len(r.data):5} records | {len(r.exceptions):3} exceptions on "
              f"{r.exceptions['record_id'].nunique():3} records | {r.rule_count} rules | DQ score {r.overall}%")
        det = None if (a.emir if r.regime == "EMIR" else a.mifir) else detection_report(r)
        if det is not None:
            print(f"       seeded errors caught: {int(det['detected'].sum())}/{len(det)}")

    groups = None
    if a.triage:
        ex = pd.concat([r.exceptions for r in results], ignore_index=True)
        data = pd.concat([r.data[["record_id", "source_system"]] for r in results], ignore_index=True)
        groups, status = triage(ex, data)
        print(f"\n=== AI triage === {status}")
        for g in groups.itertuples():
            print(f"{g.group_id} [{g.regime}] {g.title} - {g.record_count} record(s) ({g.confidence})")
        groups.to_csv(OUTPUT_DIR / "triage_groups.csv", index=False)

    xlsx = OUTPUT_DIR / f"dq_exceptions_report_{date.today():%Y%m%d}.xlsx"
    build_excel(results, groups, path=xlsx)
    print(f"\nOutputs: {paths['exceptions']}\n         {xlsx}")


if __name__ == "__main__":
    main()
