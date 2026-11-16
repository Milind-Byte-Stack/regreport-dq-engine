"""Synthetic data generator for EMIR trades and MiFIR transactions.

All entities, identifiers and trades are made up. LEIs use the prefix
"SYNT00" and ISINs use "SYN" in the issuer code so they are clearly fake,
while still passing the real check-digit algorithms.

Errors are seeded on purpose (default ~5% of records) and written to a
ground-truth file so the engine's detection rate can be measured. Some error
types are deliberately concentrated in one source system, the way real
breaks usually cluster, which gives the AI triage step something to find.

Usage:
    python -m synthetic.generate --n 500 --error-rate 0.05 --seed 42
"""
from __future__ import annotations

import argparse
import random
import string
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from dqengine.identifiers import isin_check_digit, lei_check_digits

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "input"
ALNUM = string.ascii_uppercase + string.digits


# ---------------------------------------------------------------- helpers
def make_lei(rng: random.Random) -> str:
    base = "SYNT00" + "".join(rng.choices(ALNUM, k=12))
    return base + lei_check_digits(base)


def make_isin(rng: random.Random, country: str) -> str:
    base = country + "SYN" + "".join(rng.choices(ALNUM, k=6))
    return base + isin_check_digit(base)


def business_days(start: date, n: int) -> list[date]:
    days, d = [], start
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days


def utc_ts(d: date, rng: random.Random) -> str:
    t = datetime(d.year, d.month, d.day, rng.randint(7, 16), rng.randint(0, 59),
                 rng.randint(0, 59), rng.randint(0, 999) * 1000, tzinfo=timezone.utc)
    return t.strftime("%Y-%m-%dT%H:%M:%S.") + f"{t.microsecond // 1000:03d}Z"


def pick_source(rng, systems, hot_system=None, hot_share=0.7):
    if hot_system and rng.random() < hot_share:
        return hot_system
    return rng.choice(systems)


# ---------------------------------------------------------------- EMIR
EMIR_SYSTEMS = ["MUREX-EU", "CALYPSO-LDN", "SUMMIT-FX"]
# error type -> (rule it should trip, system it clusters in)
EMIR_ERRORS = {
    "lei_bad_check_digits": ("EMIR-001", "CALYPSO-LDN"),
    "lei_bad_length":       ("EMIR-001", None),
    "uti_missing":          ("EMIR-002", None),
    "uti_duplicate":        ("EMIR-002", "SUMMIT-FX"),
    "valuation_missing":    ("EMIR-003", "MUREX-EU"),
    "collateral_code_missing":    ("EMIR-004", "MUREX-EU"),
    "collateral_code_unexpected": ("EMIR-004", None),
    "mandatory_blank":      ("EMIR-005", None),
    "direction_invalid":    ("EMIR-006", "SUMMIT-FX"),
    "same_counterparty":    ("EMIR-006", None),
}


def generate_emir(n=500, error_rate=0.05, seed=42, start=date(2026, 11, 9)):
    rng = random.Random(seed)
    reporting = [make_lei(rng) for _ in range(3)]          # "our" entities
    others = [make_lei(rng) for _ in range(25)]            # clients/brokers
    days = business_days(start, 5)
    products = {"IR": ["Swap", "Option"], "CR": ["Swap"], "FX": ["Forward", "Swap", "Option"],
                "EQ": ["Swap", "Option"], "CO": ["Swap", "Forward"]}

    rows = []
    for i in range(1, n + 1):
        d = rng.choice(days)
        asset = rng.choice(list(products))
        rep = rng.choice(reporting)
        action = rng.choices(["NEWT", "MODI", "VALU", "TERM"], weights=[30, 10, 55, 5])[0]
        coll = rng.random() < 0.6
        ccy = rng.choice(["EUR", "USD", "GBP", "CHF", "JPY"])
        rows.append({
            "record_id": f"E{i:05d}",
            "report_date": d.isoformat(),
            "source_system": rng.choice(EMIR_SYSTEMS),
            "uti": rep + "".join(rng.choices(ALNUM, k=rng.randint(12, 24))),
            "reporting_cpty_lei": rep,
            "other_cpty_lei": rng.choice(others),
            "direction": rng.choice(["BYER", "SLLR"]),
            "action_type": action,
            "asset_class": asset,
            "product_type": rng.choice(products[asset]),
            "execution_timestamp": utc_ts(d - timedelta(days=rng.randint(0, 400)), rng),
            "maturity_date": (d + timedelta(days=rng.randint(30, 3650))).isoformat(),
            "notional": f"{rng.randint(1, 500) * 100000:.2f}",
            "notional_currency": ccy,
            "valuation_amount": "" if action == "TERM" else f"{rng.uniform(-2e6, 2e6):.2f}",
            "valuation_currency": "" if action == "TERM" else ccy,
            "valuation_date": "" if action == "TERM" else d.isoformat(),
            "collateral_portfolio_indicator": "TRUE" if coll else "FALSE",
            "collateral_portfolio_code": ("PF" + "".join(rng.choices(string.digits, k=6))) if coll else "",
        })
    df = pd.DataFrame(rows)

    truth = []
    n_err = max(1, round(n * error_rate))
    targets = rng.sample(range(n), n_err)
    types = list(EMIR_ERRORS)
    for k, idx in enumerate(targets):
        etype = types[k % len(types)]
        rule, hot = EMIR_ERRORS[etype]
        df.at[idx, "source_system"] = pick_source(rng, EMIR_SYSTEMS, hot)
        rid = df.at[idx, "record_id"]
        if etype == "lei_bad_check_digits":
            lei = df.at[idx, "other_cpty_lei"]
            df.at[idx, "other_cpty_lei"] = lei[:18] + f"{(int(lei[18:]) + 1) % 100:02d}"
        elif etype == "lei_bad_length":
            df.at[idx, "other_cpty_lei"] = df.at[idx, "other_cpty_lei"][:19]
        elif etype == "uti_missing":
            df.at[idx, "uti"] = ""
        elif etype == "uti_duplicate":
            src = rng.choice([j for j in range(n) if j not in targets])
            df.at[idx, "uti"] = df.at[src, "uti"]
            truth.append({"record_id": df.at[src, "record_id"], "error_type": "uti_duplicate (original)",
                          "expected_rule": rule})
        elif etype == "valuation_missing":
            df.at[idx, "action_type"] = "VALU"
            df.at[idx, "valuation_amount"] = ""
        elif etype == "collateral_code_missing":
            df.at[idx, "collateral_portfolio_indicator"] = "TRUE"
            df.at[idx, "collateral_portfolio_code"] = ""
        elif etype == "collateral_code_unexpected":
            df.at[idx, "collateral_portfolio_indicator"] = "FALSE"
            df.at[idx, "collateral_portfolio_code"] = "PF" + "".join(rng.choices(string.digits, k=6))
        elif etype == "mandatory_blank":
            df.at[idx, rng.choice(["maturity_date", "notional_currency", "product_type"])] = ""
        elif etype == "direction_invalid":
            df.at[idx, "direction"] = rng.choice(["BUY", "SELL", "B"])
        elif etype == "same_counterparty":
            df.at[idx, "other_cpty_lei"] = df.at[idx, "reporting_cpty_lei"]
        truth.append({"record_id": rid, "error_type": etype, "expected_rule": rule})
    return df, pd.DataFrame(truth)


# ---------------------------------------------------------------- MiFIR
MIFIR_SYSTEMS = ["OMS-EQ", "OMS-FI", "DMA-GW"]
MIFIR_ERRORS = {
    "isin_bad_check_digit": ("MIFIR-001", "OMS-FI"),
    "isin_bad_length":      ("MIFIR-001", None),
    "venue_not_mic":        ("MIFIR-002", "DMA-GW"),
    "timestamp_not_utc":    ("MIFIR-003", "DMA-GW"),
    "price_zero":           ("MIFIR-004", None),
    "quantity_negative":    ("MIFIR-004", "OMS-EQ"),
    "currency_minor_unit":  ("MIFIR-005", "OMS-EQ"),
    "buyer_lei_typed_concat": ("MIFIR-006", "OMS-EQ"),
    "concat_malformed":     ("MIFIR-006", None),
}
FIRST = ["ARJUN", "SOFIA", "LUCAS", "EMMA", "NOAH", "MIA", "LEO", "ANNA", "OMAR", "ELLA"]
LAST = ["FERNANDES", "MULLER", "ROSSI", "DUBOIS", "SILVA", "NOVAK", "JANSEN", "BERG", "LOPEZ", "KOWAL"]
VENUES = {"GB": ["XLON", "TRQX"], "FR": ["XPAR", "AQXE"], "DE": ["XETR", "XFRA"],
          "NL": ["XAMS", "CHIX", "BATE"], "IE": ["XDUB"]}
CCY = {"GB": "GBP", "FR": "EUR", "DE": "EUR", "NL": "EUR", "IE": "EUR"}


def make_concat(rng, country="FR"):
    dob = date(rng.randint(1955, 2000), rng.randint(1, 12), rng.randint(1, 28))
    first = rng.choice(FIRST)[:5].ljust(5, "#")
    last = rng.choice(LAST)[:5].ljust(5, "#")
    return f"{country}{dob:%Y%m%d}{first}{last}"


def generate_mifir(n=500, error_rate=0.05, seed=7, start=date(2026, 11, 9)):
    rng = random.Random(seed)
    firm = make_lei(rng)
    clients_lei = [make_lei(rng) for _ in range(20)]
    instruments = [(c, make_isin(rng, c)) for c in rng.choices(list(VENUES), k=40)]
    days = business_days(start, 5)

    def party():
        if rng.random() < 0.7:
            return "LEI", rng.choice(clients_lei)
        return "CONCAT", make_concat(rng, rng.choice(["FR", "DE", "NL", "IE"]))

    rows = []
    for i in range(1, n + 1):
        d = rng.choice(days)
        country, isin = rng.choice(instruments)
        is_bond = rng.random() < 0.3
        venue = rng.choice(VENUES[country] + ["XOFF"])
        b_type, b_id = party()
        side_firm = rng.random() < 0.5
        s_type, s_id = ("LEI", firm) if side_firm else party()
        if not side_firm:
            b_type, b_id = "LEI", firm
        rows.append({
            "record_id": f"M{i:05d}",
            "report_date": d.isoformat(),
            "source_system": rng.choice(MIFIR_SYSTEMS),
            "transaction_ref": f"TX{d:%Y%m%d}{i:06d}",
            "executing_entity_lei": firm,
            "instrument_type": "BOND" if is_bond else "EQUITY",
            "isin": isin,
            "venue": venue,
            "trading_datetime": utc_ts(d, rng),
            "price": f"{rng.uniform(95, 105):.4f}" if is_bond else f"{rng.uniform(5, 400):.2f}",
            "price_currency": CCY[country],
            "quantity": f"{rng.randint(1, 500) * 1000}" if is_bond else f"{rng.randint(1, 5000)}",
            "buyer_id_type": b_type, "buyer_id": b_id,
            "seller_id_type": s_type, "seller_id": s_id,
        })
    df = pd.DataFrame(rows)

    truth = []
    targets = rng.sample(range(n), max(1, round(n * error_rate)))
    types = list(MIFIR_ERRORS)
    for k, idx in enumerate(targets):
        etype = types[k % len(types)]
        rule, hot = MIFIR_ERRORS[etype]
        df.at[idx, "source_system"] = pick_source(rng, MIFIR_SYSTEMS, hot)
        if etype == "isin_bad_check_digit":
            isin = df.at[idx, "isin"]
            df.at[idx, "isin"] = isin[:11] + str((int(isin[11]) + 3) % 10)
        elif etype == "isin_bad_length":
            df.at[idx, "isin"] = df.at[idx, "isin"][:11]
        elif etype == "venue_not_mic":
            df.at[idx, "venue"] = rng.choice(["LSE", "XLDN", "EPA", "OTC"])
        elif etype == "timestamp_not_utc":
            ts = df.at[idx, "trading_datetime"]
            df.at[idx, "trading_datetime"] = rng.choice([ts[:19].replace("T", " "), ts[:23] + "+01:00"])
        elif etype == "price_zero":
            df.at[idx, "price"] = "0"
        elif etype == "quantity_negative":
            df.at[idx, "quantity"] = "-" + df.at[idx, "quantity"]
        elif etype == "currency_minor_unit":
            df.at[idx, "price_currency"] = rng.choice(["GBX", "EUD"])
        elif etype == "buyer_lei_typed_concat":
            df.at[idx, "buyer_id_type"] = "LEI"
            df.at[idx, "buyer_id"] = make_concat(rng)
        elif etype == "concat_malformed":
            df.at[idx, "seller_id_type"] = "CONCAT"
            df.at[idx, "seller_id"] = make_concat(rng)[:-3]          # truncated surname
        truth.append({"record_id": df.at[idx, "record_id"], "error_type": etype, "expected_rule": rule})
    return df, pd.DataFrame(truth)


# ---------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description="Generate synthetic EMIR and MiFIR report files.")
    ap.add_argument("--n", type=int, default=500, help="records per regime")
    ap.add_argument("--error-rate", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)

    emir, emir_truth = generate_emir(a.n, a.error_rate, a.seed)
    mifir, mifir_truth = generate_mifir(a.n, a.error_rate, a.seed + 1)
    emir.to_csv(a.out / "emir_trades.csv", index=False)
    emir_truth.to_csv(a.out / "emir_seeded_errors.csv", index=False)
    mifir.to_csv(a.out / "mifir_transactions.csv", index=False)
    mifir_truth.to_csv(a.out / "mifir_seeded_errors.csv", index=False)
    print(f"EMIR:  {len(emir)} trades, {len(emir_truth)} seeded error records -> {a.out}")
    print(f"MiFIR: {len(mifir)} transactions, {len(mifir_truth)} seeded error records -> {a.out}")


if __name__ == "__main__":
    main()
