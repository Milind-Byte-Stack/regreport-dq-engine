import pandas as pd
import pytest

from dqengine import identifiers as ids
from dqengine.engine import RulePack, validate
from dqengine.scoring import field_scores, overall_score
from dqengine.triage import _attach_members, _extract_json, offline_groups
from synthetic.generate import generate_emir, generate_mifir, make_concat, make_lei


# ---------- identifiers ----------
@pytest.mark.parametrize("isin", ["US0378331005", "GB0002634946", "DE0007164600"])
def test_known_valid_isins(isin):
    assert ids.is_valid_isin(isin)


@pytest.mark.parametrize("isin", ["US0378331006", "US037833100", "", None, "us0378331005x"])
def test_invalid_isins(isin):
    assert not ids.is_valid_isin(isin)


def test_lei_roundtrip():
    import random
    lei = make_lei(random.Random(1))
    assert ids.is_valid_lei(lei)
    assert not ids.is_valid_lei(lei[:18] + f"{(int(lei[18:]) + 1) % 100:02d}")
    assert not ids.is_valid_lei(lei[:19])


def test_concat_and_timestamps():
    import random
    assert ids.is_valid_concat(make_concat(random.Random(3)))
    assert not ids.is_valid_concat("FR19801301JOHN#SMITH")      # month 13
    assert ids.is_utc_timestamp("2026-11-09T09:30:15.123Z")
    assert not ids.is_utc_timestamp("2026-11-09 09:30:15")
    assert not ids.is_utc_timestamp("2026-11-09T09:30:15.123+01:00")


# ---------- rule packs on synthetic data ----------
@pytest.mark.parametrize("regime,gen", [("emir", generate_emir), ("mifir", generate_mifir)])
def test_all_seeded_errors_caught_and_no_false_positives(regime, gen):
    df, truth = gen(n=1000, error_rate=0.08, seed=11)
    ex = validate(df, RulePack.load(regime))
    hits = set(zip(ex["record_id"], ex["rule_id"]))
    missed = [(r, rule) for r, rule in zip(truth["record_id"], truth["expected_rule"]) if (r, rule) not in hits]
    assert missed == []
    assert set(ex["record_id"]) <= set(truth["record_id"])


@pytest.mark.parametrize("regime,gen", [("emir", generate_emir), ("mifir", generate_mifir)])
def test_clean_data_has_no_exceptions(regime, gen):
    df, _ = gen(n=300, error_rate=0.0001, seed=5)   # rounds to 1 seeded error minimum
    ex = validate(df, RulePack.load(regime))
    assert ex["record_id"].nunique() <= 2


def test_scores():
    df, _ = generate_emir(n=200, error_rate=0.05, seed=2)
    ex = validate(df, RulePack.load("emir"))
    assert 0 < overall_score(df, ex) < 100
    fs = field_scores(df, ex)
    assert fs["dq_score"].between(0, 100).all()


# ---------- triage ----------
def test_offline_triage_covers_every_exception():
    df, _ = generate_mifir(n=500, error_rate=0.06, seed=9)
    ex = validate(df, RulePack.load("mifir"))
    groups = _attach_members(offline_groups(ex, df), ex, df, "offline")
    covered = {r for ids_ in groups["record_ids"] for r in ids_.split(", ")}
    assert covered == set(ex["record_id"])


def test_llm_groups_with_unknown_rules_fall_into_manual_review():
    df, _ = generate_emir(n=300, error_rate=0.05, seed=4)
    ex = validate(df, RulePack.load("emir"))
    fake = [{"title": "x", "regime": "EMIR", "rule_ids": ["EMIR-001"], "source_systems": ["ALL"]}]
    groups = _attach_members(fake, ex, df, "claude")
    assert groups.iloc[-1]["title"].startswith("Not grouped")
    assert groups["record_count"].sum() >= ex["record_id"].nunique()


def test_extract_json_from_fenced_reply():
    assert _extract_json('```json\n{"groups": []}\n```') == {"groups": []}
