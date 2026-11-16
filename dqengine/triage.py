"""AI triage: group exceptions by likely root cause and draft fixes.

How it works
1. Exceptions are summarised per rule (counts, split by source system,
   a few sample values). Only this summary goes to the LLM, never the full file.
2. The LLM returns JSON groups. Each group names the rule IDs and source
   systems it covers, a likely root cause, and a suggested fix.
3. Group membership is then computed by the engine from those rule/system
   filters, so record counts are exact and never invented by the model.
4. A human approves or rejects each group (see decisions.py).

If no ANTHROPIC_API_KEY is set, an offline rule-based grouping is used so
the demo still runs. Groups are labelled with their source either way.
"""
from __future__ import annotations

import json
import os
import re

import pandas as pd

DEFAULT_MODEL = "claude-sonnet-4-5"

GROUP_COLUMNS = [
    "group_id", "regime", "title", "rule_ids", "source_systems", "record_count",
    "likely_root_cause", "suggested_fix", "owner", "confidence", "record_ids", "source",
]

SYSTEM_PROMPT = """You are a senior regulatory reporting data-quality analyst \
for EMIR and MiFIR. You receive a summary of validation exceptions from a \
data-quality engine. Group them by likely ROOT CAUSE (not just by rule): \
exceptions that share an upstream cause, e.g. one source system mapping \
issue, belong together even across rules; one rule can split into several \
groups if the causes differ.

For each group give a short title, the likely root cause, a concrete \
suggested fix (what to change, where, and whether a resubmission is needed), \
the team most likely to own it, and your confidence (HIGH/MEDIUM/LOW). Be \
specific and practical. Do not invent record IDs or counts.

Reply with JSON only, no prose, in exactly this shape:
{"groups": [{"title": str, "regime": "EMIR"|"MiFIR", "rule_ids": [str], \
"source_systems": [str] or ["ALL"], "likely_root_cause": str, \
"suggested_fix": str, "owner": str, "confidence": "HIGH"|"MEDIUM"|"LOW"}]}"""

# Offline fallback knowledge, keyed by rule ID.
OFFLINE_PLAYBOOK = {
    "EMIR-001": ("Counterparty static data holds invalid LEIs",
                 "Correct the LEI in counterparty static data (check GLEIF), then resubmit affected trades.",
                 "Client Reference Data"),
    "EMIR-002": ("UTI generation or sharing is broken",
                 "Regenerate missing UTIs per the UTI waterfall and remove duplicate submissions; "
                 "agree UTIs with counterparties before resubmitting.", "Trade Support"),
    "EMIR-003": ("Valuation feed did not populate open trades",
                 "Re-run the end-of-day valuation feed for the affected date and resend VALU reports.",
                 "Valuations / Product Control"),
    "EMIR-004": ("Collateral portfolio mapping is inconsistent",
                 "Align the portfolio indicator with the collateral agreement and populate or clear the "
                 "portfolio code; resend as MODI.", "Collateral Operations"),
    "EMIR-005": ("Mandatory trade attributes missing at booking",
                 "Fix the trade in the booking system and add a mandatory-field check to the booking screen.",
                 "Front Office Support"),
    "EMIR-006": ("Direction or counterparty mapping error",
                 "Map internal buy/sell codes to BYER/SLLR and stop internal entities being used as "
                 "the other counterparty.", "Regulatory Reporting IT"),
    "MIFIR-001": ("Instrument reference data holds bad ISINs",
                  "Correct the ISIN from the instrument master / ANNA source and resubmit.",
                  "Instrument Reference Data"),
    "MIFIR-002": ("Venue codes not mapped to ISO 10383 MICs",
                  "Map exchange short names to segment MICs (or XOFF/XXXX) in the venue mapping table.",
                  "Regulatory Reporting IT"),
    "MIFIR-003": ("Timestamps sent in local time",
                  "Convert execution timestamps to UTC at source and add a 'Z' suffix check.",
                  "Trading Technology"),
    "MIFIR-004": ("Price or quantity captured incorrectly",
                  "Fix sign conventions for quantity and block zero prices at capture; resubmit.",
                  "Middle Office"),
    "MIFIR-005": ("Minor currency units reported as currency",
                  "Report price in the major currency (e.g. GBP not GBX) and convert the price; resubmit.",
                  "Regulatory Reporting IT"),
    "MIFIR-006": ("Client identifier type does not match the ID",
                  "Use LEI for legal entities and a correctly built CONCAT/national ID for natural persons; "
                  "fix the client onboarding record.", "Client Onboarding / KYC"),
}


# ------------------------------------------------------------------ payload
def build_payload(exceptions: pd.DataFrame, data: pd.DataFrame, max_samples: int = 6) -> dict:
    """Compact, model-friendly summary of the exceptions."""
    ex = exceptions.merge(data[["record_id", "source_system"]], on="record_id", how="left") \
        if "source_system" in data.columns else exceptions.assign(source_system="UNKNOWN")
    rules = []
    for (regime, rule_id), g in ex.groupby(["regime", "rule_id"]):
        rules.append({
            "regime": regime,
            "rule_id": rule_id,
            "rule_name": g["rule_name"].iloc[0],
            "severity": g["severity"].iloc[0],
            "message": g["message"].iloc[0],
            "exception_count": int(len(g)),
            "by_source_system": {k: int(v) for k, v in g["source_system"].value_counts().items()},
            "samples": g.head(max_samples)[["record_id", "source_system", "values"]].to_dict("records"),
        })
    return {"total_exceptions": int(len(ex)), "rules": rules}


# ------------------------------------------------------------------ membership
def _attach_members(groups: list[dict], exceptions: pd.DataFrame, data: pd.DataFrame,
                    source: str) -> pd.DataFrame:
    ex = exceptions.merge(data[["record_id", "source_system"]], on="record_id", how="left")
    ex["assigned"] = False
    rows = []
    for g in groups:
        systems = [s.upper() for s in g.get("source_systems") or ["ALL"]]
        m = ex["rule_id"].isin(g.get("rule_ids", [])) & ~ex["assigned"]
        if "ALL" not in systems:
            m &= ex["source_system"].str.upper().isin(systems)
        if not m.any():
            continue
        ex.loc[m, "assigned"] = True
        ids = sorted(ex.loc[m, "record_id"].unique())
        rows.append({
            "group_id": f"G{len(rows) + 1:02d}",
            "regime": g.get("regime") or ex.loc[m, "regime"].iloc[0],
            "title": g.get("title", ""),
            "rule_ids": ", ".join(g.get("rule_ids", [])),
            "source_systems": ", ".join(sorted(ex.loc[m, "source_system"].unique())),
            "record_count": len(ids),
            "likely_root_cause": g.get("likely_root_cause", ""),
            "suggested_fix": g.get("suggested_fix", ""),
            "owner": g.get("owner", ""),
            "confidence": str(g.get("confidence", "")).upper(),
            "record_ids": ", ".join(ids),
            "source": source,
        })
    left = ex[~ex["assigned"]]
    if not left.empty:
        ids = sorted(left["record_id"].unique())
        rows.append({
            "group_id": f"G{len(rows) + 1:02d}", "regime": ", ".join(sorted(left["regime"].unique())),
            "title": "Not grouped - needs manual review", "rule_ids": ", ".join(sorted(left["rule_id"].unique())),
            "source_systems": ", ".join(sorted(left["source_system"].dropna().unique())),
            "record_count": len(ids), "likely_root_cause": "", "suggested_fix": "Review individually.",
            "owner": "", "confidence": "LOW", "record_ids": ", ".join(ids), "source": source,
        })
    return pd.DataFrame(rows, columns=GROUP_COLUMNS)


# ------------------------------------------------------------------ offline
def offline_groups(exceptions: pd.DataFrame, data: pd.DataFrame, hot_share: float = 0.6) -> list[dict]:
    """Rule-based grouping: one group per rule, split out a dominant source system."""
    ex = exceptions.merge(data[["record_id", "source_system"]], on="record_id", how="left")
    groups = []
    for (regime, rule_id), g in ex.groupby(["regime", "rule_id"]):
        title, fix, owner = OFFLINE_PLAYBOOK.get(rule_id, (g["rule_name"].iloc[0], "Review.", ""))
        counts = g["source_system"].value_counts()
        top, share = counts.index[0], counts.iloc[0] / len(g)
        if share >= hot_share and len(g) >= 2:
            groups.append({"title": f"{title} ({top})", "regime": regime, "rule_ids": [rule_id],
                           "source_systems": [top],
                           "likely_root_cause": f"{share:.0%} of {rule_id} breaks come from {top}, "
                                                f"which points to a mapping or feed issue in that system.",
                           "suggested_fix": fix, "owner": owner, "confidence": "MEDIUM"})
        groups.append({"title": title, "regime": regime, "rule_ids": [rule_id], "source_systems": ["ALL"],
                       "likely_root_cause": "Isolated data-entry or static-data errors.",
                       "suggested_fix": fix, "owner": owner, "confidence": "LOW"})
    return groups


# ------------------------------------------------------------------ Claude
def _extract_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    start, end = text.find("{"), text.rfind("}")
    return json.loads(text[start:end + 1])


def claude_groups(payload: dict, model: str | None = None, api_key: str | None = None) -> tuple[list[dict], str]:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
    model = model or os.environ.get("ANTHROPIC_MODEL") or DEFAULT_MODEL
    msg = [{"role": "user", "content": "Exception summary:\n" + json.dumps(payload, indent=1)}]
    try:
        resp = client.messages.create(model=model, max_tokens=4000, system=SYSTEM_PROMPT, messages=msg)
    except anthropic.NotFoundError:
        # Model name retired: fall back to the newest model the key can use.
        model = next(iter(client.models.list())).id
        resp = client.messages.create(model=model, max_tokens=4000, system=SYSTEM_PROMPT, messages=msg)
    text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
    return _extract_json(text)["groups"], model


def triage(exceptions: pd.DataFrame, data: pd.DataFrame, use_ai: bool | None = None,
           model: str | None = None) -> tuple[pd.DataFrame, str]:
    """Return (groups DataFrame, status message)."""
    if exceptions.empty:
        return pd.DataFrame(columns=GROUP_COLUMNS), "No exceptions to triage."
    if use_ai is None:
        use_ai = bool(os.environ.get("ANTHROPIC_API_KEY"))
    if use_ai:
        try:
            groups, used = claude_groups(build_payload(exceptions, data), model=model)
            return _attach_members(groups, exceptions, data, f"claude ({used})"), f"AI triage by {used}."
        except Exception as e:                       # keep the demo running
            status = f"AI triage failed ({type(e).__name__}: {e}); used offline grouping."
            return _attach_members(offline_groups(exceptions, data), exceptions, data, "offline"), status
    return (_attach_members(offline_groups(exceptions, data), exceptions, data, "offline"),
            "No ANTHROPIC_API_KEY set; used offline rule-based grouping.")
