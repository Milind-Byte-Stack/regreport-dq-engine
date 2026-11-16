"""Streamlit dashboard for the RegReport DQ Engine.

    streamlit run app.py
"""
from __future__ import annotations

import os
import tempfile
from datetime import date
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from dqengine.decisions import load_decisions, log_decision
from dqengine.engine import RulePack
from dqengine.pipeline import DEFAULT_FILES, detection_report, run_regime
from dqengine.report import build_excel
from dqengine.triage import triage
from synthetic import generate

st.set_page_config(page_title="RegReport DQ Engine", page_icon="✅", layout="wide")


# ------------------------------------------------------------------ helpers
def run_all(emir_path=None, mifir_path=None):
    st.session_state.results = [run_regime("emir", emir_path), run_regime("mifir", mifir_path)]
    st.session_state.synthetic = emir_path is None and mifir_path is None
    st.session_state.pop("groups", None)
    st.session_state.pop("triage_status", None)


def save_upload(upload) -> str:
    tmp = Path(tempfile.gettempdir()) / f"dq_{upload.name}"
    tmp.write_bytes(upload.getvalue())
    return str(tmp)


def all_exceptions() -> pd.DataFrame:
    return pd.concat([r.exceptions for r in st.session_state.results], ignore_index=True)


def all_data() -> pd.DataFrame:
    return pd.concat([r.data[["record_id", "source_system"]] if "source_system" in r.data
                      else r.data[["record_id"]].assign(source_system="UNKNOWN")
                      for r in st.session_state.results], ignore_index=True)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Data")
    mode = st.radio("Source", ["Synthetic data", "Upload CSVs"], label_visibility="collapsed")
    if mode == "Synthetic data":
        n = st.slider("Records per regime", 100, 5000, 500, step=100)
        rate = st.slider("Seeded error rate", 0.01, 0.20, 0.05, step=0.01)
        seed = st.number_input("Random seed", value=42, step=1)
        if st.button("Generate and run checks", type="primary", width="stretch"):
            generate.main(["--n", str(n), "--error-rate", str(rate), "--seed", str(int(seed))])
            run_all()
    else:
        emir_up = st.file_uploader("EMIR trades CSV", type="csv")
        mifir_up = st.file_uploader("MiFIR transactions CSV", type="csv")
        if st.button("Run checks", type="primary", width="stretch",
                     disabled=not (emir_up and mifir_up)):
            run_all(save_upload(emir_up), save_upload(mifir_up))

    st.divider()
    st.header("AI triage")
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    st.caption("Claude API key found." if has_key else
               "No ANTHROPIC_API_KEY set: triage uses offline rule-based grouping.")
    reviewer = st.text_input("Reviewer name", value=st.session_state.get("reviewer", ""))
    st.session_state.reviewer = reviewer

if "results" not in st.session_state:
    if not all(p.exists() for p in DEFAULT_FILES.values()):
        generate.main([])
    run_all()

results = st.session_state.results
emir, mifir = results

# ------------------------------------------------------------------ header
st.title("RegReport DQ Engine")
st.caption("EMIR and MiFIR data-quality checks with AI exception triage · **Synthetic data only**")

cols = st.columns(4)
cols[0].metric("EMIR DQ score", f"{emir.overall}%", help="Share of trades with no exceptions")
cols[1].metric("MiFIR DQ score", f"{mifir.overall}%", help="Share of transactions with no exceptions")
cols[2].metric("Exceptions", f"{len(emir.exceptions) + len(mifir.exceptions)}")
cols[3].metric("Records checked", f"{len(emir.data) + len(mifir.data):,}",
               help=f"{emir.rule_count + mifir.rule_count} rules across two rule packs")

tab_over, tab_ex, tab_ai, tab_log, tab_rules = st.tabs(
    ["Overview", "Exceptions", "AI triage", "Decision log", "Rule packs"])

# ------------------------------------------------------------------ overview
with tab_over:
    left, right = st.columns(2)
    for col, r in ((left, emir), (right, mifir)):
        with col:
            st.subheader(r.regime)
            st.caption(f"{len(r.data)} records · {len(r.exceptions)} exceptions on "
                       f"{r.exceptions['record_id'].nunique()} records · {r.rule_count} rules")
            st.markdown("**DQ score by report date**")
            lo = max(0, min(90, r.daily["dq_score"].min() - 5))
            chart = alt.Chart(r.daily).mark_bar(clip=True).encode(
                x=alt.X("report_date:N", title=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y("dq_score:Q", title="DQ score %", scale=alt.Scale(domain=[lo, 100])),
                tooltip=["report_date", "records", "failed_records", "dq_score"])
            st.altair_chart(chart.properties(height=220), width="stretch")
            st.markdown("**Exceptions by rule**")
            st.dataframe(r.rules[["rule_id", "rule_name", "severity", "exceptions"]],
                         hide_index=True, width="stretch")
            st.markdown("**Weakest fields**")
            st.dataframe(r.fields[r.fields["failed_records"] > 0].head(8),
                         hide_index=True, width="stretch")
            if st.session_state.get("synthetic"):
                det = detection_report(r)
                if det is not None:
                    st.caption(f"Seeded errors caught: {int(det['detected'].sum())} of {len(det)}")

# ------------------------------------------------------------------ exceptions
with tab_ex:
    ex = all_exceptions().merge(all_data(), on="record_id", how="left")
    f1, f2, f3, f4 = st.columns(4)
    regimes = f1.multiselect("Regime", sorted(ex["regime"].unique()), default=sorted(ex["regime"].unique()))
    sevs = f2.multiselect("Severity", ["HIGH", "MEDIUM", "LOW"], default=["HIGH", "MEDIUM", "LOW"])
    rules = f3.multiselect("Rule", sorted(ex["rule_id"].unique()))
    systems = f4.multiselect("Source system", sorted(ex["source_system"].dropna().unique()))
    view = ex[ex["regime"].isin(regimes) & ex["severity"].isin(sevs)]
    if rules:
        view = view[view["rule_id"].isin(rules)]
    if systems:
        view = view[view["source_system"].isin(systems)]
    st.caption(f"{len(view)} exceptions shown")
    st.dataframe(view[["regime", "rule_id", "severity", "record_id", "report_date", "source_system",
                       "values", "message"]], hide_index=True, width="stretch", height=480)
    st.download_button("Download exceptions CSV", view.to_csv(index=False), "exceptions.csv", "text/csv")

# ------------------------------------------------------------------ AI triage
with tab_ai:
    st.markdown("The engine sends a **summary** of the exceptions (counts by rule and source system, "
                "a few samples) to the LLM, which groups them by likely root cause and drafts a fix. "
                "Record counts are computed by the engine, not the model. Every group needs a human decision.")
    if st.button("Run AI triage", type="primary"):
        with st.spinner("Triaging exceptions..."):
            groups, status = triage(all_exceptions(), all_data())
        st.session_state.groups, st.session_state.triage_status = groups, status

    groups = st.session_state.get("groups")
    if groups is not None:
        st.info(st.session_state.triage_status)
        decided = load_decisions()
        latest = decided.drop_duplicates("group_id", keep="last").set_index("group_id")["decision"] \
            if not decided.empty else pd.Series(dtype=str)
        for g in groups.to_dict("records"):
            badge = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "⚪"}.get(g["confidence"], "⚪")
            with st.container(border=True):
                top, act = st.columns([4, 1.4])
                with top:
                    st.markdown(f"**{g['group_id']} · {g['title']}**  \n"
                                f"{g['regime']} · rules {g['rule_ids']} · {g['source_systems']} · "
                                f"**{g['record_count']} record(s)** · {badge} {g['confidence']} confidence")
                    st.markdown(f"**Likely root cause:** {g['likely_root_cause'] or '-'}  \n"
                                f"**Suggested fix:** {g['suggested_fix']}  \n"
                                f"**Owner:** {g['owner'] or '-'}")
                    with st.expander("Records"):
                        st.write(g["record_ids"])
                with act:
                    comment = st.text_input("Comment", key=f"c_{g['group_id']}", label_visibility="collapsed",
                                            placeholder="Comment (optional)")
                    a, r = st.columns(2)
                    if a.button("Approve", key=f"a_{g['group_id']}", width="stretch"):
                        log_decision(g, "APPROVED", st.session_state.reviewer, comment)
                        st.rerun()
                    if r.button("Reject", key=f"r_{g['group_id']}", width="stretch"):
                        log_decision(g, "REJECTED", st.session_state.reviewer, comment)
                        st.rerun()
                    if g["group_id"] in latest.index:
                        st.caption(f"Decision: **{latest[g['group_id']]}**")

# ------------------------------------------------------------------ decision log
with tab_log:
    log = load_decisions()
    if log.empty:
        st.write("No decisions logged yet.")
    else:
        st.dataframe(log.iloc[::-1], hide_index=True, width="stretch")
        st.download_button("Download decision log", log.to_csv(index=False), "decisions_log.csv", "text/csv")

# ------------------------------------------------------------------ rule packs
with tab_rules:
    st.markdown("Rules live in readable YAML files under `config/rules/`. "
                "Adding a regime means adding a rule pack, not changing the engine.")
    for name in ("emir", "mifir"):
        pack = RulePack.load(name)
        st.subheader(f"{pack.regime} v{pack.version}")
        st.dataframe(pd.DataFrame([{"id": r.id, "name": r.name, "severity": r.severity,
                                    "fields": ", ".join(r.fields),
                                    "checks": ", ".join(s["check"] for s in r.steps),
                                    "message": r.message} for r in pack.rules]),
                     hide_index=True, width="stretch")

# ------------------------------------------------------------------ excel
st.divider()
xlsx = build_excel(results, st.session_state.get("groups"), load_decisions())
st.download_button("Download daily Excel exceptions report", xlsx,
                   f"dq_exceptions_report_{date.today():%Y%m%d}.xlsx",
                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary")
