from __future__ import annotations

from typing import Callable

import pandas as pd
import streamlit as st

from engine.scoring import workflow_date_range_text as _workflow_date_range_text
from ui.common import (
    _html_escape,
    _numeric_rate,
    _nonnull_rate,
    _rate_status,
    _render_card_grid,
    clean_metric_card,
    safe_dataframe,
    section_anchor,
)
from ui.methodology import render_methodology_trust_panel


def _status_label(status: str) -> str:
    """Local fallback so this module does not depend on a private common helper."""
    return {
        "good": "Green",
        "warn": "Yellow",
        "bad": "Red",
        "neutral": "Info",
    }.get(status, "Info")


def render_upload_file_cards(
    trade_file_names: list[str],
    bond_file_name: str | None,
    issuer_mapping_file_name: str | None,
    mmd_file_name: str | None,
    use_external_mmd_fallback: bool,
):
    """Show selected upload files as role-based cards before the heavier audit."""
    trade_count = len(trade_file_names)
    cards = [
        {
            "class_name": "file-card",
            "status": "good" if trade_count else "bad",
            "kicker": "Required",
            "title": "Trade files",
            "value": f"{trade_count:,} selected" if trade_count else "No trade file selected",
            "detail": ", ".join(trade_file_names[:2]) + (" ..." if trade_count > 2 else "") if trade_count else "Required.",
        },
        {
            "class_name": "file-card",
            "status": "good" if bond_file_name else "neutral",
            "kicker": "Optional",
            "title": "Bond reference",
            "value": bond_file_name or "Not uploaded",
            "detail": "Security metadata.",
        },
        {
            "class_name": "file-card",
            "status": "good" if issuer_mapping_file_name else "neutral",
            "kicker": "Optional",
            "title": "Issuer mapping",
            "value": issuer_mapping_file_name or "Not uploaded",
            "detail": "Issuer / sector labels.",
        },
    ]
    if mmd_file_name or use_external_mmd_fallback:
        cards.append(
            {
                "class_name": "file-card",
                "status": "good" if mmd_file_name else "warn",
                "kicker": "Optional benchmark",
                "title": "MMD / AAA curve",
                "value": mmd_file_name or "Fallback enabled; no file selected",
                "detail": "AAA fallback curve.",
            }
        )
    _render_card_grid(cards, "file-card-grid")


def build_upload_audit_cards(
    trade_reports: list[dict],
    market_df: pd.DataFrame,
    benchmark_source_mode: str,
    use_external_mmd_fallback: bool,
    mmd_file_provided: bool,
) -> tuple[list[dict], dict]:
    """Return red/yellow/green audit cards and a compact readiness summary."""
    required_ok = bool(trade_reports) and all(report.get("can_run") for report in trade_reports)
    required_missing = sorted(
        {
            missing
            for report in trade_reports
            for missing in report.get("missing_required", [])
        }
    )

    trade_date_rate = _nonnull_rate(market_df, "trade_date")
    dates = pd.to_datetime(market_df["trade_date"], errors="coerce").dropna() if "trade_date" in market_df.columns else pd.Series(dtype="datetime64[ns]")
    if dates.empty:
        date_status = "bad"
        date_value = "No valid trade dates"
        date_detail = "Trade date is required for time-series charts and snapshot period filters."
    else:
        unique_dates = dates.dt.normalize().nunique()
        date_status = "good" if trade_date_rate and trade_date_rate >= 95 and unique_dates >= 2 else "warn"
        date_value = f"{dates.min():%m/%d/%Y} - {dates.max():%m/%d/%Y}"
        date_detail = f"{trade_date_rate:.1f}% valid date rows across {unique_dates:,} unique date(s)."

    cusip_rate = _nonnull_rate(market_df, "cusip")
    cusip_status = _rate_status(cusip_rate, good_threshold=95, warn_threshold=80)

    yield_rate = _numeric_rate(market_df, "yield")
    index_rate = _numeric_rate(market_df, "index_rate")
    spread_rate = _numeric_rate(market_df, "spread")
    yield_status = _rate_status(yield_rate, good_threshold=90, warn_threshold=70)

    best_benchmark_input_rate = max([x for x in [index_rate, spread_rate] if x is not None], default=0)
    if benchmark_source_mode in {"Trade Sheet Index / Index Rate", "Uploaded MMD fallback"}:
        benchmark_status = "good"
    elif use_external_mmd_fallback and not mmd_file_provided:
        benchmark_status = "warn"
    else:
        benchmark_status = "bad"

    if benchmark_source_mode == "Trade Sheet Index / Index Rate":
        benchmark_value = "Trade sheet Index / Index Rate"
        benchmark_detail = "Primary benchmark is active. External MMD is not mixed into this run."
    elif benchmark_source_mode == "Uploaded MMD fallback":
        benchmark_value = "Uploaded MMD fallback"
        benchmark_detail = "Uploaded MMD is active as the AAA benchmark because trade index data was unavailable."
    else:
        benchmark_value = "No active benchmark"
        benchmark_detail = "Yield/liquidity analytics can run, but spread-to-benchmark views are degraded."

    if yield_status == "bad":
        spread_input_status = "bad"
    elif best_benchmark_input_rate >= 70:
        spread_input_status = "good"
    elif benchmark_status == "good":
        spread_input_status = "warn"
    else:
        spread_input_status = "bad"

    cards = [
        {
            "status": "good" if required_ok else "bad",
            "kicker": "Required fields",
            "title": "Minimum schema",
            "value": "Pass" if required_ok else "Blocking issue",
            "detail": "All required trade fields were detected." if required_ok else "Missing: " + ", ".join(required_missing),
        },
        {
            "status": date_status,
            "kicker": "Date coverage",
            "title": "Trade date window",
            "value": date_value,
            "detail": date_detail,
        },
        {
            "status": cusip_status,
            "kicker": "CUSIP quality",
            "title": "Valid CUSIP rate",
            "value": "N/A" if cusip_rate is None else f"{cusip_rate:.1f}%",
            "detail": "CUSIP-level drilldown and watchlist depend on this field.",
        },
        {
            "status": benchmark_status,
            "kicker": "Benchmark source",
            "title": "Active curve policy",
            "value": benchmark_value,
            "detail": benchmark_detail,
        },
        {
            "status": yield_status,
            "kicker": "Yield availability",
            "title": "Numeric yield rows",
            "value": "N/A" if yield_rate is None else f"{yield_rate:.1f}%",
            "detail": "Yield is required for spread, curve, and RV calculations.",
        },
        {
            "status": spread_input_status,
            "kicker": "Spread inputs",
            "title": "Index Rate / Spread",
            "value": f"Index {index_rate or 0:.1f}% / Spread {spread_rate or 0:.1f}%",
            "detail": "Trade Index Rate is preferred; uploaded MMD can fill the benchmark role only as fallback.",
        },
    ]

    blocking = [c for c in cards if c["status"] == "bad" and c["kicker"] in {"Required fields", "Date coverage", "Yield availability"}]
    benchmark_missing = benchmark_status == "bad"
    if blocking:
        ready_status = "bad"
        ready_value = "Not ready"
        next_step = "Fix blocking fields."
    elif benchmark_missing:
        ready_status = "warn"
        ready_value = "Yield / liquidity only"
        next_step = "Add benchmark for spread/RV."
    elif any(c["status"] == "warn" for c in cards):
        ready_status = "warn"
        ready_value = "Ready with warnings"
        next_step = "Snapshot next; review warnings."
    else:
        ready_status = "good"
        ready_value = "Ready to analyze"
        next_step = "Open Desk Snapshot."

    readiness = {
        "status": ready_status,
        "value": ready_value,
        "next_step": next_step,
        "bad_count": sum(1 for c in cards if c["status"] == "bad"),
        "warn_count": sum(1 for c in cards if c["status"] == "warn"),
    }
    return cards, readiness


def render_ready_to_analyze_card(readiness: dict):
    status = readiness.get("status", "neutral")
    st.markdown(
        f"""
<div class="ready-card status-{_html_escape(status)}">
  <div class="status-pill">{_html_escape(_status_label(status))}</div>
  <div class="card-kicker">Ready to Analyze</div>
  <div class="card-title">{_html_escape(readiness.get('value', 'Review upload'))}</div>
  <div class="card-value">{_html_escape(readiness.get('next_step', 'Review the audit cards above.'))}</div>
  <div class="card-detail">Warn {_html_escape(readiness.get('warn_count', 0))} | Block {_html_escape(readiness.get('bad_count', 0))}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_focused_upload_audit(
    trade_reports: list[dict],
    bond_report: dict | None,
    mmd_report: dict | None,
    market_df: pd.DataFrame,
    bonds_df: pd.DataFrame,
    issuer_master: pd.DataFrame,
    mmd_df: pd.DataFrame,
    trade_payloads: list[tuple[str, bytes]],
    failed_files: list[str],
    duplicates_removed: int,
    benchmark_source_mode: str,
    benchmark_priority: str,
    benchmark_conflict_policy: str,
    use_external_mmd_fallback: bool,
    mmd_file_provided: bool,
    render_benchmark_methodology_block: Callable[[pd.DataFrame, str, str, str], None] | None = None,
):
    section_anchor("workflow-upload-audit", "Upload / Data Audit")
    st.markdown(
        "<div class='focus-band'><b>Check:</b> files, dates, CUSIP, benchmark.</div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        clean_metric_card("Trade Files", f"{len(trade_payloads):,}", size="small")
    with c2:
        clean_metric_card("Trade Rows", f"{len(market_df):,}", size="small")
    with c3:
        clean_metric_card("Issuers", f"{market_df['issuer'].nunique() if 'issuer' in market_df.columns else 0:,}", size="small")
    with c4:
        clean_metric_card("CUSIPs", f"{market_df['cusip'].nunique() if 'cusip' in market_df.columns else 0:,}", size="small")
    with c5:
        clean_metric_card("Duplicates Removed", f"{duplicates_removed:,}", size="small")

    audit_cards, readiness = build_upload_audit_cards(
        trade_reports=trade_reports,
        market_df=market_df,
        benchmark_source_mode=benchmark_source_mode,
        use_external_mmd_fallback=use_external_mmd_fallback,
        mmd_file_provided=mmd_file_provided,
    )
    render_ready_to_analyze_card(readiness)

    st.subheader("Data Audit Status")
    _render_card_grid(audit_cards, "status-card-grid")

    audit_rows = []
    for report in trade_reports:
        audit_rows.append(
            {
                "File": report.get("dataset"),
                "Rows": report.get("row_count"),
                "Columns": report.get("column_count"),
                "Ready": "Yes" if report.get("can_run") else "No",
                "Missing Required": ", ".join(report.get("missing_required", [])) or "None",
                "Missing Recommended": ", ".join(report.get("missing_recommended", [])) or "None",
            }
        )
    if bond_report:
        audit_rows.append(
            {
                "File": "Optional bond reference",
                "Rows": bond_report.get("row_count", 0),
                "Columns": bond_report.get("column_count", 0),
                "Ready": "Yes" if bond_report.get("can_run", True) else "No",
                "Missing Required": ", ".join(bond_report.get("missing_required", [])) or "None",
                "Missing Recommended": ", ".join(bond_report.get("missing_recommended", [])) or "None",
            }
        )
    if mmd_report:
        audit_rows.append(
            {
                "File": "Optional MMD / benchmark curve",
                "Rows": mmd_report.get("row_count", 0),
                "Columns": mmd_report.get("column_count", 0),
                "Ready": "Yes" if mmd_report.get("can_run", True) else "No",
                "Missing Required": ", ".join(mmd_report.get("missing_required", [])) or "None",
                "Missing Recommended": ", ".join(mmd_report.get("missing_recommended", [])) or "None",
            }
        )
    with st.expander("Audit Details / Benchmark Methodology", expanded=readiness.get("status") == "bad"):
        if audit_rows:
            st.subheader("File Readiness Summary")
            safe_dataframe(pd.DataFrame(audit_rows), width="stretch", hide_index=True, auto_collapse=False)

        coverage_rows = [
            {"Metric": "Trade date coverage", "Value": _workflow_date_range_text(market_df)},
            {"Metric": "Security reference rows", "Value": f"{len(bonds_df):,}"},
            {"Metric": "Issuer master rows", "Value": f"{len(issuer_master):,}"},
            {"Metric": "Benchmark rows", "Value": f"{len(mmd_df):,}"},
            {"Metric": "Failed files", "Value": ", ".join(map(str, failed_files)) if failed_files else "None"},
        ]
        st.subheader("Data Coverage")
        safe_dataframe(pd.DataFrame(coverage_rows), width="stretch", hide_index=True, auto_collapse=False)

        st.subheader("Unified Methodology / Evidence")
        render_methodology_trust_panel(
            market_df=market_df,
            issuer_df=market_df,
            mmd_df=mmd_df,
            benchmark_source_mode=benchmark_source_mode,
            benchmark_priority=benchmark_priority,
            benchmark_conflict_policy=benchmark_conflict_policy,
            title="Audit Evidence / Benchmark Methodology",
            expanded=False,
            render_benchmark_methodology_block=render_benchmark_methodology_block,
        )


def display_validation_report(title: str, report: dict, warnings: list[str] | None = None):
    """Render a user-facing readiness card in Streamlit."""
    warnings = warnings or []
    status_icon = "[OK]" if report["can_run"] else "[FAIL]"
    with st.expander(f"{status_icon} {title} readiness check", expanded=not report["can_run"]):
        st.caption(f"Rows: {report['row_count']:,} | Columns: {report['column_count']:,}")

        c1, c2, c3 = st.columns(3)
        c1.metric("Required detected", f"{len(report['detected_required'])}/{len(report['detected_required']) + len(report['missing_required'])}")
        c2.metric("Recommended detected", f"{len(report['detected_recommended'])}/{len(report['detected_recommended']) + len(report['missing_recommended'])}")
        c3.metric("Ready to run", "Yes" if report["can_run"] else "No")

        if report["missing_required"]:
            st.error("Missing required fields: " + ", ".join(report["missing_required"]))
        if report["missing_recommended"]:
            st.warning("Missing recommended fields: " + ", ".join(report["missing_recommended"]))
        if warnings:
            for warning in warnings:
                st.warning(warning)

        mapping_rows = [
            {"Internal Field": key, "Uploaded Column Detected": value or "-"}
            for key, value in report["mapping"].items()
        ]
        safe_dataframe(pd.DataFrame(mapping_rows), width="stretch", hide_index=True)
