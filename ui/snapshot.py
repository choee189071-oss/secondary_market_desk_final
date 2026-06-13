from __future__ import annotations

import pandas as pd
import streamlit as st

from engine.scoring import (
    add_workflow_spread_bps as _add_workflow_spread_bps,
    build_workflow_cusip_summary as _build_workflow_cusip_summary,
    workflow_date_range_text as _workflow_date_range_text,
)
from ui.common import (
    _first_existing_col,
    _fmt_bps,
    _fmt_num,
    _numeric_rate,
    _nonnull_rate,
    _rate_status,
    _render_card_grid,
    clean_metric_card,
    safe_dataframe,
    section_anchor,
)


def _build_snapshot_methodology_cards(
    issuer_df: pd.DataFrame,
    mmd_df: pd.DataFrame,
    benchmark_source_mode: str,
) -> list[dict]:
    """Build compact methodology warnings for the desk snapshot."""
    index_rate = _numeric_rate(issuer_df, "index_rate")
    spread_rate = _numeric_rate(issuer_df, "spread")
    cusip_rate = _nonnull_rate(issuer_df, "cusip")
    rating_col = _first_existing_col(issuer_df, ["ratings_m_s_f", "rating", "ratings", "benchmark_rating"])
    rating_rate = _nonnull_rate(issuer_df, rating_col) if rating_col else None
    benchmark_rows = len(mmd_df) if isinstance(mmd_df, pd.DataFrame) else 0

    if benchmark_source_mode == "Uploaded MMD fallback":
        mmd_status = "good"
        mmd_value = "Uploaded MMD active"
        mmd_detail = f"External MMD is being used as the AAA benchmark curve with {benchmark_rows:,} benchmark row(s)."
    elif benchmark_source_mode == "Trade Sheet Index / Index Rate":
        mmd_status = "warn"
        mmd_value = "Trade index active"
        mmd_detail = f"No external MMD is active in this run; trade-sheet Index Rate is the benchmark source with {benchmark_rows:,} benchmark row(s)."
    else:
        mmd_status = "bad"
        mmd_value = "No active benchmark"
        mmd_detail = "No trade index or uploaded MMD benchmark is available."

    if (index_rate or 0) >= 70:
        index_status = "good"
        index_value = f"{index_rate:.1f}% numeric"
        index_detail = "Trade-sheet Index Rate is available for benchmark spread analytics."
    elif benchmark_source_mode == "Uploaded MMD fallback":
        index_status = "warn"
        index_value = "Index Rate weak / absent"
        index_detail = "Uploaded MMD fallback is active; verify MMD date/tenor coverage before relying on spread outputs."
    elif (spread_rate or 0) >= 70:
        index_status = "warn"
        index_value = f"Spread field {spread_rate:.1f}% numeric"
        index_detail = "Spread field exists, but Index Rate is preferred for transparent benchmark governance."
    else:
        index_status = "bad"
        index_value = "No usable Index Rate"
        index_detail = "Spread-to-benchmark and RV outputs may be degraded."

    if rating_rate is None:
        rating_status = "warn"
        rating_value = "Missing"
        rating_detail = "Ratings are unavailable; peer grouping should fall back to sector and maturity."
    elif rating_rate >= 80:
        rating_status = "good"
        rating_value = f"{rating_rate:.1f}% populated"
        rating_detail = f"Rating field detected: {rating_col}."
    elif rating_rate >= 30:
        rating_status = "warn"
        rating_value = f"{rating_rate:.1f}% populated"
        rating_detail = "Partial ratings coverage; attribution should disclose fallback logic."
    else:
        rating_status = "warn"
        rating_value = f"{rating_rate:.1f}% populated"
        rating_detail = "Ratings are sparse; use sector/maturity fallback for peer grouping."

    return [
        {
            "status": mmd_status,
            "kicker": "MMD / benchmark",
            "title": "AAA curve availability",
            "value": mmd_value,
            "detail": mmd_detail,
        },
        {
            "status": index_status,
            "kicker": "Index Rate",
            "title": "Benchmark input",
            "value": index_value,
            "detail": index_detail,
        },
        {
            "status": rating_status,
            "kicker": "Ratings",
            "title": "Peer grouping input",
            "value": rating_value,
            "detail": rating_detail,
        },
        {
            "status": _rate_status(cusip_rate, good_threshold=95, warn_threshold=80),
            "kicker": "CUSIP quality",
            "title": "CUSIP-level reliability",
            "value": "N/A" if cusip_rate is None else f"{cusip_rate:.1f}% valid",
            "detail": "Low CUSIP quality weakens drilldown, watchlist, and opportunity ranking.",
        },
    ]


def _build_snapshot_takeaway(
    issuer_df: pd.DataFrame,
    market_df: pd.DataFrame,
    cusip_summary: pd.DataFrame,
    selected_issuer: str,
    benchmark_source_mode: str,
) -> tuple[list[str], dict]:
    """Return deterministic analyst takeaway bullets and supporting labels."""
    issuer_base = _add_workflow_spread_bps(issuer_df)
    universe_base = _add_workflow_spread_bps(market_df)
    issuer_spreads = pd.to_numeric(issuer_base.get("spread_bps"), errors="coerce").dropna()
    universe_spreads = pd.to_numeric(universe_base.get("spread_bps"), errors="coerce").dropna()

    spread_label = "Spread unavailable"
    spread_detail = "No usable spread/index-rate data was found."
    if not issuer_spreads.empty and not universe_spreads.empty:
        issuer_median = float(issuer_spreads.median())
        universe_median = float(universe_spreads.median())
        p25 = float(universe_spreads.quantile(0.25))
        p75 = float(universe_spreads.quantile(0.75))
        if issuer_median >= p75:
            spread_label = "Wide / cheaper"
            spread_detail = f"{selected_issuer} median spread is {issuer_median:.1f} bps vs universe median {universe_median:.1f} bps."
        elif issuer_median <= p25:
            spread_label = "Tight / richer"
            spread_detail = f"{selected_issuer} median spread is {issuer_median:.1f} bps vs universe median {universe_median:.1f} bps."
        else:
            spread_label = "Near uploaded universe"
            spread_detail = f"{selected_issuer} median spread is {issuer_median:.1f} bps vs universe median {universe_median:.1f} bps."

    liquidity_label = "Liquidity unavailable"
    liquidity_detail = "CUSIP-level liquidity could not be scored."
    if not cusip_summary.empty and "liquidity_score" in cusip_summary.columns:
        liquidity_scores = pd.to_numeric(cusip_summary["liquidity_score"], errors="coerce").dropna()
        if not liquidity_scores.empty:
            median_liq = float(liquidity_scores.median())
            top_liq = float(liquidity_scores.max())
            if median_liq >= 70:
                liquidity_label = "Liquidity strong"
            elif median_liq >= 45:
                liquidity_label = "Liquidity mixed"
            else:
                liquidity_label = "Liquidity thin"
            liquidity_detail = f"Median CUSIP liquidity score is {median_liq:.1f}; top score is {top_liq:.1f}."

    top_label = "No top CUSIP"
    top_detail = "No CUSIP summary is available."
    if not cusip_summary.empty:
        top = cusip_summary.iloc[0]
        top_label = str(top.get("cusip", "N/A"))
        top_detail = (
            f"{top.get('signal', 'Monitor')} with RV score {_fmt_num(top.get('rv_score'))} "
            f"and liquidity score {_fmt_num(top.get('liquidity_score'))}."
        )

    if benchmark_source_mode == "Trade Sheet Index / Index Rate":
        benchmark_label = "Benchmark OK"
        benchmark_detail = "Using trade-sheet Index / Index Rate; external MMD is not mixed into this run."
    elif benchmark_source_mode == "Uploaded MMD fallback":
        benchmark_label = "MMD fallback active"
        benchmark_detail = "Uploaded MMD is being used as the AAA benchmark curve."
    else:
        benchmark_label = "Benchmark warning"
        benchmark_detail = "No active benchmark source; spread/RV conclusions should be treated as incomplete."

    bullets = [
        f"Spread posture: {spread_label}. {spread_detail}",
        f"Liquidity posture: {liquidity_label}. {liquidity_detail}",
        f"Top CUSIP: {top_label}. {top_detail}",
        f"Benchmark: {benchmark_label}. {benchmark_detail}",
    ]
    labels = {
        "spread_label": spread_label,
        "liquidity_label": liquidity_label,
        "top_label": top_label,
        "benchmark_label": benchmark_label,
    }
    return bullets, labels


def render_focused_snapshot(
    market_df: pd.DataFrame,
    bonds_df: pd.DataFrame,
    issuer_trades: pd.DataFrame,
    issuer_bonds: pd.DataFrame,
    mmd_df: pd.DataFrame,
    selected_issuer: str,
    selected_sector: str,
    benchmark_source_mode: str,
):
    section_anchor("workflow-desk-snapshot", "Desk Snapshot")
    st.markdown(
        "<div class='focus-band'>Decision-first view. Read this before opening detailed charts: coverage, current spread, liquidity, and the strongest CUSIP candidates.</div>",
        unsafe_allow_html=True,
    )
    issuer_base = _add_workflow_spread_bps(issuer_trades)
    cusip_summary = _build_workflow_cusip_summary(issuer_base)
    date_range = _workflow_date_range_text(issuer_base)
    spread_series = pd.to_numeric(issuer_base.get("spread_bps"), errors="coerce") if "spread_bps" in issuer_base.columns else pd.Series(dtype="float64")
    top_candidate = cusip_summary.iloc[0] if not cusip_summary.empty else None
    cusip_count = issuer_base["cusip"].nunique() if "cusip" in issuer_base.columns else 0
    top_liquidity = None if top_candidate is None else top_candidate.get("liquidity_score")
    top_candidate_text = "N/A" if top_candidate is None else str(top_candidate.get("cusip", "N/A"))
    top_candidate_note = None
    if top_candidate is not None:
        top_candidate_note = f"{top_candidate.get('signal', 'Monitor')} | liquidity {_fmt_num(top_liquidity)}"

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        clean_metric_card("Issuer", selected_issuer, size="small", note=selected_sector)
    with c2:
        clean_metric_card("Date Range", date_range, size="small")
    with c3:
        clean_metric_card("Trade Rows", f"{len(issuer_trades):,}", size="small")
    with c4:
        clean_metric_card("CUSIPs", f"{cusip_count:,}", size="small")
    with c5:
        clean_metric_card("Median Spread", "N/A" if spread_series.dropna().empty else f"{spread_series.median():.1f} bps", size="small")
    with c6:
        clean_metric_card("Liquidity / Top", top_candidate_text, size="small", note=top_candidate_note)

    st.subheader("Analyst Takeaway")
    bullets, takeaway_labels = _build_snapshot_takeaway(
        issuer_df=issuer_base,
        market_df=market_df,
        cusip_summary=cusip_summary,
        selected_issuer=selected_issuer,
        benchmark_source_mode=benchmark_source_mode,
    )
    for bullet in bullets:
        st.markdown(f"- {bullet}")

    if not cusip_summary.empty:
        st.subheader("Top 5 Opportunities")
        opps = cusip_summary.head(5).copy()
        if "current_spread_bps" in opps.columns:
            opps["current_spread_bps"] = pd.to_numeric(opps["current_spread_bps"], errors="coerce").round(1)
        for col in ["liquidity_score", "rv_score"]:
            if col in opps.columns:
                opps[col] = pd.to_numeric(opps[col], errors="coerce").round(1)
        if "total_trade_amount" in opps.columns:
            opps["total_trade_amount"] = pd.to_numeric(opps["total_trade_amount"], errors="coerce").round(0)
        opps["snapshot_reason"] = opps.apply(
            lambda row: (
                f"{row.get('signal', 'Monitor')}; spread {_fmt_bps(row.get('current_spread_bps'))}; "
                f"liquidity {_fmt_num(row.get('liquidity_score'))}; RV {_fmt_num(row.get('rv_score'))}"
            ),
            axis=1,
        )
        display_cols = [
            "cusip", "signal", "maturity_bucket", "current_spread_bps", "liquidity_score",
            "rv_score", "trade_count", "total_trade_amount", "latest_trade", "snapshot_reason",
        ]
        safe_dataframe(opps[[c for c in display_cols if c in opps.columns]], hide_index=True, auto_collapse=False)
    else:
        st.info("No CUSIP-level opportunity table is available for the current filter.")

    st.subheader("Methodology Warnings")
    warning_cards = _build_snapshot_methodology_cards(
        issuer_df=issuer_base,
        mmd_df=mmd_df,
        benchmark_source_mode=benchmark_source_mode,
    )
    _render_card_grid(warning_cards, "status-card-grid")

    with st.expander("Snapshot calculation notes", expanded=False):
        st.markdown(
            f"""
- **Spread posture** compares the selected issuer median spread against the uploaded universe distribution.
- **Liquidity posture** uses the CUSIP-level liquidity score from trade count, par amount, and recency.
- **Top opportunity ranking** uses the focused workflow RV score, which combines spread rank and liquidity rank.
- **Benchmark source shown:** `{benchmark_source_mode}`.
- **Current snapshot labels:** spread = `{takeaway_labels.get('spread_label')}`, liquidity = `{takeaway_labels.get('liquidity_label')}`, benchmark = `{takeaway_labels.get('benchmark_label')}`.
            """
        )
