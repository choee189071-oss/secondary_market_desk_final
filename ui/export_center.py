from __future__ import annotations

import streamlit as st

from reports.export_center import (
    focused_report_bundle_bytes,
    focused_report_filename,
    focused_report_html,
    focused_report_markdown,
    focused_report_pdf_bytes,
    focused_report_pptx_bytes,
)


def _select_existing(df, columns: list[str]):
    return df[[c for c in columns if c in df.columns]]


def render_focused_export_methodology(
    selected_issuer: str,
    selected_sector: str,
    market_df,
    issuer_trades,
    issuer_bonds,
    mmd_df,
    benchmark_source_mode: str,
    benchmark_priority: str,
    benchmark_conflict_policy: str,
    *,
    build_report_context,
    section_anchor,
    clean_metric_card,
    safe_dataframe,
    render_benchmark_methodology_block,
):
    section_anchor("workflow-export-methodology", "Export / Methodology")
    st.markdown(
        "<div class='focus-band'>Final reporting center. Package the desk snapshot, saved watchlist, chart guide, and benchmark methodology into shareable files.</div>",
        unsafe_allow_html=True,
    )

    default_title = f"{selected_issuer} Secondary Market Desk Report"
    ctrl1, ctrl2 = st.columns([1.35, 1])
    with ctrl1:
        report_title = st.text_input("Report title", value=default_title, key="focused_report_title")
    with ctrl2:
        prepared_for = st.text_input("Prepared for", value="Internal desk review", key="focused_report_prepared_for")
    analyst_note = st.text_area(
        "Analyst note",
        key="focused_report_analyst_note",
        height=86,
        placeholder="Optional framing note, follow-up question, or desk instruction to carry into the report.",
    )
    include_col1, include_col2, include_col3 = st.columns(3)
    with include_col1:
        include_watchlist = st.checkbox("Include saved watchlist", value=True, key="focused_report_include_watchlist")
    with include_col2:
        include_methodology = st.checkbox("Include methodology appendix", value=True, key="focused_report_include_methodology")
    with include_col3:
        include_optional_formats = st.checkbox("Show PDF / PPTX downloads", value=True, key="focused_report_include_optional_formats")

    context = build_report_context(
        report_title=report_title,
        prepared_for=prepared_for,
        analyst_note=analyst_note,
        selected_issuer=selected_issuer,
        selected_sector=selected_sector,
        market_df=market_df,
        issuer_trades=issuer_trades,
        issuer_bonds=issuer_bonds,
        mmd_df=mmd_df,
        benchmark_source_mode=benchmark_source_mode,
        benchmark_priority=benchmark_priority,
        benchmark_conflict_policy=benchmark_conflict_policy,
    )
    report_md = focused_report_markdown(context, include_watchlist=include_watchlist, include_methodology=include_methodology)
    report_html = focused_report_html(context, include_watchlist=include_watchlist, include_methodology=include_methodology)
    bundle_bytes = focused_report_bundle_bytes(context, report_md, report_html)

    st.subheader("Report Snapshot")
    metric_lookup = {row["Metric"]: row["Value"] for _, row in context["metrics"].iterrows()}
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1:
        clean_metric_card("Issuer", metric_lookup.get("Issuer"), size="small", note=metric_lookup.get("Sector"))
    with m2:
        clean_metric_card("Date Range", metric_lookup.get("Trade date range"), size="small")
    with m3:
        clean_metric_card("Trade Rows", metric_lookup.get("Trade rows"), size="small")
    with m4:
        clean_metric_card("Median Spread", metric_lookup.get("Median spread"), size="small")
    with m5:
        clean_metric_card("Saved", metric_lookup.get("Saved watchlist"), size="small")
    with m6:
        clean_metric_card("Benchmark", metric_lookup.get("Benchmark source"), size="small")

    st.subheader("Analyst Takeaway Preview")
    for bullet in context["takeaway_bullets"]:
        st.markdown(f"- {bullet}")
    if context.get("top_candidate_note"):
        st.caption(f"Top candidate read-through: {context['top_candidate_note']}")

    st.subheader("Top Opportunities Included")
    top_cols = [
        "cusip", "signal", "maturity_bucket", "current_spread_bps", "peer_median_gap_bps",
        "liquidity_score", "rv_score", "trade_count", "total_trade_amount", "latest_trade",
    ]
    if context["top_opportunities"].empty:
        st.info("No CUSIP opportunity rows are available for this report.")
    else:
        safe_dataframe(_select_existing(context["top_opportunities"], top_cols), hide_index=True, auto_collapse=False)

    st.subheader("Saved Watchlist Included")
    watch_cols = [
        "cusip", "issuer", "signal", "maturity_bucket", "current_spread_bps", "peer_median_gap_bps",
        "liquidity_score", "rv_score", "trade_count", "total_trade_amount", "latest_trade",
        "note", "source", "updated_at",
    ]
    if context["saved_watchlist"].empty:
        st.info("No saved watchlist candidates yet. Save CUSIPs from CUSIP Drilldown or RV / Watchlist before final export.")
    else:
        safe_dataframe(_select_existing(context["saved_watchlist"], watch_cols), hide_index=True, auto_collapse=False)

    st.subheader("Core Chart Guide Included")
    safe_dataframe(context["chart_explanations"], hide_index=True, auto_collapse=False)

    st.subheader("Downloads")
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.download_button(
            "Download Report Markdown",
            data=report_md.encode("utf-8"),
            file_name=focused_report_filename(selected_issuer, "desk_report.md"),
            mime="text/markdown",
        )
    with d2:
        st.download_button(
            "Download Print HTML",
            data=report_html.encode("utf-8"),
            file_name=focused_report_filename(selected_issuer, "desk_report.html"),
            mime="text/html",
            help="Open in a browser and use Print to save a visual PDF if needed.",
        )
    with d3:
        st.download_button(
            "Download Report Bundle",
            data=bundle_bytes,
            file_name=focused_report_filename(selected_issuer, "report_bundle.zip"),
            mime="application/zip",
        )
    with d4:
        st.download_button(
            "Download Watchlist CSV",
            data=context["saved_watchlist"].to_csv(index=False).encode("utf-8"),
            file_name=focused_report_filename(selected_issuer, "saved_watchlist.csv"),
            mime="text/csv",
            disabled=context["saved_watchlist"].empty,
        )

    if include_optional_formats:
        opt1, opt2 = st.columns(2)
        with opt1:
            pdf_bytes, pdf_error = focused_report_pdf_bytes(context)
            if pdf_bytes:
                st.download_button(
                    "Download PDF Summary",
                    data=pdf_bytes,
                    file_name=focused_report_filename(selected_issuer, "desk_summary.pdf"),
                    mime="application/pdf",
                )
            else:
                st.info(f"PDF export requires `reportlab`. Current error: {pdf_error}")
        with opt2:
            pptx_bytes, pptx_error = focused_report_pptx_bytes(context)
            if pptx_bytes:
                st.download_button(
                    "Download PPTX Outline",
                    data=pptx_bytes,
                    file_name=focused_report_filename(selected_issuer, "desk_report_outline.pptx"),
                    mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                )
            else:
                st.info(f"PPTX export requires `python-pptx`. Current error: {pptx_error}")

    with st.expander("Preview Markdown Report", expanded=False):
        st.markdown(report_md)

    st.subheader("Methodology / Benchmark Audit")
    render_benchmark_methodology_block(mmd_df, benchmark_source_mode, benchmark_priority, benchmark_conflict_policy)
    st.subheader("Methodology Appendix for Report")
    safe_dataframe(context["methodology"], hide_index=True, auto_collapse=False)
