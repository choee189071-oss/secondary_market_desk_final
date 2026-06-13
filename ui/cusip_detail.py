from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from engine.scoring import (
    add_workflow_spread_bps as _add_workflow_spread_bps,
    build_workflow_cusip_summary as _build_workflow_cusip_summary,
    focused_trade_side as _focused_trade_side,
)
from ui.common import (
    _first_existing_col,
    _fmt_bps,
    _fmt_date,
    _fmt_mm,
    _fmt_num,
    _fmt_pct,
    clean_metric_card,
    safe_dataframe,
    safe_plotly_chart,
    section_anchor,
)


def _focused_watchlist_records() -> dict:
    """Return mutable watchlist records, migrating older list-based session state."""
    if "focused_watchlist_records" not in st.session_state:
        records = {}
        for cusip in st.session_state.get("focused_watchlist", []):
            records[str(cusip)] = {
                "cusip": str(cusip),
                "issuer": "",
                "signal": "",
                "note": "",
                "source": "Migrated",
                "added_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
            }
        st.session_state["focused_watchlist_records"] = records
    return st.session_state["focused_watchlist_records"]


def _upsert_focused_watchlist(cusip: object, issuer: str, source: str, row: pd.Series | dict | None = None, note: str = ""):
    records = _focused_watchlist_records()
    key = str(cusip)
    existing = records.get(key, {})
    row_dict = row.to_dict() if isinstance(row, pd.Series) else (row or {})
    records[key] = {
        "cusip": key,
        "issuer": issuer or existing.get("issuer", ""),
        "signal": row_dict.get("signal", existing.get("signal", "")),
        "maturity_bucket": row_dict.get("maturity_bucket", existing.get("maturity_bucket", "")),
        "current_spread_bps": row_dict.get("current_spread_bps", existing.get("current_spread_bps", pd.NA)),
        "peer_median_gap_bps": row_dict.get("peer_median_gap_bps", existing.get("peer_median_gap_bps", pd.NA)),
        "liquidity_score": row_dict.get("liquidity_score", existing.get("liquidity_score", pd.NA)),
        "rv_score": row_dict.get("rv_score", existing.get("rv_score", pd.NA)),
        "trade_count": row_dict.get("trade_count", existing.get("trade_count", pd.NA)),
        "total_trade_amount": row_dict.get("total_trade_amount", existing.get("total_trade_amount", pd.NA)),
        "latest_trade": row_dict.get("latest_trade", existing.get("latest_trade", pd.NA)),
        "note": note if note else existing.get("note", ""),
        "source": source or existing.get("source", ""),
        "added_at": existing.get("added_at") or pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "updated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
    }
    st.session_state["focused_watchlist"] = sorted(records.keys())


def _focused_watchlist_dataframe(summary: pd.DataFrame | None = None) -> pd.DataFrame:
    records = _focused_watchlist_records()
    if not records:
        return pd.DataFrame()
    out = pd.DataFrame(records.values())
    if summary is not None and not summary.empty and "cusip" in summary.columns:
        refresh_cols = [
            "cusip", "signal", "maturity_bucket", "current_spread_bps", "peer_median_gap_bps", "liquidity_score",
            "rv_score", "trade_count", "total_trade_amount", "latest_trade",
        ]
        current = summary[[c for c in refresh_cols if c in summary.columns]].copy()
        current["cusip"] = current["cusip"].astype(str)
        out = out.merge(current, on="cusip", how="left", suffixes=("", "_current"))
        for col in [c for c in current.columns if c != "cusip"]:
            current_col = f"{col}_current"
            if current_col not in out.columns:
                continue
            if col in out.columns:
                out[col] = out[current_col].combine_first(out[col])
            else:
                out[col] = out[current_col]
            out = out.drop(columns=[current_col])
    return out


def _focused_watchlist_markdown(saved_df: pd.DataFrame, issuer: str) -> str:
    if saved_df.empty:
        return f"# {issuer} Watchlist\n\nNo saved candidates."
    lines = [
        f"# {issuer} Watchlist",
        "",
        f"Generated: {pd.Timestamp.now():%Y-%m-%d %H:%M}",
        "",
    ]
    for _, row in saved_df.iterrows():
        lines.extend(
            [
                f"## {row.get('cusip', 'N/A')}",
                f"- Signal: {row.get('signal', 'N/A')}",
                f"- Maturity bucket: {row.get('maturity_bucket', 'N/A')}",
                f"- Spread: {_fmt_bps(row.get('current_spread_bps'))}",
                f"- Peer median gap: {_fmt_bps(row.get('peer_median_gap_bps'))}",
                f"- Liquidity score: {_fmt_num(row.get('liquidity_score'))}",
                f"- RV score: {_fmt_num(row.get('rv_score'))}",
                f"- Note: {row.get('note', '') or 'N/A'}",
                "",
            ]
        )
    return "\n".join(lines)


def render_focused_cusip_drilldown(issuer_trades: pd.DataFrame, selected_issuer: str):
    section_anchor("workflow-cusip-drilldown", "CUSIP Drilldown")
    st.markdown(
        "<div class='focus-band'><b>CUSIP evidence:</b> metrics, path, peers.</div>",
        unsafe_allow_html=True,
    )
    summary = _build_workflow_cusip_summary(issuer_trades)
    if summary.empty:
        st.info("No CUSIP-level rows are available for the selected issuer/filter.")
        return

    selector_options = summary["cusip"].dropna().astype(str).tolist()
    selected_cusip = st.selectbox("Select CUSIP", selector_options)
    selected_row = summary[summary["cusip"].astype(str) == str(selected_cusip)].iloc[0]
    detail = _add_workflow_spread_bps(issuer_trades[issuer_trades["cusip"].astype(str) == str(selected_cusip)].copy())
    detail["trade_date"] = pd.to_datetime(detail.get("trade_date"), errors="coerce")
    for col in ["yield", "price", "trade_amount", "spread_bps"]:
        if col in detail.columns:
            detail[col] = pd.to_numeric(detail[col], errors="coerce")
    if "trade_amount" not in detail.columns:
        detail["trade_amount"] = 0.0
    if "yield" not in detail.columns:
        detail["yield"] = pd.NA
    if "price" not in detail.columns:
        detail["price"] = pd.NA

    detail_sorted = detail.sort_values("trade_date").copy()
    latest_trade_row = detail_sorted.dropna(subset=["trade_date"]).tail(1)
    latest_date = latest_trade_row["trade_date"].iloc[0] if not latest_trade_row.empty else pd.NaT
    latest_yield = latest_trade_row["yield"].iloc[0] if not latest_trade_row.empty and "yield" in latest_trade_row.columns else pd.NA
    latest_price = latest_trade_row["price"].iloc[0] if not latest_trade_row.empty and "price" in latest_trade_row.columns else pd.NA
    total_par = pd.to_numeric(detail_sorted["trade_amount"], errors="coerce").sum()
    avg_trade_size = pd.to_numeric(detail_sorted["trade_amount"], errors="coerce").mean()

    path = (
        detail_sorted.dropna(subset=["trade_date"])
        .groupby("trade_date", as_index=False)
        .agg(
            spread_bps=("spread_bps", "median"),
            avg_yield=("yield", "mean"),
            avg_price=("price", "mean"),
            par=("trade_amount", "sum"),
            trade_count=("trade_amount", "count"),
        )
        .sort_values("trade_date")
    )
    spread_change = pd.NA
    if not path.empty and pd.to_numeric(path["spread_bps"], errors="coerce").notna().sum() >= 2:
        clean_path_spread = path.dropna(subset=["spread_bps"])
        spread_change = float(clean_path_spread["spread_bps"].iloc[-1] - clean_path_spread["spread_bps"].iloc[0])

    side_col = _first_existing_col(detail, ["trade_type", "side", "buy_sell", "customer_side", "dealer_side"])
    if side_col:
        detail["flow_side"] = detail[side_col].map(_focused_trade_side)
    else:
        detail["flow_side"] = "Unknown"
    buy_count = int((detail["flow_side"] == "Buy").sum())
    sell_count = int((detail["flow_side"] == "Sell").sum())
    other_count = int((~detail["flow_side"].isin(["Buy", "Sell"])).sum())

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        clean_metric_card("CUSIP", selected_cusip, size="small")
    with c2:
        clean_metric_card("Signal", selected_row.get("signal"), size="small")
    with c3:
        clean_metric_card("Spread", _fmt_bps(selected_row.get("current_spread_bps")), size="small")
    with c4:
        clean_metric_card("Liquidity", _fmt_num(selected_row.get("liquidity_score")), size="small")
    with c5:
        clean_metric_card("Trades", f"{int(selected_row.get('trade_count', 0)):,}", size="small")

    with st.expander("More CUSIP metrics", expanded=False):
        d1, d2, d3, d4, d5 = st.columns(5)
        with d1:
            clean_metric_card("Latest Trade", _fmt_date(latest_date) if pd.notna(latest_date) else "N/A", size="small")
        with d2:
            clean_metric_card("Latest Yield", _fmt_pct(latest_yield), size="small")
        with d3:
            clean_metric_card("Latest Price", _fmt_num(latest_price), size="small")
        with d4:
            clean_metric_card("Total Par", _fmt_mm(total_par), size="small")
        with d5:
            clean_metric_card("Path Change", _fmt_bps(spread_change), size="small")

    st.subheader("Analyst Read-Through")
    readthrough = [
        f"{len(detail):,} trades in current filter.",
        f"Spread {_fmt_bps(selected_row.get('current_spread_bps'))}; liquidity {_fmt_num(selected_row.get('liquidity_score'))}.",
        f"Flow: {buy_count:,} buy / {sell_count:,} sell / {other_count:,} other.",
    ]
    if pd.notna(spread_change):
        readthrough.append(f"Path change: {_fmt_bps(spread_change)}.")
    bucket = selected_row.get("maturity_bucket") if "maturity_bucket" in summary.columns else None
    if pd.notna(bucket):
        readthrough.append(f"Peer bucket: {bucket}.")
    for line in readthrough:
        st.markdown(f"- {line}")

    records = _focused_watchlist_records()
    existing_note = records.get(str(selected_cusip), {}).get("note", "")
    note_col, action_col = st.columns([2.2, 1])
    with note_col:
        watch_note = st.text_area(
            "Watchlist note",
            value=existing_note,
            key=f"cusip_watch_note_{selected_cusip}",
            height=86,
            placeholder="Why this CUSIP is worth saving, what to verify, or how to frame it in the report.",
        )
    with action_col:
        st.caption("Save for watchlist/export.")
        if st.button("Save / Update Watchlist", key=f"save_watch_{selected_cusip}"):
            _upsert_focused_watchlist(selected_cusip, selected_issuer, "CUSIP Drilldown", selected_row, watch_note)
            st.success(f"Saved {selected_cusip} to watchlist.")

    st.subheader("Trade Path")
    if not path.empty:
        fig_path = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.42, 0.30, 0.28],
            subplot_titles=("Spread Path", "Yield / Price", "Par Amount"),
            specs=[[{}], [{"secondary_y": True}], [{}]],
        )
        if pd.to_numeric(path["spread_bps"], errors="coerce").notna().any():
            fig_path.add_trace(
                go.Scatter(
                    x=path["trade_date"],
                    y=path["spread_bps"],
                    mode="lines+markers",
                    name="Spread",
                    line=dict(width=3),
                    customdata=np.stack([path["trade_count"].fillna(0), path["par"].fillna(0)], axis=-1),
                    hovertemplate="%{x|%m/%d/%Y}<br>Spread: %{y:.1f} bps<br>Trades: %{customdata[0]:,.0f}<br>Par: $%{customdata[1]:,.0f}<extra>Spread</extra>",
                ),
                row=1,
                col=1,
            )
        if pd.to_numeric(path["avg_yield"], errors="coerce").notna().any():
            fig_path.add_trace(
                go.Scatter(
                    x=path["trade_date"],
                    y=path["avg_yield"],
                    mode="lines+markers",
                    name="Yield",
                    line=dict(width=2.4),
                    hovertemplate="%{x|%m/%d/%Y}<br>Yield: %{y:.3f}%<extra>Yield</extra>",
                ),
                row=2,
                col=1,
                secondary_y=False,
            )
        if pd.to_numeric(path["avg_price"], errors="coerce").notna().any():
            fig_path.add_trace(
                go.Scatter(
                    x=path["trade_date"],
                    y=path["avg_price"],
                    mode="lines+markers",
                    name="Price",
                    line=dict(width=2.0, dash="dash"),
                    hovertemplate="%{x|%m/%d/%Y}<br>Price: %{y:.2f}<extra>Price</extra>",
                ),
                row=2,
                col=1,
                secondary_y=True,
            )
        fig_path.add_trace(
            go.Bar(
                x=path["trade_date"],
                y=path["par"],
                name="Par amount",
                hovertemplate="%{x|%m/%d/%Y}<br>Par: $%{y:,.0f}<extra>Par amount</extra>",
            ),
            row=3,
            col=1,
        )
        fig_path.update_layout(
            title=f"{selected_cusip} Trade Path",
            height=720,
            hovermode="x unified",
            legend_title_text="Series",
            margin=dict(l=40, r=50, t=85, b=45),
        )
        fig_path.update_yaxes(title_text="Spread (bps)", row=1, col=1)
        fig_path.update_yaxes(title_text="Yield (%)", row=2, col=1, secondary_y=False)
        fig_path.update_yaxes(title_text="Price", row=2, col=1, secondary_y=True)
        fig_path.update_yaxes(title_text="Par", row=3, col=1)
        safe_plotly_chart(fig_path, width="stretch")
        with st.expander("Trade path data", expanded=False):
            safe_dataframe(path, hide_index=True, auto_collapse=False)
    else:
        st.info("No dated trade path is available for the selected CUSIP.")

    with st.expander("Raw trade detail", expanded=False):
        display_cols = ["trade_date", "trade_type", "yield", "price", "trade_amount", "spread_bps", "maturity_bucket", "description"]
        safe_dataframe(detail[[c for c in display_cols if c in detail.columns]].sort_values("trade_date", ascending=False), hide_index=True)

    if pd.notna(bucket):
        st.subheader("Same-Bucket Peers")
        peers = summary[summary["maturity_bucket"].astype(str) == str(bucket)].copy()
        peers["is_selected"] = peers["cusip"].astype(str).eq(str(selected_cusip))
        peer_median_spread = pd.to_numeric(peers["current_spread_bps"], errors="coerce").median()
        peers["peer_median_gap_bps"] = pd.to_numeric(peers["current_spread_bps"], errors="coerce") - peer_median_spread
        peers = peers.sort_values(["rv_score", "liquidity_score", "trade_count"], ascending=False)
        if not peers.empty and pd.notna(peer_median_spread):
            selected_gap = peers.loc[peers["is_selected"], "peer_median_gap_bps"]
            selected_gap_val = selected_gap.iloc[0] if not selected_gap.empty else pd.NA
            st.info(
                f"Same-bucket median spread is {_fmt_bps(peer_median_spread)}. "
                f"{selected_cusip} screens {_fmt_bps(selected_gap_val)} versus that peer median."
            )
        peer_cols = [
            "cusip", "is_selected", "signal", "current_spread_bps", "peer_median_gap_bps",
            "liquidity_score", "rv_score", "trade_count", "total_trade_amount", "latest_trade",
        ]
        with st.expander("Same-bucket peer table", expanded=False):
            safe_dataframe(peers[[c for c in peer_cols if c in peers.columns]].head(20), hide_index=True)


def render_focused_rv_watchlist(issuer_trades: pd.DataFrame, selected_issuer: str):
    section_anchor("workflow-rv-watchlist", "RV / Watchlist")
    st.markdown(
        "<div class='focus-band'><b>Rank:</b> filter, save, export.</div>",
        unsafe_allow_html=True,
    )
    summary = _build_workflow_cusip_summary(issuer_trades)
    if summary.empty:
        st.info("No CUSIP-level rows are available for RV ranking.")
        return

    summary = summary.copy()
    if "maturity_bucket" in summary.columns:
        summary["peer_median_spread_bps"] = summary.groupby("maturity_bucket")["current_spread_bps"].transform("median")
        summary["peer_median_gap_bps"] = pd.to_numeric(summary["current_spread_bps"], errors="coerce") - pd.to_numeric(summary["peer_median_spread_bps"], errors="coerce")
    else:
        summary["peer_median_gap_bps"] = pd.NA

    filt1, filt2, filt3, filt4 = st.columns([1, 1, 1.2, 1.2])
    with filt1:
        min_liq = st.slider("Minimum liquidity score", 0, 100, 40)
    with filt2:
        min_trades = st.number_input("Minimum trade count", min_value=1, max_value=1000, value=2, step=1)
    with filt3:
        signal_options = sorted(summary["signal"].dropna().astype(str).unique().tolist()) if "signal" in summary.columns else []
        selected_signals = st.multiselect(
            "Signals",
            signal_options,
            default=signal_options,
            key="focused_rv_signal_filter",
        )
    with filt4:
        bucket_options = sorted(summary["maturity_bucket"].dropna().astype(str).unique().tolist()) if "maturity_bucket" in summary.columns else []
        selected_buckets = st.multiselect(
            "Maturity buckets",
            bucket_options,
            default=bucket_options,
            key="focused_rv_bucket_filter",
        )

    ranked = summary[
        (pd.to_numeric(summary["liquidity_score"], errors="coerce") >= min_liq)
        & (pd.to_numeric(summary["trade_count"], errors="coerce") >= min_trades)
    ].copy()
    if signal_options and "signal" in ranked.columns:
        ranked = ranked[ranked["signal"].astype(str).isin(selected_signals)].copy()
    if bucket_options and "maturity_bucket" in ranked.columns:
        ranked = ranked[ranked["maturity_bucket"].astype(str).isin(selected_buckets)].copy()
    ranked = ranked.sort_values(["rv_score", "liquidity_score", "trade_count"], ascending=False)

    display_cols = [
        "cusip", "signal", "maturity_bucket", "current_spread_bps", "peer_median_gap_bps",
        "liquidity_score", "rv_score", "trade_count", "total_trade_amount", "latest_trade",
    ]
    st.subheader("Opportunity Ranking")
    if ranked.empty:
        st.info("No candidates meet the current RV/watchlist filters.")
    else:
        r1, r2, r3, r4 = st.columns(4)
        with r1:
            clean_metric_card("Candidates", f"{len(ranked):,}", size="small")
        with r2:
            clean_metric_card("Top RV", _fmt_num(ranked["rv_score"].max()), size="small")
        with r3:
            clean_metric_card("Top Liquidity", _fmt_num(ranked["liquidity_score"].max()), size="small")
        with r4:
            clean_metric_card("Median Peer Gap", _fmt_bps(ranked["peer_median_gap_bps"].median()), size="small")
        st.caption("Top 10. Expand for more.")
        safe_dataframe(ranked[[c for c in display_cols if c in ranked.columns]].head(10), hide_index=True, auto_collapse=False)
        with st.expander("Full opportunity ranking preview", expanded=False):
            safe_dataframe(ranked[[c for c in display_cols if c in ranked.columns]].head(50), hide_index=True)

    st.subheader("Watchlist")
    _focused_watchlist_records()
    add_options = ranked["cusip"].dropna().astype(str).head(150).tolist() if not ranked.empty else []
    add_col, note_col = st.columns([1.2, 2])
    with add_col:
        selected_add = st.multiselect("Add CUSIPs", add_options, key="focused_rv_add_cusips")
    with note_col:
        bulk_note = st.text_input(
            "Note for selected CUSIPs",
            key="focused_rv_bulk_note",
            placeholder="Why these belong on the shortlist, or what to verify next.",
        )
    add_button_col, clear_button_col = st.columns([1, 1])
    with add_button_col:
        if st.button("Add selected to watchlist", key="focused_rv_add_selected"):
            for item in selected_add:
                row_match = ranked[ranked["cusip"].astype(str) == str(item)]
                row = row_match.iloc[0] if not row_match.empty else {"cusip": item}
                _upsert_focused_watchlist(item, selected_issuer, "RV Ranking", row, bulk_note)
            st.success(f"Saved {len(selected_add):,} selected CUSIP(s).")
    with clear_button_col:
        if st.button("Clear full watchlist", key="focused_rv_clear_watchlist"):
            st.session_state["focused_watchlist_records"] = {}
            st.session_state["focused_watchlist"] = []
            st.info("Watchlist cleared.")

    saved = _focused_watchlist_dataframe(summary)
    if saved.empty:
        st.info("No saved CUSIPs yet.")
    else:
        st.caption(f"{len(saved):,} saved.")
        saved_display_cols = [
            "cusip", "issuer", "signal", "maturity_bucket", "current_spread_bps", "peer_median_gap_bps",
            "liquidity_score", "rv_score", "trade_count", "total_trade_amount", "latest_trade",
            "note", "source", "updated_at",
        ]
        with st.expander("Saved candidate table", expanded=False):
            safe_dataframe(saved[[c for c in saved_display_cols if c in saved.columns]], hide_index=True, auto_collapse=False)

        edit_col1, edit_col2 = st.columns([1, 2])
        with edit_col1:
            saved_cusips = saved["cusip"].dropna().astype(str).tolist()
            edit_cusip = st.selectbox("Edit saved CUSIP", saved_cusips, key="focused_watch_edit_cusip")
        with edit_col2:
            current_records = _focused_watchlist_records()
            current_note = current_records.get(str(edit_cusip), {}).get("note", "")
            edited_note = st.text_area(
                "Saved note",
                value=current_note,
                key=f"focused_watch_edit_note_{edit_cusip}",
                height=92,
            )
        update_col, remove_col = st.columns([1, 1])
        with update_col:
            if st.button("Update saved note", key="focused_watch_update_note"):
                current_records = _focused_watchlist_records()
                if str(edit_cusip) in current_records:
                    current_records[str(edit_cusip)]["note"] = edited_note
                    current_records[str(edit_cusip)]["updated_at"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
                    st.success(f"Updated note for {edit_cusip}.")
        with remove_col:
            if st.button("Remove saved CUSIP", key="focused_watch_remove_cusip"):
                current_records = _focused_watchlist_records()
                current_records.pop(str(edit_cusip), None)
                st.session_state["focused_watchlist"] = sorted(current_records.keys())
                st.info(f"Removed {edit_cusip}.")

        export_col1, export_col2 = st.columns([1, 1])
        with export_col1:
            st.download_button(
                "Download Watchlist CSV",
                data=saved.to_csv(index=False).encode("utf-8"),
                file_name=f"{selected_issuer}_watchlist.csv".replace(" ", "_"),
                mime="text/csv",
            )
        with export_col2:
            watch_md = _focused_watchlist_markdown(saved, selected_issuer)
            st.download_button(
                "Download Watchlist Markdown",
                data=watch_md.encode("utf-8"),
                file_name=f"{selected_issuer}_watchlist.md".replace(" ", "_"),
                mime="text/markdown",
            )
