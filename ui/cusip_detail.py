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


WATCHLIST_STAGE_OPTIONS = ["New", "Reviewing", "Need Data Check", "Approved", "Rejected"]
WATCHLIST_DECISION_OPTIONS = ["No decision", "Approve", "Reject", "Need more data", "Monitor"]
LEGACY_WATCHLIST_STATUS_MAP = {
    "Review": "Reviewing",
    "High priority": "Reviewing",
    "Needs data check": "Need Data Check",
    "Pass / monitor": "Approved",
    "Not saved": "New",
}


def _normalize_watchlist_status(status: object) -> str:
    value = str(status or "").strip()
    if value in WATCHLIST_STAGE_OPTIONS:
        return value
    return LEGACY_WATCHLIST_STATUS_MAP.get(value, "New")


def _focused_watchlist_records() -> dict:
    """Return mutable watchlist records, migrating older list-based session state."""
    if "focused_watchlist_records" not in st.session_state:
        records = {}
        for cusip in st.session_state.get("focused_watchlist", []):
            records[str(cusip)] = {
                "cusip": str(cusip),
                "issuer": "",
                "signal": "",
                "status": "New",
                "reason": "",
                "next_step": "",
                "note": "",
                "reviewer": "",
                "review_decision": "No decision",
                "review_note": "",
                "reviewed_at": "",
                "source": "Migrated",
                "added_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
            }
        st.session_state["focused_watchlist_records"] = records
    records = st.session_state["focused_watchlist_records"]
    for record in records.values():
        record["status"] = _normalize_watchlist_status(record.get("status"))
        record.setdefault("reason", "")
        record.setdefault("next_step", "")
        record.setdefault("reviewer", "")
        record.setdefault("review_decision", "No decision")
        record.setdefault("review_note", "")
        record.setdefault("reviewed_at", "")
    return records


def _upsert_focused_watchlist(
    cusip: object,
    issuer: str,
    source: str,
    row: pd.Series | dict | None = None,
    note: str = "",
    status: str = "New",
    reason: str = "",
    next_step: str = "",
    reviewer: str = "",
    review_decision: str = "",
    review_note: str = "",
):
    records = _focused_watchlist_records()
    key = str(cusip)
    existing = records.get(key, {})
    row_dict = row.to_dict() if isinstance(row, pd.Series) else (row or {})
    default_reason = reason or row_dict.get("signal", existing.get("reason", ""))
    records[key] = {
        "cusip": key,
        "issuer": issuer or existing.get("issuer", ""),
        "signal": row_dict.get("signal", existing.get("signal", "")),
        "status": _normalize_watchlist_status(status or existing.get("status", "New")),
        "reason": default_reason,
        "next_step": next_step if next_step else existing.get("next_step", ""),
        "maturity_bucket": row_dict.get("maturity_bucket", existing.get("maturity_bucket", "")),
        "current_spread_bps": row_dict.get("current_spread_bps", existing.get("current_spread_bps", pd.NA)),
        "peer_median_gap_bps": row_dict.get("peer_median_gap_bps", existing.get("peer_median_gap_bps", pd.NA)),
        "liquidity_score": row_dict.get("liquidity_score", existing.get("liquidity_score", pd.NA)),
        "rv_score": row_dict.get("rv_score", existing.get("rv_score", pd.NA)),
        "trade_count": row_dict.get("trade_count", existing.get("trade_count", pd.NA)),
        "total_trade_amount": row_dict.get("total_trade_amount", existing.get("total_trade_amount", pd.NA)),
        "latest_trade": row_dict.get("latest_trade", existing.get("latest_trade", pd.NA)),
        "note": note if note else existing.get("note", ""),
        "reviewer": reviewer if reviewer else existing.get("reviewer", ""),
        "review_decision": review_decision if review_decision else existing.get("review_decision", "No decision"),
        "review_note": review_note if review_note else existing.get("review_note", ""),
        "reviewed_at": existing.get("reviewed_at", ""),
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
                f"- Status: {row.get('status', 'New')}",
                f"- Reviewer: {row.get('reviewer', '') or 'N/A'}",
                f"- Decision: {row.get('review_decision', '') or 'N/A'}",
                f"- Reason: {row.get('reason', '') or 'N/A'}",
                f"- Next step: {row.get('next_step', '') or 'N/A'}",
                f"- Maturity bucket: {row.get('maturity_bucket', 'N/A')}",
                f"- Spread: {_fmt_bps(row.get('current_spread_bps'))}",
                f"- Peer median gap: {_fmt_bps(row.get('peer_median_gap_bps'))}",
                f"- Liquidity score: {_fmt_num(row.get('liquidity_score'))}",
                f"- RV score: {_fmt_num(row.get('rv_score'))}",
                f"- Note: {row.get('note', '') or 'N/A'}",
                f"- Review note: {row.get('review_note', '') or 'N/A'}",
                "",
            ]
        )
    return "\n".join(lines)


def _watchlist_card_html(row: pd.Series) -> str:
    status = _normalize_watchlist_status(row.get("status"))
    return f"""
<div style="border:1px solid #dbe3ee;border-left:5px solid #277568;border-radius:10px;padding:10px 11px;margin:8px 0;background:#fff;">
  <div style="font-size:0.78rem;color:#64748b;font-weight:780;">{_fmt_bps(row.get('current_spread_bps'))} / RV {_fmt_num(row.get('rv_score'))}</div>
  <div style="font-size:1rem;color:#111827;font-weight:820;line-height:1.2;overflow-wrap:anywhere;">{row.get('cusip', 'N/A')}</div>
  <div style="font-size:0.82rem;color:#475569;margin-top:4px;">{row.get('signal', 'Monitor')} / {row.get('maturity_bucket', 'N/A')}</div>
  <div style="font-size:0.78rem;color:#64748b;margin-top:6px;">{status} · next: {row.get('next_step', '') or 'N/A'}</div>
</div>
"""


def _update_watchlist_review_record(
    cusip: str,
    *,
    status: str,
    reviewer: str,
    review_decision: str,
    review_note: str,
    next_step: str,
    note: str,
) -> None:
    records = _focused_watchlist_records()
    if cusip not in records:
        return
    records[cusip]["status"] = _normalize_watchlist_status(status)
    records[cusip]["reviewer"] = reviewer
    records[cusip]["review_decision"] = review_decision
    records[cusip]["review_note"] = review_note
    records[cusip]["next_step"] = next_step
    records[cusip]["note"] = note
    records[cusip]["reviewed_at"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    records[cusip]["updated_at"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")


def render_watchlist_board(
    saved: pd.DataFrame,
    selected_issuer: str,
    safe_dataframe_func,
    *,
    key_prefix: str = "watchlist_board",
) -> pd.DataFrame:
    """Render saved candidates as a review pipeline board."""
    if saved is None or saved.empty:
        st.info("No saved CUSIPs yet.")
        return pd.DataFrame()

    board = saved.copy()
    board["status"] = board.get("status", pd.Series(index=board.index, dtype="object")).map(_normalize_watchlist_status)
    status_counts = board["status"].value_counts().to_dict()

    metric_cols = st.columns(len(WATCHLIST_STAGE_OPTIONS))
    for idx, stage in enumerate(WATCHLIST_STAGE_OPTIONS):
        with metric_cols[idx]:
            clean_metric_card(stage, f"{int(status_counts.get(stage, 0)):,}", size="small")

    st.subheader("Watchlist Board")
    board_cols = st.columns(len(WATCHLIST_STAGE_OPTIONS))
    for idx, stage in enumerate(WATCHLIST_STAGE_OPTIONS):
        stage_df = board[board["status"] == stage].copy()
        with board_cols[idx]:
            st.markdown(f"**{stage}**")
            if stage_df.empty:
                st.caption("No candidates.")
                continue
            sort_cols = [c for c in ["rv_score", "liquidity_score", "trade_count"] if c in stage_df.columns]
            if sort_cols:
                stage_df = stage_df.sort_values(sort_cols, ascending=False)
            for _, row in stage_df.head(4).iterrows():
                st.markdown(_watchlist_card_html(row), unsafe_allow_html=True)
            if len(stage_df) > 4:
                st.caption(f"+ {len(stage_df) - 4:,} more")

    st.subheader("Reviewer Mode")
    saved_cusips = board["cusip"].dropna().astype(str).tolist()
    selected = st.selectbox("Candidate", saved_cusips, key=f"{key_prefix}_selected_cusip")
    selected_row = board[board["cusip"].astype(str) == str(selected)].iloc[0]
    records = _focused_watchlist_records()
    current = records.get(str(selected), {})

    r1, r2, r3 = st.columns([1, 1, 1])
    with r1:
        reviewer = st.text_input(
            "Reviewer",
            value=str(current.get("reviewer", "")),
            key=f"{key_prefix}_reviewer_{selected}",
            placeholder="Name or team",
        )
    with r2:
        stage = st.selectbox(
            "Review stage",
            WATCHLIST_STAGE_OPTIONS,
            index=WATCHLIST_STAGE_OPTIONS.index(_normalize_watchlist_status(current.get("status", selected_row.get("status")))),
            key=f"{key_prefix}_stage_{selected}",
        )
    with r3:
        decision = st.selectbox(
            "Decision",
            WATCHLIST_DECISION_OPTIONS,
            index=WATCHLIST_DECISION_OPTIONS.index(current.get("review_decision", "No decision")) if current.get("review_decision", "No decision") in WATCHLIST_DECISION_OPTIONS else 0,
            key=f"{key_prefix}_decision_{selected}",
        )

    n1, n2 = st.columns([1, 1])
    with n1:
        next_step = st.text_input(
            "Next step",
            value=str(current.get("next_step", "")),
            key=f"{key_prefix}_next_{selected}",
            placeholder="Verify data / call desk / add to report",
        )
        note = st.text_area(
            "Candidate note",
            value=str(current.get("note", "")),
            key=f"{key_prefix}_note_{selected}",
            height=92,
        )
    with n2:
        review_note = st.text_area(
            "Reviewer note",
            value=str(current.get("review_note", "")),
            key=f"{key_prefix}_review_note_{selected}",
            height=138,
            placeholder="What was reviewed, what passed, what still needs evidence?",
        )

    save_col, remove_col = st.columns([1, 1])
    with save_col:
        if st.button("Save Review", key=f"{key_prefix}_save_{selected}"):
            _update_watchlist_review_record(
                str(selected),
                status=stage,
                reviewer=reviewer,
                review_decision=decision,
                review_note=review_note,
                next_step=next_step,
                note=note,
            )
            st.success(f"Saved review for {selected}.")
    with remove_col:
        if st.button("Remove Candidate", key=f"{key_prefix}_remove_{selected}"):
            records.pop(str(selected), None)
            st.session_state["focused_watchlist"] = sorted(records.keys())
            st.info(f"Removed {selected}.")

    st.subheader("Pipeline Table")
    table_cols = [
        "cusip", "issuer", "status", "review_decision", "reviewer", "next_step", "signal",
        "maturity_bucket", "current_spread_bps", "peer_median_gap_bps", "liquidity_score",
        "rv_score", "trade_count", "total_trade_amount", "latest_trade", "note",
        "review_note", "source", "updated_at", "reviewed_at",
    ]
    safe_dataframe_func(board[[c for c in table_cols if c in board.columns]], hide_index=True, auto_collapse=False)
    return board


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
    current_cusip = str(st.session_state.get("workbench_selected_cusip") or "").strip()
    default_idx = selector_options.index(current_cusip) if current_cusip in selector_options else 0
    if current_cusip in selector_options and st.session_state.get("focused_cusip_detail_select") != current_cusip:
        st.session_state["focused_cusip_detail_select"] = current_cusip
    elif st.session_state.get("focused_cusip_detail_select") not in selector_options:
        st.session_state["focused_cusip_detail_select"] = selector_options[default_idx]
    selected_cusip = st.selectbox(
        "Select CUSIP",
        selector_options,
        index=default_idx,
        key="focused_cusip_detail_select",
    )
    st.session_state["workbench_selected_cusip"] = selected_cusip
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
    existing_record = records.get(str(selected_cusip), {})
    existing_note = existing_record.get("note", "")
    status_options = WATCHLIST_STAGE_OPTIONS
    current_status = _normalize_watchlist_status(existing_record.get("status", "New"))
    if current_status not in status_options:
        current_status = "New"
    note_col, status_col, action_col = st.columns([2.1, 1, 1])
    with note_col:
        watch_note = st.text_area(
            "Watchlist note",
            value=existing_note,
            key=f"cusip_watch_note_{selected_cusip}",
            height=86,
            placeholder="Why this CUSIP is worth saving, what to verify, or how to frame it in the report.",
        )
    with status_col:
        watch_status = st.selectbox(
            "Status",
            status_options,
            index=status_options.index(current_status),
            key=f"cusip_watch_status_{selected_cusip}",
        )
        watch_next_step = st.text_input(
            "Next step",
            value=existing_record.get("next_step", ""),
            key=f"cusip_watch_next_step_{selected_cusip}",
            placeholder="Call / verify / monitor",
        )
    with action_col:
        st.caption("Save for watchlist/export.")
        if st.button("Save / Update Watchlist", key=f"save_watch_{selected_cusip}"):
            _upsert_focused_watchlist(
                selected_cusip,
                selected_issuer,
                "CUSIP Drilldown",
                selected_row,
                watch_note,
                status=watch_status,
                reason=str(selected_row.get("signal", "")),
                next_step=watch_next_step,
            )
            st.success(f"Saved {selected_cusip} to watchlist.")

    st.subheader("Trade Path")
    if not path.empty:
        path_panels = st.multiselect(
            "Path Panels",
            ["Spread", "Yield", "Price", "Par"],
            default=["Spread", "Par"],
            key=f"cusip_path_panels_{selected_cusip}",
            help="Spread is the primary panel. Add yield, price, or par as needed.",
        )
        if "Spread" not in path_panels:
            path_panels = ["Spread"] + path_panels

        subplot_titles = []
        specs = []
        row_heights = []
        if "Spread" in path_panels:
            subplot_titles.append("Spread Path")
            specs.append([{}])
            row_heights.append(0.50)
        if "Yield" in path_panels or "Price" in path_panels:
            subplot_titles.append("Yield / Price")
            specs.append([{"secondary_y": True}])
            row_heights.append(0.30)
        if "Par" in path_panels:
            subplot_titles.append("Par Amount")
            specs.append([{}])
            row_heights.append(0.20)
        total_height = sum(row_heights) or 1
        row_heights = [x / total_height for x in row_heights]

        fig_path = make_subplots(
            rows=len(subplot_titles),
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=row_heights,
            subplot_titles=tuple(subplot_titles),
            specs=specs,
        )
        row_idx = 1
        if "Spread" in path_panels:
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
                    row=row_idx,
                    col=1,
                )
            fig_path.update_yaxes(title_text="Spread (bps)", row=row_idx, col=1)
            row_idx += 1
        if "Yield" in path_panels or "Price" in path_panels:
            panel_row = row_idx
            if "Yield" in path_panels and pd.to_numeric(path["avg_yield"], errors="coerce").notna().any():
                fig_path.add_trace(
                    go.Scatter(
                        x=path["trade_date"],
                        y=path["avg_yield"],
                        mode="lines+markers",
                        name="Yield",
                        line=dict(width=2.4),
                        hovertemplate="%{x|%m/%d/%Y}<br>Yield: %{y:.3f}%<extra>Yield</extra>",
                    ),
                    row=panel_row,
                    col=1,
                    secondary_y=False,
                )
                fig_path.update_yaxes(title_text="Yield (%)", row=panel_row, col=1, secondary_y=False)
            if "Price" in path_panels and pd.to_numeric(path["avg_price"], errors="coerce").notna().any():
                fig_path.add_trace(
                    go.Scatter(
                        x=path["trade_date"],
                        y=path["avg_price"],
                        mode="lines+markers",
                        name="Price",
                        line=dict(width=2.0, dash="dash"),
                        hovertemplate="%{x|%m/%d/%Y}<br>Price: %{y:.2f}<extra>Price</extra>",
                    ),
                    row=panel_row,
                    col=1,
                    secondary_y=True,
                )
                fig_path.update_yaxes(title_text="Price", row=panel_row, col=1, secondary_y=True)
            row_idx += 1
        if "Par" in path_panels:
            fig_path.add_trace(
                go.Bar(
                    x=path["trade_date"],
                    y=path["par"],
                    name="Par amount",
                    hovertemplate="%{x|%m/%d/%Y}<br>Par: $%{y:,.0f}<extra>Par amount</extra>",
                ),
                row=row_idx,
                col=1,
            )
            fig_path.update_yaxes(title_text="Par", row=row_idx, col=1)
        fig_path.update_layout(
            title=f"{selected_cusip} Trade Path",
            height=760 if len(subplot_titles) >= 3 else 640,
            hovermode="x unified",
            legend_title_text="Series",
            margin=dict(l=40, r=50, t=85, b=45),
        )
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
            p1, p2, p3 = st.columns(3)
            with p1:
                clean_metric_card("Peer Median", _fmt_bps(peer_median_spread), size="small")
            with p2:
                clean_metric_card("Selected Gap", _fmt_bps(selected_gap_val), size="small")
            with p3:
                clean_metric_card("Peer Count", f"{len(peers):,}", size="small")
            peer_chart = pd.concat([peers[peers["is_selected"]], peers[~peers["is_selected"]].head(7)], ignore_index=True)
            peer_chart = peer_chart.drop_duplicates(subset=["cusip"]).copy()
            peer_chart["peer_median_gap_bps"] = pd.to_numeric(peer_chart["peer_median_gap_bps"], errors="coerce")
            peer_chart = peer_chart.dropna(subset=["peer_median_gap_bps"])
            if not peer_chart.empty:
                peer_fig = go.Figure()
                liq_series = pd.to_numeric(
                    peer_chart["liquidity_score"] if "liquidity_score" in peer_chart.columns else pd.Series(np.nan, index=peer_chart.index),
                    errors="coerce",
                )
                rv_series = pd.to_numeric(
                    peer_chart["rv_score"] if "rv_score" in peer_chart.columns else pd.Series(np.nan, index=peer_chart.index),
                    errors="coerce",
                )
                peer_fig.add_trace(
                    go.Bar(
                        x=peer_chart["cusip"].astype(str),
                        y=peer_chart["peer_median_gap_bps"],
                        marker_color=np.where(peer_chart["is_selected"], "#e11d48", "#2f7f73"),
                        customdata=np.stack(
                            [
                                liq_series.fillna(np.nan),
                                rv_series.fillna(np.nan),
                            ],
                            axis=-1,
                        ),
                        hovertemplate=(
                            "%{x}<br>Peer gap: %{y:.1f} bps<br>"
                            "Liquidity: %{customdata[0]:.1f}<br>"
                            "RV: %{customdata[1]:.1f}<extra></extra>"
                        ),
                    )
                )
                peer_fig.add_hline(y=0, line_dash="dash", line_width=1)
                peer_fig.update_layout(
                    title="Same-Bucket Peer Gap",
                    height=320,
                    margin=dict(l=40, r=30, t=58, b=50),
                    yaxis_title="Gap to peer median (bps)",
                    xaxis_title="CUSIP",
                )
                safe_plotly_chart(peer_fig, width="stretch")
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
        top_cards = ranked.head(3)
        card_cols = st.columns(len(top_cards))
        for idx, (_, row) in enumerate(top_cards.iterrows()):
            with card_cols[idx]:
                clean_metric_card(
                    str(row.get("cusip", "N/A")),
                    row.get("signal", "Monitor"),
                    size="small",
                    note=f"RV {_fmt_num(row.get('rv_score'))} | Spread {_fmt_bps(row.get('current_spread_bps'))}",
                )
        with st.expander("Opportunity ranking table", expanded=False):
            safe_dataframe(ranked[[c for c in display_cols if c in ranked.columns]].head(50), hide_index=True)

    st.subheader("Watchlist")
    _focused_watchlist_records()
    add_options = ranked["cusip"].dropna().astype(str).head(150).tolist() if not ranked.empty else []
    add_col, note_col, status_col = st.columns([1.1, 1.6, 1])
    with add_col:
        selected_add = st.multiselect("Add CUSIPs", add_options, key="focused_rv_add_cusips")
    with note_col:
        bulk_note = st.text_input(
            "Note for selected CUSIPs",
            key="focused_rv_bulk_note",
            placeholder="Why these belong on the shortlist, or what to verify next.",
        )
    with status_col:
        bulk_status = st.selectbox(
            "Status",
            WATCHLIST_STAGE_OPTIONS,
            key="focused_rv_bulk_status",
        )
        bulk_next_step = st.text_input(
            "Next step",
            key="focused_rv_bulk_next_step",
            placeholder="Verify / call / monitor",
        )
    add_button_col, clear_button_col = st.columns([1, 1])
    with add_button_col:
        if st.button("Add selected to watchlist", key="focused_rv_add_selected"):
            for item in selected_add:
                row_match = ranked[ranked["cusip"].astype(str) == str(item)]
                row = row_match.iloc[0] if not row_match.empty else {"cusip": item}
                _upsert_focused_watchlist(
                    item,
                    selected_issuer,
                    "RV Ranking",
                    row,
                    bulk_note,
                    status=bulk_status,
                    reason=str(row.get("signal", "")),
                    next_step=bulk_next_step,
                )
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
        render_watchlist_board(saved, selected_issuer, safe_dataframe, key_prefix="focused_rv_watchlist_board")
        saved_display_cols = [
            "cusip", "issuer", "status", "review_decision", "reviewer", "reason", "next_step", "signal", "maturity_bucket", "current_spread_bps", "peer_median_gap_bps",
            "liquidity_score", "rv_score", "trade_count", "total_trade_amount", "latest_trade",
            "note", "review_note", "source", "updated_at", "reviewed_at",
        ]
        with st.expander("Saved candidate table", expanded=False):
            safe_dataframe(saved[[c for c in saved_display_cols if c in saved.columns]], hide_index=True, auto_collapse=False)

        edit_col1, edit_col2, edit_col3 = st.columns([1, 1.3, 1.5])
        with edit_col1:
            saved_cusips = saved["cusip"].dropna().astype(str).tolist()
            edit_cusip = st.selectbox("Edit saved CUSIP", saved_cusips, key="focused_watch_edit_cusip")
        with edit_col2:
            current_records = _focused_watchlist_records()
            edit_status_options = WATCHLIST_STAGE_OPTIONS
            current_status = _normalize_watchlist_status(current_records.get(str(edit_cusip), {}).get("status", "New"))
            if current_status not in edit_status_options:
                current_status = "New"
            edited_status = st.selectbox(
                "Saved status",
                edit_status_options,
                index=edit_status_options.index(current_status),
                key=f"focused_watch_edit_status_{edit_cusip}",
            )
            edited_next_step = st.text_input(
                "Saved next step",
                value=current_records.get(str(edit_cusip), {}).get("next_step", ""),
                key=f"focused_watch_edit_next_{edit_cusip}",
            )
        with edit_col3:
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
                    current_records[str(edit_cusip)]["status"] = edited_status
                    current_records[str(edit_cusip)]["next_step"] = edited_next_step
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
