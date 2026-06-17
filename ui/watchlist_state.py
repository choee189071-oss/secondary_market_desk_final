from __future__ import annotations

import pandas as pd
import streamlit as st

try:
    from ui.common import clean_metric_card
except Exception:  # pragma: no cover - deployment fallback
    def clean_metric_card(label, value, size="small", note=None, status="neutral"):
        st.metric(label, value)
        if note:
            st.caption(note)


WATCHLIST_STAGE_OPTIONS = ["New", "Reviewing", "Need Data Check", "Approved", "Rejected"]
WATCHLIST_DECISION_OPTIONS = ["No decision", "Approve", "Reject", "Need more data", "Monitor"]
LEGACY_WATCHLIST_STATUS_MAP = {
    "Review": "Reviewing",
    "High priority": "Reviewing",
    "Needs data check": "Need Data Check",
    "Pass / monitor": "Approved",
    "Not saved": "New",
}


def _fmt_num(x, digits: int = 1) -> str:
    try:
        if pd.isna(x):
            return "N/A"
        return f"{float(x):,.{digits}f}"
    except Exception:
        return "N/A"


def _fmt_bps(x, digits: int = 1) -> str:
    try:
        if pd.isna(x):
            return "N/A"
        return f"{float(x):,.{digits}f} bps"
    except Exception:
        return "N/A"


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
            "cusip",
            "signal",
            "maturity_bucket",
            "current_spread_bps",
            "peer_median_gap_bps",
            "liquidity_score",
            "rv_score",
            "trade_count",
            "total_trade_amount",
            "latest_trade",
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
  <div style="font-size:0.78rem;color:#64748b;margin-top:6px;">{status} - next: {row.get('next_step', '') or 'N/A'}</div>
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
        current_decision = current.get("review_decision", "No decision")
        decision = st.selectbox(
            "Decision",
            WATCHLIST_DECISION_OPTIONS,
            index=WATCHLIST_DECISION_OPTIONS.index(current_decision) if current_decision in WATCHLIST_DECISION_OPTIONS else 0,
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
        "cusip",
        "issuer",
        "status",
        "review_decision",
        "reviewer",
        "next_step",
        "signal",
        "maturity_bucket",
        "current_spread_bps",
        "peer_median_gap_bps",
        "liquidity_score",
        "rv_score",
        "trade_count",
        "total_trade_amount",
        "latest_trade",
        "note",
        "review_note",
        "source",
        "updated_at",
        "reviewed_at",
    ]
    safe_dataframe_func(board[[c for c in table_cols if c in board.columns]], hide_index=True, auto_collapse=False)
    return board
