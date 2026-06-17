from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from engine.scoring import add_workflow_spread_bps
from ui.common import (
    _fmt_bps,
    _fmt_mm,
    _fmt_num,
    _fmt_pct,
    _html_escape,
    clean_metric_card,
    safe_dataframe,
    safe_plotly_chart,
    section_anchor,
)


MATURITY_BUCKETS = [
    "All",
    "0-5 Years",
    "5-10 Years",
    "10-15 Years",
    "15-20 Years",
    "20-30 Years",
    "30+ Years",
]

TRADE_SIZE_BUCKETS = [
    "All",
    "Less than $1MM",
    "$1MM-$5MM",
    "$5MM-$10MM",
    "$10MM-$25MM",
    "Greater than $25MM",
]

TRADE_TYPE_BUCKETS = [
    "All",
    "Customer Buy",
    "Customer Sell",
    "Dealer Buy",
    "Dealer Sell",
    "Interdealer",
    "Other / Unknown",
]

LOT_BUCKETS = ["All", "Odd Lot", "Round Lot", "Block Trade"]

DATE_RANGE_OPTIONS = ["All", "1 Week", "1 Month", "3 Months", "6 Months", "1 Year", "Custom"]


@dataclass
class WorkbenchSelection:
    sector: str
    issuer: str
    date_range_label: str
    date_range: tuple[pd.Timestamp, pd.Timestamp] | None
    maturity_bucket: str
    trade_size_bucket: str
    trade_type_bucket: str
    lot_bucket: str


def _coerce_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def _coerce_amount(series: pd.Series | None, index: pd.Index) -> pd.Series:
    if series is None:
        return pd.Series(0.0, index=index)
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def _maturity_years(df: pd.DataFrame) -> pd.Series:
    if "maturity_year" in df.columns:
        return pd.to_numeric(df["maturity_year"], errors="coerce")
    if "years_to_maturity" in df.columns:
        return pd.to_numeric(df["years_to_maturity"], errors="coerce")
    if {"maturity_date", "trade_date"}.issubset(df.columns):
        maturity = pd.to_datetime(df["maturity_date"], errors="coerce")
        trade_date = pd.to_datetime(df["trade_date"], errors="coerce")
        return (maturity - trade_date).dt.days / 365.25
    return pd.Series(np.nan, index=df.index)


def _maturity_segment(years: object) -> str:
    try:
        y = float(years)
    except Exception:
        return "Unknown"
    if pd.isna(y):
        return "Unknown"
    if y < 5:
        return "0-5 Years"
    if y < 10:
        return "5-10 Years"
    if y < 15:
        return "10-15 Years"
    if y < 20:
        return "15-20 Years"
    if y < 30:
        return "20-30 Years"
    return "30+ Years"


def _trade_size_bucket(amount: object) -> str:
    try:
        value = float(amount)
    except Exception:
        return "Unknown"
    if pd.isna(value):
        return "Unknown"
    if value < 1_000_000:
        return "Less than $1MM"
    if value < 5_000_000:
        return "$1MM-$5MM"
    if value < 10_000_000:
        return "$5MM-$10MM"
    if value < 25_000_000:
        return "$10MM-$25MM"
    return "Greater than $25MM"


def _lot_bucket(amount: object) -> str:
    try:
        value = float(amount)
    except Exception:
        return "Unknown"
    if pd.isna(value):
        return "Unknown"
    if value < 100_000:
        return "Odd Lot"
    if value < 1_000_000:
        return "Round Lot"
    return "Block Trade"


def _trade_type_category(value: object) -> str:
    text = str(value or "").strip().lower()
    if not text or text in {"nan", "none"}:
        return "Other / Unknown"
    if "inter" in text and "dealer" in text:
        return "Interdealer"
    if "customer" in text or "cust" in text:
        if any(token in text for token in ["sell", "sold", "sld"]):
            return "Customer Sell"
        if any(token in text for token in ["buy", "bought", "bot"]):
            return "Customer Buy"
    if "dealer" in text:
        if any(token in text for token in ["sell", "sold", "sld"]):
            return "Dealer Sell"
        if any(token in text for token in ["buy", "bought", "bot"]):
            return "Dealer Buy"
    if text in {"b", "buy", "bought", "purchase"}:
        return "Customer Buy"
    if text in {"s", "sell", "sold"}:
        return "Customer Sell"
    return "Other / Unknown"


def _participant_group(value: object) -> str:
    category = str(value)
    if category.startswith("Customer"):
        return "Customer"
    if category.startswith("Dealer"):
        return "Dealer"
    if category == "Interdealer":
        return "Interdealer"
    return "Other / Unknown"


def prepare_workbench_data(market_df: pd.DataFrame) -> pd.DataFrame:
    base = add_workflow_spread_bps(market_df.copy())
    if base.empty:
        return base
    if "trade_date" in base.columns:
        base["trade_date"] = _coerce_date(base["trade_date"])
    else:
        base["trade_date"] = pd.NaT
    if "sector" not in base.columns:
        base["sector"] = "Unknown"
    base["sector"] = base["sector"].fillna("Unknown").astype(str).replace({"nan": "Unknown", "": "Unknown"})
    base["trade_amount"] = _coerce_amount(base.get("trade_amount"), base.index)
    base["yield"] = pd.to_numeric(base["yield"], errors="coerce") if "yield" in base.columns else pd.Series(np.nan, index=base.index)
    base["spread_bps"] = pd.to_numeric(base["spread_bps"], errors="coerce") if "spread_bps" in base.columns else pd.Series(np.nan, index=base.index)
    base["workbench_maturity_year"] = _maturity_years(base)
    base["workbench_maturity_bucket"] = base["workbench_maturity_year"].map(_maturity_segment)
    base["trade_size_bucket"] = base["trade_amount"].map(_trade_size_bucket)
    base["lot_bucket"] = base["trade_amount"].map(_lot_bucket)
    raw_trade_type = base["trade_type"] if "trade_type" in base.columns else pd.Series("", index=base.index)
    base["trade_type_bucket"] = raw_trade_type.map(_trade_type_category)
    base["participant_group"] = base["trade_type_bucket"].map(_participant_group)
    return base


def _date_range_for_option(df: pd.DataFrame, option: str) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    dates = pd.to_datetime(df.get("trade_date"), errors="coerce").dropna()
    if dates.empty:
        return None
    end = dates.max().normalize()
    if option == "All":
        return None
    if option == "1 Week":
        start = end - pd.DateOffset(weeks=1)
    elif option == "1 Month":
        start = end - pd.DateOffset(months=1)
    elif option == "3 Months":
        start = end - pd.DateOffset(months=3)
    elif option == "6 Months":
        start = end - pd.DateOffset(months=6)
    elif option == "1 Year":
        start = end - pd.DateOffset(years=1)
    else:
        start = dates.min().normalize()
    return max(start, dates.min().normalize()), end


def _apply_workbench_filters(df: pd.DataFrame, selection: WorkbenchSelection, issuer: str | None = None) -> pd.DataFrame:
    out = df.copy()
    if selection.sector != "All" and "sector" in out.columns:
        out = out[out["sector"].astype(str) == str(selection.sector)]
    if issuer:
        out = out[out["issuer"].astype(str) == str(issuer)]
    if selection.date_range is not None and "trade_date" in out.columns:
        start, end = selection.date_range
        dates = pd.to_datetime(out["trade_date"], errors="coerce")
        out = out[(dates >= start) & (dates <= end)]
    if selection.maturity_bucket != "All":
        out = out[out["workbench_maturity_bucket"] == selection.maturity_bucket]
    if selection.trade_size_bucket != "All":
        out = out[out["trade_size_bucket"] == selection.trade_size_bucket]
    if selection.trade_type_bucket != "All":
        out = out[out["trade_type_bucket"] == selection.trade_type_bucket]
    if selection.lot_bucket != "All":
        out = out[out["lot_bucket"] == selection.lot_bucket]
    return out


def _aggregation_by(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if df.empty or group_col not in df.columns:
        return pd.DataFrame(columns=[group_col, "trade_count", "par_traded", "avg_trade_size"])
    grouped = (
        df.groupby(group_col, dropna=False)
        .agg(
            trade_count=("cusip", "count") if "cusip" in df.columns else ("trade_amount", "count"),
            par_traded=("trade_amount", "sum"),
            avg_trade_size=("trade_amount", "mean"),
            avg_yield=("yield", "mean"),
            avg_spread_bps=("spread_bps", "mean"),
        )
        .reset_index()
    )
    return grouped.sort_values("par_traded", ascending=False)


def _top_bucket(df: pd.DataFrame, group_col: str, metric: str = "trade_count") -> str:
    if df.empty or group_col not in df.columns:
        return "N/A"
    grouped = _aggregation_by(df, group_col)
    if grouped.empty or metric not in grouped.columns:
        return "N/A"
    row = grouped.dropna(subset=[metric]).sort_values(metric, ascending=False).head(1)
    if row.empty:
        return "N/A"
    return str(row[group_col].iloc[0])


def _security_detail_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "cusip" not in df.columns:
        return pd.DataFrame()
    agg = {
        "issuer": ("issuer", "first") if "issuer" in df.columns else ("cusip", "count"),
        "par_traded": ("trade_amount", "sum"),
        "trade_count": ("cusip", "count"),
        "average_yield": ("yield", "mean"),
        "average_spread_bps": ("spread_bps", "mean"),
        "last_trade_date": ("trade_date", "max"),
        "maturity_bucket": ("workbench_maturity_bucket", "first"),
        "trade_size_bucket": ("trade_size_bucket", "first"),
    }
    if "coupon" in df.columns:
        agg["coupon"] = ("coupon", "first")
    if "maturity_date" in df.columns:
        agg["maturity_date"] = ("maturity_date", "first")
    elif "maturity_bond" in df.columns:
        agg["maturity_date"] = ("maturity_bond", "first")

    detail = df.groupby("cusip", dropna=False).agg(**agg).reset_index()
    for col in ["average_yield", "average_spread_bps"]:
        if col in detail.columns:
            detail[col] = pd.to_numeric(detail[col], errors="coerce").round(3 if col == "average_yield" else 1)
    return detail.sort_values(["par_traded", "trade_count"], ascending=False)


def _peer_metrics(df: pd.DataFrame, issuers: list[str]) -> pd.DataFrame:
    if df.empty or not issuers:
        return pd.DataFrame()
    rows = []
    for issuer, group in df[df["issuer"].astype(str).isin([str(x) for x in issuers])].groupby("issuer"):
        trade_count = len(group)
        par_traded = float(pd.to_numeric(group["trade_amount"], errors="coerce").sum())
        avg_yield = pd.to_numeric(group["yield"], errors="coerce").mean()
        avg_spread = pd.to_numeric(group["spread_bps"], errors="coerce").mean()
        dates = pd.to_datetime(group["trade_date"], errors="coerce").dropna()
        days_since_last = np.nan if dates.empty else (pd.to_datetime(df["trade_date"], errors="coerce").max() - dates.max()).days
        count_rank = trade_count
        amount_rank = par_traded
        recency_component = max(0.0, 100.0 - float(days_since_last or 0))
        liquidity_score = min(100.0, (np.log1p(count_rank) * 15) + (np.log1p(max(amount_rank, 0)) * 2.2) + recency_component * 0.25)
        rows.append(
            {
                "Issuer": issuer,
                "Trade Volume": par_traded,
                "Trade Count": trade_count,
                "Average Yield": avg_yield,
                "Average Spread": avg_spread,
                "Liquidity Score": round(liquidity_score, 1),
            }
        )
    return pd.DataFrame(rows).sort_values("Trade Volume", ascending=False)


def _active_filter_summary(selection: WorkbenchSelection):
    date_text = selection.date_range_label
    if selection.date_range is not None:
        date_text = f"{selection.date_range[0]:%m/%d/%Y} to {selection.date_range[1]:%m/%d/%Y}"
    summary_items = [
        ("Sector", selection.sector),
        ("Issuer", selection.issuer),
        ("Date", date_text),
        ("Maturity", selection.maturity_bucket),
        ("Trade Size", selection.trade_size_bucket),
        ("Trade Type", selection.trade_type_bucket),
        ("Lot", selection.lot_bucket),
    ]
    parts = ["<div class='focus-band'><b>Active Filters:</b> "]
    parts.extend(
        [
            f"<span style='margin-right:14px;'><b>{_html_escape(k)}:</b> {_html_escape(v)}</span>"
            for k, v in summary_items
        ]
    )
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def _render_summary_cards(filtered_df: pd.DataFrame):
    total_par = float(pd.to_numeric(filtered_df.get("trade_amount"), errors="coerce").sum()) if not filtered_df.empty else 0.0
    trade_count = len(filtered_df)
    avg_trade_size = total_par / trade_count if trade_count else np.nan
    c1, c2, c3 = st.columns(3)
    with c1:
        clean_metric_card("Most Active Maturity", _top_bucket(filtered_df, "workbench_maturity_bucket", "trade_count"), size="small")
    with c2:
        clean_metric_card("Largest Size Bucket", _top_bucket(filtered_df, "trade_size_bucket", "par_traded"), size="small")
    with c3:
        clean_metric_card("Most Active Type", _top_bucket(filtered_df, "trade_type_bucket", "trade_count"), size="small")
    c4, c5, c6 = st.columns(3)
    with c4:
        clean_metric_card("Total Par Traded", _fmt_mm(total_par), size="small")
    with c5:
        clean_metric_card("Trade Count", f"{trade_count:,}", size="small")
    with c6:
        clean_metric_card("Average Trade Size", _fmt_mm(avg_trade_size), size="small")


def _render_volume_overview(filtered_df: pd.DataFrame):
    st.subheader("Trading Volume Overview")
    if filtered_df.empty:
        st.info("No trades match the selected filters.")
        return
    metric_label = st.radio(
        "Volume metric",
        ["Par Traded", "Trade Count", "Average Trade Size"],
        horizontal=True,
        label_visibility="collapsed",
    )
    metric_col = {
        "Par Traded": "par_traded",
        "Trade Count": "trade_count",
        "Average Trade Size": "avg_trade_size",
    }[metric_label]
    tabs = st.tabs(["Maturity", "Trade Size", "Trade Type"])
    group_specs = [
        ("workbench_maturity_bucket", "Maturity Bucket"),
        ("trade_size_bucket", "Trade Size Bucket"),
        ("trade_type_bucket", "Trade Type"),
    ]
    for tab, (group_col, label) in zip(tabs, group_specs):
        with tab:
            chart_df = _aggregation_by(filtered_df, group_col)
            fig = px.bar(chart_df, x=group_col, y=metric_col, text_auto=".2s", title=f"{metric_label} by {label}")
            fig.update_layout(height=360, xaxis_title=label, yaxis_title=metric_label)
            safe_plotly_chart(fig)


def _render_activity_concentration_map(filtered_df: pd.DataFrame):
    st.subheader("Activity Concentration Map")
    if filtered_df.empty:
        st.info("No trades match the selected filters.")
        return
    size_metric = st.radio(
        "Bubble size",
        ["Par Amount", "Trade Count"],
        horizontal=True,
        label_visibility="collapsed",
    )
    grouped = (
        filtered_df.groupby(["workbench_maturity_bucket", "trade_size_bucket"], dropna=False)
        .agg(
            trade_count=("cusip", "count") if "cusip" in filtered_df.columns else ("trade_amount", "count"),
            par_amount=("trade_amount", "sum"),
            average_yield=("yield", "mean"),
            average_spread_bps=("spread_bps", "mean"),
        )
        .reset_index()
    )
    if grouped.empty:
        st.info("No maturity / trade-size concentration can be calculated.")
        return
    participant = (
        filtered_df.groupby(["workbench_maturity_bucket", "trade_size_bucket", "participant_group"], dropna=False)
        .agg(participant_par=("trade_amount", "sum"), participant_count=("cusip", "count") if "cusip" in filtered_df.columns else ("trade_amount", "count"))
        .reset_index()
        .sort_values(["workbench_maturity_bucket", "trade_size_bucket", "participant_par", "participant_count"], ascending=[True, True, False, False])
    )
    dominant = participant.drop_duplicates(["workbench_maturity_bucket", "trade_size_bucket"])[
        ["workbench_maturity_bucket", "trade_size_bucket", "participant_group"]
    ]
    grouped = grouped.merge(dominant, on=["workbench_maturity_bucket", "trade_size_bucket"], how="left")
    grouped = grouped[grouped["workbench_maturity_bucket"].isin([x for x in MATURITY_BUCKETS if x != "All"])]
    grouped = grouped[grouped["trade_size_bucket"].isin([x for x in TRADE_SIZE_BUCKETS if x != "All"])]
    if grouped.empty:
        st.info("No known maturity / trade-size buckets match the selected filters.")
        return
    size_col = "par_amount" if size_metric == "Par Amount" else "trade_count"
    fig = px.scatter(
        grouped,
        x="workbench_maturity_bucket",
        y="trade_size_bucket",
        size=size_col,
        color="participant_group",
        category_orders={
            "workbench_maturity_bucket": [x for x in MATURITY_BUCKETS if x != "All"],
            "trade_size_bucket": [x for x in TRADE_SIZE_BUCKETS if x != "All"],
        },
        hover_data={
            "workbench_maturity_bucket": True,
            "trade_size_bucket": True,
            "participant_group": True,
            "par_amount": ":,.0f",
            "trade_count": ":,",
            "average_yield": ":.3f",
            "average_spread_bps": ":.1f",
        },
        labels={
            "workbench_maturity_bucket": "Maturity Bucket",
            "trade_size_bucket": "Trade Size Bucket",
            "participant_group": "Dominant Participant",
            "par_amount": "Par Amount",
            "trade_count": "Trade Count",
            "average_yield": "Average Yield",
            "average_spread_bps": "Average Spread",
        },
        title="Where Filtered Trading Is Concentrated",
        size_max=48,
    )
    fig.update_layout(height=430, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
    safe_plotly_chart(fig)


def _render_participation(filtered_df: pd.DataFrame):
    st.subheader("Dealer vs Customer Participation")
    if filtered_df.empty:
        st.info("No trades match the selected filters.")
        return
    part = (
        filtered_df.groupby("participant_group", dropna=False)
        .agg(par_traded=("trade_amount", "sum"), trade_count=("cusip", "count"))
        .reset_index()
        .sort_values("par_traded", ascending=False)
    )
    total_par = part["par_traded"].sum()
    part["par_share"] = np.where(total_par > 0, part["par_traded"] / total_par * 100, 0)
    c1, c2 = st.columns([0.48, 0.52])
    with c1:
        fig = px.pie(part, values="par_traded", names="participant_group", hole=0.45, title="Par Share")
        fig.update_layout(height=360)
        safe_plotly_chart(fig)
    with c2:
        safe_dataframe(
            part.rename(
                columns={
                    "participant_group": "Participant",
                    "par_traded": "Par Traded",
                    "trade_count": "Trade Count",
                    "par_share": "Par %",
                }
            ),
            hide_index=True,
            auto_collapse=False,
        )


def _render_liquidity_dashboard(filtered_df: pd.DataFrame):
    st.subheader("Liquidity Dashboard")
    if filtered_df.empty:
        st.info("No trades match the selected filters.")
        return
    max_date = pd.to_datetime(filtered_df["trade_date"], errors="coerce").max()
    grouped = (
        filtered_df.groupby(["workbench_maturity_bucket", "trade_size_bucket"], dropna=False)
        .agg(
            average_spread=("spread_bps", "mean"),
            average_yield=("yield", "mean"),
            trade_frequency=("cusip", "count"),
            last_trade_date=("trade_date", "max"),
            par_traded=("trade_amount", "sum"),
        )
        .reset_index()
    )
    grouped["days_since_last_trade"] = (max_date - pd.to_datetime(grouped["last_trade_date"], errors="coerce")).dt.days
    metric = st.selectbox(
        "Liquidity metric",
        ["Trade Frequency", "Par Traded", "Average Spread", "Average Yield", "Days Since Last Trade"],
        index=0,
    )
    metric_col = {
        "Trade Frequency": "trade_frequency",
        "Par Traded": "par_traded",
        "Average Spread": "average_spread",
        "Average Yield": "average_yield",
        "Days Since Last Trade": "days_since_last_trade",
    }[metric]
    grouped = grouped[grouped["workbench_maturity_bucket"].isin([x for x in MATURITY_BUCKETS if x != "All"])]
    grouped = grouped[grouped["trade_size_bucket"].isin([x for x in TRADE_SIZE_BUCKETS if x != "All"])]
    if grouped.empty:
        st.info("No liquidity bands match the selected filters.")
        return
    grouped["bucket_label"] = grouped["workbench_maturity_bucket"].astype(str) + " / " + grouped["trade_size_bucket"].astype(str)
    grouped = grouped.sort_values(metric_col, ascending=(metric_col == "days_since_last_trade"))
    fig = px.bar(
        grouped.head(18),
        x=metric_col,
        y="bucket_label",
        orientation="h",
        color="workbench_maturity_bucket",
        hover_data={
            "trade_frequency": ":,",
            "par_traded": ":,.0f",
            "average_spread": ":.1f",
            "average_yield": ":.3f",
            "days_since_last_trade": ":.0f",
        },
        labels={
            metric_col: metric,
            "bucket_label": "Maturity / Size Band",
            "workbench_maturity_bucket": "Maturity",
        },
        title=f"Ranked Liquidity Bands by {metric}",
    )
    fig.update_layout(height=440, yaxis=dict(categoryorder="total ascending"), legend=dict(orientation="h"))
    safe_plotly_chart(fig)
    with st.expander("Liquidity detail table", expanded=False):
        safe_dataframe(grouped, hide_index=True)


def _render_security_drilldown(filtered_df: pd.DataFrame) -> pd.DataFrame:
    section_anchor("workbench-security-drilldown", "4. Security Drilldown")
    st.markdown(
        "<div class='focus-band'><b>Workflow:</b> use aggregate filters above, then inspect the CUSIPs driving the selected activity.</div>",
        unsafe_allow_html=True,
    )
    if filtered_df.empty:
        st.info("No security rows match the selected filters.")
        return pd.DataFrame()

    c1, c2 = st.columns(2)
    maturity_options = ["Current filters"] + sorted([x for x in filtered_df["workbench_maturity_bucket"].dropna().unique().tolist() if x != "Unknown"])
    size_options = ["Current filters"] + sorted([x for x in filtered_df["trade_size_bucket"].dropna().unique().tolist() if x != "Unknown"])
    with c1:
        drill_maturity = st.selectbox("Drilldown maturity", maturity_options)
    with c2:
        drill_size = st.selectbox("Drilldown trade size", size_options)

    drill_df = filtered_df.copy()
    if drill_maturity != "Current filters":
        drill_df = drill_df[drill_df["workbench_maturity_bucket"] == drill_maturity]
    if drill_size != "Current filters":
        drill_df = drill_df[drill_df["trade_size_bucket"] == drill_size]

    detail = _security_detail_table(drill_df)
    display_cols = [
        c
        for c in [
            "cusip",
            "issuer",
            "coupon",
            "maturity_date",
            "par_traded",
            "trade_count",
            "average_yield",
            "average_spread_bps",
            "last_trade_date",
            "maturity_bucket",
            "trade_size_bucket",
        ]
        if c in detail.columns
    ]
    safe_dataframe(detail[display_cols] if display_cols else detail, hide_index=True, top_rows=15)
    if not detail.empty:
        selected_cusip = st.selectbox("CUSIP path", detail["cusip"].astype(str).tolist())
        path = drill_df[drill_df["cusip"].astype(str) == str(selected_cusip)].copy()
        path = path.sort_values("trade_date")
        if not path.empty and "trade_date" in path.columns:
            fig = go.Figure()
            if pd.to_numeric(path["spread_bps"], errors="coerce").notna().any():
                fig.add_trace(go.Scatter(x=path["trade_date"], y=path["spread_bps"], mode="lines+markers", name="Spread bps"))
            if pd.to_numeric(path["yield"], errors="coerce").notna().any():
                fig.add_trace(go.Scatter(x=path["trade_date"], y=path["yield"], mode="lines+markers", name="Yield", yaxis="y2"))
            fig.update_layout(
                height=380,
                title=f"{selected_cusip} trade path",
                yaxis_title="Spread bps",
                yaxis2=dict(title="Yield", overlaying="y", side="right", showgrid=False),
                legend=dict(orientation="h"),
            )
            safe_plotly_chart(fig)
    return detail


def _render_peer_comparison(prepared_df: pd.DataFrame, selection: WorkbenchSelection, filtered_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    section_anchor("workbench-peer-comparison", "5. Peer Comparison")
    same_filter_universe = _apply_workbench_filters(prepared_df, selection, issuer=None)
    peers_available = [x for x in sorted(same_filter_universe["issuer"].dropna().astype(str).unique().tolist()) if x != selection.issuer]
    default_peers = peers_available[:2]
    peer_issuers = st.multiselect("Peer issuers", peers_available, default=default_peers)
    issuers = [selection.issuer] + peer_issuers
    metrics = _peer_metrics(same_filter_universe, issuers)
    if metrics.empty:
        st.info("No peer trades match the same filters.")
        return pd.DataFrame(), peer_issuers
    c1, c2 = st.columns([0.55, 0.45])
    with c1:
        fig = px.bar(metrics, x="Issuer", y="Trade Volume", text_auto=".2s", title="Trade Volume Under Same Filters")
        fig.update_layout(height=360)
        safe_plotly_chart(fig)
    with c2:
        safe_dataframe(metrics, hide_index=True, auto_collapse=False)
    return metrics, peer_issuers


def _build_narrative(filtered_df: pd.DataFrame, security_detail: pd.DataFrame, selection: WorkbenchSelection) -> list[str]:
    if filtered_df.empty:
        return ["No observations: no trades match the active filters."]
    total_par = float(pd.to_numeric(filtered_df["trade_amount"], errors="coerce").sum())
    trade_count = len(filtered_df)
    top_maturity = _top_bucket(filtered_df, "workbench_maturity_bucket", "trade_count")
    top_size = _top_bucket(filtered_df, "trade_size_bucket", "par_traded")
    top_type = _top_bucket(filtered_df, "trade_type_bucket", "trade_count")
    observations = [
        f"Most activity occurred in {top_maturity}, based on {trade_count:,} filtered trade(s).",
        f"The largest par concentration was in {top_size}, with total par traded of {_fmt_mm(total_par)}.",
        f"The most active trade type was {top_type}.",
    ]

    part = filtered_df.groupby("participant_group").agg(par_traded=("trade_amount", "sum")).reset_index()
    if not part.empty and total_par > 0:
        top_part = part.sort_values("par_traded", ascending=False).iloc[0]
        observations.append(f"{top_part['participant_group']} activity represented {_fmt_pct(top_part['par_traded'] / total_par * 100)} of traded par.")

    long_end = filtered_df[filtered_df["workbench_maturity_bucket"].isin(["20-30 Years", "30+ Years"])]
    if not long_end.empty and len(long_end) / len(filtered_df) >= 0.45:
        observations.append("Trading was concentrated in longer maturities, suggesting stronger institutional or duration-focused participation.")

    if not security_detail.empty:
        top_security = security_detail.iloc[0]
        observations.append(
            f"{top_security.get('cusip')} drove the largest CUSIP-level par footprint at {_fmt_mm(top_security.get('par_traded'))}."
        )
    if selection.trade_type_bucket != "All":
        observations.append(f"All observations are filtered to {selection.trade_type_bucket}; compare against All to separate participation effects.")
    return observations


def _excel_bytes(filtered_df: pd.DataFrame, security_detail: pd.DataFrame, peer_metrics: pd.DataFrame) -> bytes | None:
    buffer = io.BytesIO()
    try:
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            filtered_df.to_excel(writer, sheet_name="Filtered Trades", index=False)
            security_detail.to_excel(writer, sheet_name="Security Drilldown", index=False)
            peer_metrics.to_excel(writer, sheet_name="Peer Comparison", index=False)
        return buffer.getvalue()
    except Exception:
        return None


def _pdf_bytes(selection: WorkbenchSelection, observations: list[str], filtered_df: pd.DataFrame) -> bytes | None:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    except Exception:
        return None
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    total_par = float(pd.to_numeric(filtered_df.get("trade_amount"), errors="coerce").sum()) if not filtered_df.empty else 0.0
    rows = [
        ["Issuer", selection.issuer],
        ["Sector", selection.sector],
        ["Maturity", selection.maturity_bucket],
        ["Trade Size", selection.trade_size_bucket],
        ["Trade Type", selection.trade_type_bucket],
        ["Trade Count", f"{len(filtered_df):,}"],
        ["Total Par", _fmt_mm(total_par)],
    ]
    story = [Paragraph("Trading Workbench Summary", styles["Title"]), Spacer(1, 10)]
    table = Table(rows, colWidths=[120, 340])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.extend([table, Spacer(1, 14), Paragraph("Narrative Observations", styles["Heading2"])])
    for obs in observations:
        story.append(Paragraph(f"- {obs}", styles["BodyText"]))
    doc.build(story)
    return buffer.getvalue()


def render_trading_workbench(
    market_df: pd.DataFrame,
    bonds_df: pd.DataFrame | None = None,
    issuer_master: pd.DataFrame | None = None,
    benchmark_source_mode: str = "Trade Sheet Index / Index Rate",
) -> dict:
    """Render the redesigned trading analysis workbench."""
    prepared = prepare_workbench_data(market_df)
    if prepared.empty:
        st.info("No usable trade rows are available for the trading workbench.")
        return {}

    section_anchor("workbench-issuer-selection", "1. Issuer Selection")
    st.markdown(
        "<div class='focus-band'><b>Goal:</b> set the issuer, sector, and date lens before looking at trading activity.</div>",
        unsafe_allow_html=True,
    )
    sector_options = ["All"] + sorted([x for x in prepared["sector"].dropna().astype(str).unique().tolist() if x and x != "nan"])
    c1, c2, c3 = st.columns([0.28, 0.34, 0.38])
    with c1:
        selected_sector = st.selectbox("Sector", sector_options)
    issuer_pool = prepared if selected_sector == "All" else prepared[prepared["sector"].astype(str) == selected_sector]
    issuer_options = sorted(issuer_pool["issuer"].dropna().astype(str).unique().tolist())
    if not issuer_options:
        issuer_options = sorted(prepared["issuer"].dropna().astype(str).unique().tolist())
    with c2:
        selected_issuer = st.selectbox("Issuer", issuer_options)
    issuer_base = prepared[prepared["issuer"].astype(str) == selected_issuer].copy()
    with c3:
        date_option = st.selectbox("Date Range", DATE_RANGE_OPTIONS, index=5)
    date_range = _date_range_for_option(issuer_base, date_option)
    if date_option == "Custom":
        dates = pd.to_datetime(issuer_base["trade_date"], errors="coerce").dropna()
        if not dates.empty:
            custom = st.date_input(
                "Custom Date Range",
                value=(dates.min().date(), dates.max().date()),
                min_value=dates.min().date(),
                max_value=dates.max().date(),
            )
            if isinstance(custom, (tuple, list)) and len(custom) == 2:
                date_range = (pd.Timestamp(custom[0]), pd.Timestamp(custom[1]))

    section_anchor("workbench-trading-filters", "2. Trading Filters")
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        maturity_bucket = st.selectbox("Maturity Filters", MATURITY_BUCKETS)
    with f2:
        trade_size_bucket = st.selectbox("Trade Size Filters", TRADE_SIZE_BUCKETS)
    with f3:
        observed_types = [x for x in TRADE_TYPE_BUCKETS if x == "All" or x in issuer_base["trade_type_bucket"].unique()]
        trade_type_bucket = st.selectbox("Trade Type Filters", observed_types or TRADE_TYPE_BUCKETS)
    with f4:
        lot_bucket = st.selectbox("Lot / Block Filter", LOT_BUCKETS)

    selection = WorkbenchSelection(
        sector=selected_sector,
        issuer=selected_issuer,
        date_range_label=date_option,
        date_range=date_range,
        maturity_bucket=maturity_bucket,
        trade_size_bucket=trade_size_bucket,
        trade_type_bucket=trade_type_bucket,
        lot_bucket=lot_bucket,
    )
    filtered_universe = _apply_workbench_filters(prepared, selection, issuer=None)
    filtered_issuer = _apply_workbench_filters(prepared, selection, issuer=selected_issuer)
    _active_filter_summary(selection)
    _render_summary_cards(filtered_issuer)

    section_anchor("workbench-market-analytics", "3. Market Analytics")
    st.caption(f"Benchmark source retained for spread calculations: {benchmark_source_mode}. Trading analysis is driven by the uploaded trade tape.")
    _render_volume_overview(filtered_issuer)
    a1, a2 = st.columns([0.52, 0.48])
    with a1:
        _render_activity_concentration_map(filtered_issuer)
    with a2:
        _render_participation(filtered_issuer)
    _render_liquidity_dashboard(filtered_issuer)

    security_detail = _render_security_drilldown(filtered_issuer)
    peer_metrics, peer_issuers = _render_peer_comparison(prepared, selection, filtered_issuer)

    section_anchor("workbench-narrative-insights", "5B. Narrative Read-Through")
    observations = _build_narrative(filtered_issuer, security_detail, selection)
    for obs in observations:
        st.markdown(f"<div class='methodology-note'>{_html_escape(obs)}</div>", unsafe_allow_html=True)

    return {
        "selection": selection,
        "prepared_df": prepared,
        "filtered_universe_df": filtered_universe,
        "filtered_issuer_df": filtered_issuer,
        "security_detail_df": security_detail,
        "peer_metrics_df": peer_metrics,
        "observations": observations,
        "selected_issuer": selected_issuer,
        "selected_sector": selected_sector,
        "peer_issuers": peer_issuers,
        "filtered_rows": len(filtered_issuer),
        "security_rows": len(security_detail),
        "peer_rows": len(peer_metrics),
    }
