from __future__ import annotations
import io
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from engine.scoring import (
    add_workflow_spread_bps,
    build_workflow_cusip_summary,
    focused_summary_with_peer_gaps,
)
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

MATURITY_YEAR_OPTIONS = [f"{year}Y" for year in range(1, 41)]

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

MAX_DYNAMIC_FILTER_OPTIONS = 500


COMMAND_TARGETS = [
    ("export", "Export / Methodology", "workflow-export-methodology"),
    ("methodology", "Export / Methodology", "workflow-export-methodology"),
    ("watch", "RV / Watchlist", "watchlist"),
    ("curve", "Issuer Curve", "issuer-curve"),
    ("rv", "RV / Watchlist", "rv-positioning"),
    ("cusip", "CUSIP Drilldown", "workbench-security-drilldown"),
    ("security", "CUSIP Drilldown", "workbench-security-drilldown"),
    ("path", "CUSIP Trade Path", "workbench-security-drilldown"),
    ("yield", "Yield / Relative Value", "yield-relative-value"),
    ("spread", "Desk Snapshot", "desk-market-snapshot"),
    ("chart", "Core Charts", "desk-market-snapshot"),
    ("snapshot", "Desk Snapshot", "desk-market-snapshot"),
    ("upload", "Upload / Data Audit", "file-readiness"),
    ("audit", "Upload / Data Audit", "file-readiness"),
]


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
    maturity_years: tuple[int, ...] = ()
    coupon_values: tuple[str, ...] = ()
    cusips: tuple[str, ...] = ()


@dataclass
class IssuerProfile:
    issuer: str
    sector: str
    row_count: int
    cusip_count: int
    total_par: float
    median_yield: float
    median_spread_bps: float
    latest_trade_date: pd.Timestamp | pd.NaT
    top_maturity: str
    top_trade_type: str
    benchmark_source_mode: str
    benchmark_coverage_pct: float | None
    cusip_quality_pct: float | None
    yield_quality_pct: float | None


@dataclass
class SecurityProfile:
    cusip: str
    issuer: str
    signal: str
    maturity_bucket: str
    trade_count: int
    total_par: float
    latest_trade_date: pd.Timestamp | pd.NaT
    latest_yield: float
    latest_price: float
    current_spread_bps: float
    liquidity_score: float
    rv_score: float
    peer_median_gap_bps: float


def _coerce_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def _coerce_amount(series: pd.Series | None, index: pd.Index) -> pd.Series:
    if series is None:
        return pd.Series(0.0, index=index)
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def _parse_maturity_year(value: object) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip().upper().replace("YEARS", "").replace("YEAR", "").replace("Y", "")
    match = re.search(r"\d+(?:\.\d+)?", text)
    if match:
        try:
            year = int(np.ceil(float(match.group(0))))
            return float(year) if 1 <= year <= 40 else np.nan
        except Exception:
            return np.nan
    try:
        year = int(np.ceil(float(value)))
        return float(year) if 1 <= year <= 40 else np.nan
    except Exception:
        return np.nan


def _maturity_years(df: pd.DataFrame) -> pd.Series:
    if "maturity_year" in df.columns:
        raw = df["maturity_year"]
    elif "years_to_maturity" in df.columns:
        raw = pd.to_numeric(df["years_to_maturity"], errors="coerce")
    elif {"maturity_date", "trade_date"}.issubset(df.columns):
        maturity = pd.to_datetime(df["maturity_date"], errors="coerce")
        trade_date = pd.to_datetime(df["trade_date"], errors="coerce")
        raw = (maturity - trade_date).dt.days / 365.25
    else:
        raw = pd.Series(np.nan, index=df.index)
    return raw.apply(_parse_maturity_year)


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


def _maturity_year_label(years: object) -> str:
    try:
        y = int(float(years))
    except Exception:
        return "Unknown"
    if 1 <= y <= 40:
        return f"{y}Y"
    return "Unknown"


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


def _format_coupon_value(value: object) -> object:
    if pd.isna(value):
        return pd.NA
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "unknown"}:
        return pd.NA
    numeric_text = text.replace("%", "").replace(",", "").strip()
    numeric = pd.to_numeric(pd.Series([numeric_text]), errors="coerce").iloc[0]
    if pd.notna(numeric):
        return f"{float(numeric):g}"
    return text


def _coupon_labels(df: pd.DataFrame) -> pd.Series:
    labels = pd.Series(pd.NA, index=df.index, dtype="object")
    for col in ["coupon", "coupon_trade", "coupon_bond"]:
        if col in df.columns:
            labels = labels.combine_first(df[col].map(_format_coupon_value))
    return labels


def _coupon_sort_key(value: object) -> tuple[int, float | str]:
    try:
        return (0, float(str(value).replace("%", "").strip()))
    except Exception:
        return (1, str(value))


def _unique_filter_options(df: pd.DataFrame, column: str, *, numeric_sort: bool = False) -> list[str]:
    if df.empty or column not in df.columns:
        return []
    values = [
        str(v)
        for v in df[column].dropna().astype(str).unique().tolist()
        if str(v).strip() and str(v).strip().lower() not in {"nan", "none", "unknown", "<na>"}
    ]
    if numeric_sort:
        return sorted(values, key=_coupon_sort_key)
    return sorted(values)


def _has_filter_values(df: pd.DataFrame, column: str) -> bool:
    if df.empty or column not in df.columns:
        return False
    return df[column].notna().any()


def _search_filter_options(
    df: pd.DataFrame,
    column: str,
    query: str = "",
    *,
    limit: int = MAX_DYNAMIC_FILTER_OPTIONS,
    sort_key=None,
) -> tuple[list[str], bool]:
    if df.empty or column not in df.columns:
        return [], False

    series = df[column].dropna().astype(str).str.strip()
    series = series[
        (series != "")
        & (~series.str.lower().isin({"nan", "none", "unknown", "<na>"}))
    ]
    query = str(query or "").strip()
    if query:
        query_upper = query.upper()
        series = series[series.str.upper().str.contains(re.escape(query_upper), na=False)]

    values = series.drop_duplicates().tolist()
    values = sorted(values, key=sort_key) if sort_key else sorted(values)
    limited = len(values) > limit
    return values[:limit], limited


def _filter_existing_values(df: pd.DataFrame, column: str, values: list[object] | tuple[object, ...]) -> list[str]:
    if df.empty or column not in df.columns or not values:
        return []

    series = df[column].dropna().astype(str)
    existing: list[str] = []
    for value in values:
        value_text = str(value).strip()
        if value_text and series.eq(value_text).any():
            existing.append(value_text)
    return existing


def _filter_summary_label(values: tuple[object, ...] | list[object], all_label: str = "All") -> str:
    if not values:
        return all_label
    labels = [str(v) for v in values]
    if len(labels) <= 4:
        return ", ".join(labels)
    return f"{len(labels)} selected"


def _normalized_lookup_token(value: object) -> str:
    return re.sub(r"[^A-Z0-9]+", "", str(value or "").upper())


def _participant_group(value: object) -> str:
    category = str(value)
    if category.startswith("Customer"):
        return "Customer"
    if category.startswith("Dealer"):
        return "Dealer"
    if category == "Interdealer":
        return "Interdealer"
    return "Other / Unknown"


@st.cache_data(show_spinner=False, max_entries=16)
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
    base["workbench_maturity_label"] = base["workbench_maturity_year"].map(_maturity_year_label)
    base["workbench_coupon"] = _coupon_labels(base)
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
    if selection.maturity_years and "workbench_maturity_year" in out.columns:
        years = pd.to_numeric(out["workbench_maturity_year"], errors="coerce")
        out = out[years.isin(list(selection.maturity_years))]
    elif selection.maturity_bucket != "All":
        out = out[out["workbench_maturity_bucket"] == selection.maturity_bucket]
    if selection.coupon_values and "workbench_coupon" in out.columns:
        out = out[out["workbench_coupon"].astype(str).isin([str(x) for x in selection.coupon_values])]
    if selection.cusips and "cusip" in out.columns:
        out = out[out["cusip"].astype(str).isin([str(x) for x in selection.cusips])]
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
        "maturity_bucket": ("workbench_maturity_label", "first") if "workbench_maturity_label" in df.columns else ("workbench_maturity_bucket", "first"),
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


def _nonnull_pct(df: pd.DataFrame, col: str) -> float | None:
    if df.empty or col not in df.columns:
        return None
    return float(df[col].notna().mean() * 100)


def _numeric_pct(df: pd.DataFrame, col: str) -> float | None:
    if df.empty or col not in df.columns:
        return None
    return float(pd.to_numeric(df[col], errors="coerce").notna().mean() * 100)


def _quality_status(rate: float | None, good: float = 90, warn: float = 65) -> str:
    if rate is None:
        return "neutral"
    if rate >= good:
        return "good"
    if rate >= warn:
        return "warn"
    return "bad"


def _fmt_date_short(value: object) -> str:
    try:
        dt = pd.to_datetime(value, errors="coerce")
    except Exception:
        dt = pd.NaT
    return "N/A" if pd.isna(dt) else f"{dt:%m/%d/%Y}"


def _fmt_rate(value: object) -> str:
    try:
        if pd.isna(value):
            return "N/A"
        return f"{float(value):.3f}%"
    except Exception:
        return "N/A"


def _build_cusip_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    return focused_summary_with_peer_gaps(build_workflow_cusip_summary(df))


def _sync_workbench_selected_cusip(source_key: str):
    selected = str(st.session_state.get(source_key) or "").strip()
    if not selected:
        return
    st.session_state["workbench_selected_cusip"] = selected
    for key in ["workbench_sidebar_cusip_select", "workbench_security_path_select", "focused_cusip_detail_select"]:
        st.session_state[key] = selected


def _best_cusip_for_profile(filtered_df: pd.DataFrame, summary: pd.DataFrame) -> str:
    current = str(st.session_state.get("workbench_selected_cusip") or "").strip()
    if current and not filtered_df.empty and "cusip" in filtered_df.columns:
        available = set(filtered_df["cusip"].dropna().astype(str))
        if current in available:
            return current
    if not summary.empty and "cusip" in summary.columns:
        return str(summary.iloc[0].get("cusip", ""))
    return ""


def _sidebar_cusip_options(filtered_df: pd.DataFrame, summary: pd.DataFrame) -> list[str]:
    if not summary.empty and "cusip" in summary.columns:
        options = summary["cusip"].dropna().astype(str).tolist()
    elif not filtered_df.empty and "cusip" in filtered_df.columns:
        options = sorted(filtered_df["cusip"].dropna().astype(str).unique().tolist())
    else:
        options = []
    return list(dict.fromkeys([value for value in options if value and value.lower() != "nan"]))


def _sidebar_cusip_labels(summary: pd.DataFrame) -> dict[str, str]:
    if summary.empty or "cusip" not in summary.columns:
        return {}
    labels: dict[str, str] = {}
    for _, row in summary.iterrows():
        cusip = str(row.get("cusip", "")).strip()
        if not cusip or cusip.lower() == "nan":
            continue
        parts = [cusip]
        maturity = str(row.get("maturity_bucket", "") or "").strip()
        signal = str(row.get("signal", "") or "").strip()
        trades = row.get("trade_count", pd.NA)
        if maturity and maturity.lower() != "nan":
            parts.append(maturity)
        if signal and signal.lower() != "nan":
            parts.append(signal)
        try:
            if pd.notna(trades):
                parts.append(f"{int(float(trades)):,} trades")
        except Exception:
            pass
        labels[cusip] = " / ".join(parts)
    return labels


def _build_issuer_profile(
    filtered_df: pd.DataFrame,
    selection: WorkbenchSelection,
    benchmark_source_mode: str,
) -> IssuerProfile:
    latest_date = pd.NaT
    if not filtered_df.empty and "trade_date" in filtered_df.columns:
        latest_date = pd.to_datetime(filtered_df["trade_date"], errors="coerce").dropna().max()
    benchmark_coverage = _numeric_pct(filtered_df, "active_benchmark_yield")
    return IssuerProfile(
        issuer=selection.issuer,
        sector=selection.sector,
        row_count=len(filtered_df),
        cusip_count=filtered_df["cusip"].nunique() if not filtered_df.empty and "cusip" in filtered_df.columns else 0,
        total_par=float(pd.to_numeric(filtered_df.get("trade_amount"), errors="coerce").sum()) if not filtered_df.empty else 0.0,
        median_yield=pd.to_numeric(filtered_df.get("yield"), errors="coerce").median() if not filtered_df.empty else np.nan,
        median_spread_bps=pd.to_numeric(filtered_df.get("spread_bps"), errors="coerce").median() if not filtered_df.empty else np.nan,
        latest_trade_date=latest_date,
        top_maturity=_top_bucket(filtered_df, "workbench_maturity_label", "trade_count"),
        top_trade_type=_top_bucket(filtered_df, "trade_type_bucket", "trade_count"),
        benchmark_source_mode=benchmark_source_mode,
        benchmark_coverage_pct=benchmark_coverage,
        cusip_quality_pct=_nonnull_pct(filtered_df, "cusip"),
        yield_quality_pct=_numeric_pct(filtered_df, "yield"),
    )


def _latest_security_trade(detail_df: pd.DataFrame) -> pd.Series | None:
    if detail_df.empty or "trade_date" not in detail_df.columns:
        return None
    dated = detail_df.copy()
    dated["trade_date"] = pd.to_datetime(dated["trade_date"], errors="coerce")
    dated = dated.dropna(subset=["trade_date"]).sort_values("trade_date")
    return None if dated.empty else dated.iloc[-1]


def _build_security_profile(
    filtered_df: pd.DataFrame,
    summary: pd.DataFrame,
    selected_cusip: str,
    selected_issuer: str,
) -> SecurityProfile | None:
    if not selected_cusip or filtered_df.empty or "cusip" not in filtered_df.columns:
        return None

    detail = filtered_df[filtered_df["cusip"].astype(str) == str(selected_cusip)].copy()
    if detail.empty:
        return None

    summary_row = pd.Series(dtype="object")
    if not summary.empty and "cusip" in summary.columns:
        hit = summary[summary["cusip"].astype(str) == str(selected_cusip)]
        if not hit.empty:
            summary_row = hit.iloc[0]

    latest = _latest_security_trade(detail)

    return SecurityProfile(
        cusip=str(selected_cusip),
        issuer=selected_issuer,
        signal=str(summary_row.get("signal", "Monitor") or "Monitor"),
        maturity_bucket=str(summary_row.get("maturity_bucket", detail.get("workbench_maturity_label", detail.get("workbench_maturity_bucket", pd.Series(["N/A"]))).iloc[0]) or "N/A"),
        trade_count=int(summary_row.get("trade_count", len(detail)) or len(detail)),
        total_par=float(summary_row.get("total_trade_amount", pd.to_numeric(detail.get("trade_amount"), errors="coerce").sum()) or 0),
        latest_trade_date=latest.get("trade_date") if latest is not None else pd.NaT,
        latest_yield=latest.get("yield") if latest is not None else np.nan,
        latest_price=latest.get("price") if latest is not None and "price" in latest.index else np.nan,
        current_spread_bps=summary_row.get("current_spread_bps", pd.to_numeric(detail.get("spread_bps"), errors="coerce").median()),
        liquidity_score=summary_row.get("liquidity_score", np.nan),
        rv_score=summary_row.get("rv_score", np.nan),
        peer_median_gap_bps=summary_row.get("peer_median_gap_bps", np.nan),
    )


def _render_inspector_css():
    st.markdown(
        """
<style>
.inspector-panel {
  position: relative;
  background: #ffffff;
  border: 1px solid #dbe3ee;
  border-radius: 14px;
  padding: 14px 12px 12px 12px;
  margin: 12px 0 10px 0;
}
.inspector-kicker {
  color: #64748b;
  font-size: 0.72rem;
  font-weight: 800;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}
.inspector-title {
  color: #111827;
  font-size: 1.08rem;
  font-weight: 820;
  line-height: 1.18;
  margin: 4px 0 4px 0;
  overflow-wrap: anywhere;
}
.inspector-subtitle {
  color: #64748b;
  font-size: 0.82rem;
  line-height: 1.3;
  margin-bottom: 10px;
}
.inspector-section {
  border-top: 1px solid #e5eaf2;
  margin-top: 10px;
  padding-top: 10px;
}
.inspector-section-heading {
  color: #334155;
  font-size: 0.82rem;
  font-weight: 800;
  margin-bottom: 6px;
}
.inspector-row {
  display: block;
  border-left: 4px solid #cbd5e1;
  padding: 7px 0 7px 9px;
}
.inspector-row.status-good { border-left-color: #15803d; }
.inspector-row.status-warn { border-left-color: #ca8a04; }
.inspector-row.status-bad { border-left-color: #b91c1c; }
.inspector-label {
  color: #64748b;
  font-size: 0.76rem;
  font-weight: 720;
}
.inspector-value {
  color: #111827;
  font-size: 0.82rem;
  font-weight: 760;
  text-align: left;
  overflow-wrap: anywhere;
  margin-top: 2px;
}
.inspector-chip {
  display: inline-block;
  background: #eef8f5;
  border: 1px solid #b9dcd5;
  border-radius: 999px;
  color: #174a43;
  font-size: 0.76rem;
  font-weight: 760;
  padding: 3px 8px;
  margin: 2px 4px 2px 0;
}
.workbench-analysis-block {
  border: 1px solid #e2e8f0;
  border-radius: 12px;
  padding: 16px 16px 10px 16px;
  background: #ffffff;
  margin-bottom: 14px;
}
.workbench-block-title {
  color: #1f2937;
  font-size: 1.22rem;
  font-weight: 820;
  line-height: 1.2;
  margin: 0 0 4px 0;
}
.workbench-block-caption {
  color: #64748b;
  font-size: 0.82rem;
  line-height: 1.35;
  margin-bottom: 10px;
}
</style>
""",
        unsafe_allow_html=True,
    )


def _inspector_rows(rows: list[tuple[str, str, str]]) -> str:
    html = []
    for label, value, status in rows:
        html.append(
            "<div class='inspector-row status-{status}'>"
            "<div class='inspector-label'>{label}</div>"
            "<div class='inspector-value'>{value}</div>"
            "</div>".format(
                status=_html_escape(status or "neutral"),
                label=_html_escape(label),
                value=_html_escape(value),
            )
        )
    return "".join(html)


def _render_right_inspector(
    selection: WorkbenchSelection,
    issuer_profile: IssuerProfile,
    filtered_df: pd.DataFrame,
    summary: pd.DataFrame,
) -> tuple[str, SecurityProfile | None]:
    quality_rows = [
        ("Benchmark", issuer_profile.benchmark_source_mode, _quality_status(issuer_profile.benchmark_coverage_pct, good=80, warn=40)),
        ("Benchmark coverage", "N/A" if issuer_profile.benchmark_coverage_pct is None else f"{issuer_profile.benchmark_coverage_pct:.1f}%", _quality_status(issuer_profile.benchmark_coverage_pct, good=80, warn=40)),
        ("CUSIP quality", "N/A" if issuer_profile.cusip_quality_pct is None else f"{issuer_profile.cusip_quality_pct:.1f}%", _quality_status(issuer_profile.cusip_quality_pct, good=95, warn=80)),
        ("Yield quality", "N/A" if issuer_profile.yield_quality_pct is None else f"{issuer_profile.yield_quality_pct:.1f}%", _quality_status(issuer_profile.yield_quality_pct, good=90, warn=70)),
    ]
    overview_rows = [
        ("Issuer", issuer_profile.issuer, "neutral"),
        ("Sector", issuer_profile.sector or "All", "neutral"),
        ("Rows / CUSIPs", f"{issuer_profile.row_count:,} / {issuer_profile.cusip_count:,}", "neutral"),
        ("Par", _fmt_mm(issuer_profile.total_par), "neutral"),
        ("Median spread", _fmt_bps(issuer_profile.median_spread_bps), "neutral"),
        ("Median yield", _fmt_rate(issuer_profile.median_yield), "neutral"),
        ("Latest trade", _fmt_date_short(issuer_profile.latest_trade_date), "neutral"),
        ("Top maturity", issuer_profile.top_maturity, "neutral"),
        ("Top type", issuer_profile.top_trade_type, "neutral"),
    ]

    st.markdown(
        f"""
<div class="inspector-panel">
  <div class="inspector-kicker">Inspector</div>
  <div class="inspector-title">{_html_escape(issuer_profile.issuer)}</div>
  <div class="inspector-subtitle">Start with the active issuer/filter scope, then narrow into one CUSIP.</div>
  <span class="inspector-chip">Issuer overview</span>
  <span class="inspector-chip">Data quality</span>
  <span class="inspector-chip">CUSIP drilldown</span>
  <div class="inspector-section">
    <div class="inspector-section-heading">Scope Overview</div>
    {_inspector_rows(overview_rows)}
  </div>
  <div class="inspector-section">
    <div class="inspector-section-heading">Data Quality</div>
    {_inspector_rows(quality_rows)}
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    cusip_options = _sidebar_cusip_options(filtered_df, summary)
    selected_cusip = _best_cusip_for_profile(filtered_df, summary)
    if selected_cusip and selected_cusip in cusip_options:
        st.session_state["workbench_sidebar_cusip_select"] = selected_cusip
    elif cusip_options:
        selected_cusip = cusip_options[0]
        st.session_state["workbench_sidebar_cusip_select"] = selected_cusip
    else:
        selected_cusip = ""

    option_labels = _sidebar_cusip_labels(summary)
    security_profile = None
    if cusip_options:
        selected_cusip = st.selectbox(
            "Inspect CUSIP",
            cusip_options,
            key="workbench_sidebar_cusip_select",
            format_func=lambda value: option_labels.get(str(value), str(value)),
            help="Choose the CUSIP shown in the sidebar detail panel.",
            on_change=_sync_workbench_selected_cusip,
            args=("workbench_sidebar_cusip_select",),
        )
        st.session_state["workbench_selected_cusip"] = selected_cusip
        security_profile = _build_security_profile(filtered_df, summary, selected_cusip, selection.issuer)

    security_rows: list[tuple[str, str, str]] = []
    if security_profile:
        security_rows = [
            ("CUSIP", security_profile.cusip, "neutral"),
            ("Signal", security_profile.signal, "neutral"),
            ("Maturity", security_profile.maturity_bucket, "neutral"),
            ("Spread", _fmt_bps(security_profile.current_spread_bps), "neutral"),
            ("Peer gap", _fmt_bps(security_profile.peer_median_gap_bps), "neutral"),
            ("Liquidity", _fmt_num(security_profile.liquidity_score), "neutral"),
            ("RV", _fmt_num(security_profile.rv_score), "neutral"),
            ("Trades", f"{security_profile.trade_count:,}", "neutral"),
            ("Par", _fmt_mm(security_profile.total_par), "neutral"),
            ("Latest trade", _fmt_date_short(security_profile.latest_trade_date), "neutral"),
            ("Latest yield", _fmt_rate(security_profile.latest_yield), "neutral"),
            ("Latest price", _fmt_num(security_profile.latest_price), "neutral"),
        ]

    st.markdown(
        f"""
<div class="inspector-panel">
  <div class="inspector-kicker">Selected CUSIP</div>
  <div class="inspector-title">{_html_escape(selected_cusip or "No CUSIP")}</div>
  <div class="inspector-subtitle">{_html_escape((security_profile.signal + " / " + security_profile.maturity_bucket) if security_profile else "No CUSIP is available inside the active filter.")}</div>
  <div class="inspector-section">
    <div class="inspector-section-heading">CUSIP Detail</div>
    {_inspector_rows(security_rows) if security_rows else "<div class='inspector-subtitle'>No CUSIP is available inside the active filter.</div>"}
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
    return selected_cusip, security_profile


def _active_filter_summary(selection: WorkbenchSelection):
    date_text = selection.date_range_label
    if selection.date_range is not None:
        date_text = f"{selection.date_range[0]:%m/%d/%Y} to {selection.date_range[1]:%m/%d/%Y}"
    summary_items = [
        ("Sector", selection.sector),
        ("Issuer", selection.issuer),
        ("Date", date_text),
        ("Maturity", selection.maturity_bucket),
        ("Coupon", _filter_summary_label(selection.coupon_values, "All")),
        ("CUSIP", _filter_summary_label(selection.cusips, "All")),
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


def _command_target(command: str) -> tuple[str, str] | tuple[None, None]:
    lower = command.lower()
    for token, label, anchor in COMMAND_TARGETS:
        if token in lower:
            return label, anchor
    return None, None


def _find_command_cusip(command: str, prepared_df: pd.DataFrame) -> str | None:
    if prepared_df.empty or "cusip" not in prepared_df.columns:
        return None
    command_token = _normalized_lookup_token(command)
    if len(command_token) < 6:
        return None
    candidates = prepared_df["cusip"].dropna().astype(str).unique().tolist()
    normalized = {_normalized_lookup_token(c): c for c in candidates}
    if command_token in normalized:
        return str(normalized[command_token])
    for token, cusip in normalized.items():
        if len(token) >= 6 and token in command_token:
            return str(cusip)
    return None


def _find_command_issuer(command: str, issuer_options: list[str]) -> str | None:
    lower = command.lower().strip()
    exact = [issuer for issuer in issuer_options if lower == issuer.lower()]
    if exact:
        return exact[0]
    contained = [issuer for issuer in issuer_options if issuer.lower() in lower]
    if contained:
        return sorted(contained, key=len, reverse=True)[0]
    token = _normalized_lookup_token(command)
    normalized = {_normalized_lookup_token(issuer): issuer for issuer in issuer_options}
    if token in normalized:
        return normalized[token]
    contained_norm = [issuer for norm, issuer in normalized.items() if norm and norm in token]
    if contained_norm:
        return sorted(contained_norm, key=len, reverse=True)[0]
    return None


def _sector_for_issuer(prepared_df: pd.DataFrame, issuer: str) -> str:
    if "sector" not in prepared_df.columns:
        return "All"
    values = (
        prepared_df.loc[prepared_df["issuer"].astype(str) == str(issuer), "sector"]
        .dropna()
        .astype(str)
        .tolist()
    )
    values = [v for v in values if v and v.lower() != "nan"]
    return values[0] if values else "All"


def _apply_workbench_command(command: str, prepared_df: pd.DataFrame) -> dict:
    issuer_options = sorted(prepared_df["issuer"].dropna().astype(str).unique().tolist())
    target_label, target_anchor = _command_target(command)
    selected_cusip = _find_command_cusip(command, prepared_df)
    selected_issuer = None
    message = ""

    if selected_cusip:
        rows = prepared_df[prepared_df["cusip"].astype(str) == str(selected_cusip)]
        if not rows.empty and "issuer" in rows.columns:
            selected_issuer = str(rows["issuer"].iloc[0])
        st.session_state["workbench_selected_cusip"] = selected_cusip
        if selected_issuer:
            st.session_state["workbench_selected_issuer"] = selected_issuer
            st.session_state["workbench_selected_sector"] = _sector_for_issuer(prepared_df, selected_issuer)
        target_label = target_label or "CUSIP Drilldown"
        target_anchor = target_anchor or "workbench-security-drilldown"
        message = f"Selected CUSIP {selected_cusip}" + (f" under {selected_issuer}" if selected_issuer else "")
    else:
        selected_issuer = _find_command_issuer(command, issuer_options)
        if selected_issuer:
            st.session_state["workbench_selected_issuer"] = selected_issuer
            st.session_state["workbench_selected_sector"] = _sector_for_issuer(prepared_df, selected_issuer)
            if "cusip" in prepared_df.columns:
                current_cusip = st.session_state.get("workbench_selected_cusip")
                if current_cusip:
                    issuer_cusips = set(
                        prepared_df.loc[prepared_df["issuer"].astype(str) == str(selected_issuer), "cusip"]
                        .dropna()
                        .astype(str)
                    )
                    if str(current_cusip) not in issuer_cusips:
                        st.session_state["workbench_selected_cusip"] = ""
            target_label = target_label or "Desk Snapshot"
            target_anchor = target_anchor or "desk-market-snapshot"
            message = f"Selected issuer {selected_issuer}"

    if not message and target_anchor:
        message = f"Ready to jump to {target_label}"
    if not message:
        message = "No issuer, CUSIP, or section matched that command."

    return {
        "message": message,
        "target_label": target_label,
        "target_anchor": target_anchor,
        "issuer": selected_issuer,
        "cusip": selected_cusip,
    }


def _render_workbench_command_bar(prepared_df: pd.DataFrame):
    section_anchor("workbench-command", "Workbench Command")
    st.markdown(
        "<div class='focus-band'><b>Command:</b> type an issuer, CUSIP, or section keyword such as <code>LADWP curve</code>, <code>544532NV2</code>, <code>watchlist</code>, or <code>export</code>.</div>",
        unsafe_allow_html=True,
    )
    with st.form("workbench_command_form", clear_on_submit=False):
        c1, c2 = st.columns([0.78, 0.22])
        with c1:
            command = st.text_input(
                "Command / Search",
                key="workbench_command_input",
                placeholder="Issuer, CUSIP, chart, curve, watchlist, export...",
                label_visibility="collapsed",
            )
        with c2:
            submitted = st.form_submit_button("Go")
    if submitted:
        st.session_state["workbench_command_feedback"] = _apply_workbench_command(command, prepared_df)

    feedback = st.session_state.get("workbench_command_feedback")
    if isinstance(feedback, dict) and feedback.get("message"):
        target_anchor = feedback.get("target_anchor")
        target_html = ""
        if target_anchor:
            target_html = f" | <a href='#{_html_escape(target_anchor)}'>Open {_html_escape(feedback.get('target_label') or 'target')}</a>"
        st.markdown(
            f"<div class='object-command-result'>{_html_escape(feedback.get('message'))}{target_html}</div>",
            unsafe_allow_html=True,
        )


def _date_range_text(selection: WorkbenchSelection) -> str:
    if selection.date_range is None:
        return selection.date_range_label
    return f"{selection.date_range[0]:%m/%d/%Y} - {selection.date_range[1]:%m/%d/%Y}"


def _render_object_status_bar(
    selection: WorkbenchSelection,
    filtered_df: pd.DataFrame,
    benchmark_source_mode: str,
):
    selected_cusip = st.session_state.get("workbench_selected_cusip") or "No CUSIP selected"
    total_par = float(pd.to_numeric(filtered_df.get("trade_amount"), errors="coerce").sum()) if not filtered_df.empty else 0.0
    cusip_count = filtered_df["cusip"].nunique() if not filtered_df.empty and "cusip" in filtered_df.columns else 0
    median_spread = pd.to_numeric(filtered_df.get("spread_bps"), errors="coerce").median() if not filtered_df.empty else np.nan
    items = [
        ("Issuer", selection.issuer),
        ("CUSIP", selected_cusip),
        ("Date", _date_range_text(selection)),
        ("Maturity", selection.maturity_bucket),
        ("Coupon", _filter_summary_label(selection.coupon_values, "All")),
        ("Benchmark", benchmark_source_mode),
        ("Rows / CUSIPs", f"{len(filtered_df):,} / {cusip_count:,}"),
        ("Par", _fmt_mm(total_par)),
        ("Median Spread", _fmt_bps(median_spread)),
    ]
    parts = ["<div class='object-status-grid'>"]
    for label, value in items:
        parts.append(
            f"<div class='object-status-item'><div class='object-status-label'>{_html_escape(label)}</div><div class='object-status-value'>{_html_escape(value)}</div></div>"
        )
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def _render_summary_cards(filtered_df: pd.DataFrame):
    total_par = float(pd.to_numeric(filtered_df.get("trade_amount"), errors="coerce").sum()) if not filtered_df.empty else 0.0
    trade_count = len(filtered_df)
    avg_trade_size = total_par / trade_count if trade_count else np.nan
    c1, c2, c3 = st.columns(3)
    with c1:
        clean_metric_card("Most Active Maturity", _top_bucket(filtered_df, "workbench_maturity_label", "trade_count"), size="small")
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
        ("workbench_maturity_label", "Maturity Year"),
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
    with st.container(border=True):
        st.markdown("<div class='workbench-block-title'>Activity Concentration</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='workbench-block-caption'>Ranks the maturity and trade-size bands driving the selected issuer scope.</div>",
            unsafe_allow_html=True,
        )
        if filtered_df.empty:
            st.info("No trades match the selected filters.")
            return
        metric_label = st.radio(
            "Rank by",
            ["Par Amount", "Trade Count"],
            horizontal=True,
            label_visibility="collapsed",
        )
        grouped = (
            filtered_df.groupby(["workbench_maturity_label", "trade_size_bucket"], dropna=False)
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
        grouped = grouped[grouped["workbench_maturity_label"].isin(MATURITY_YEAR_OPTIONS)]
        grouped = grouped[grouped["trade_size_bucket"].isin([x for x in TRADE_SIZE_BUCKETS if x != "All"])]
        if grouped.empty:
            st.info("No known maturity / trade-size buckets match the selected filters.")
            return
        metric_col = "par_amount" if metric_label == "Par Amount" else "trade_count"
        plot_col = "plot_value"
        grouped["band_label"] = grouped["workbench_maturity_label"].astype(str) + " / " + grouped["trade_size_bucket"].astype(str)
        grouped[plot_col] = grouped[metric_col] / 1_000_000 if metric_col == "par_amount" else grouped[metric_col]
        grouped["display_value"] = np.where(
            metric_col == "par_amount",
            "$" + (grouped["par_amount"] / 1_000_000).round(1).astype(str) + "MM",
            grouped["trade_count"].round(0).astype(int).astype(str),
        )
        chart_df = grouped.sort_values(metric_col, ascending=False).head(10).sort_values(metric_col, ascending=True)
        fig = px.bar(
            chart_df,
            x=plot_col,
            y="band_label",
            orientation="h",
            color="workbench_maturity_label",
            text="display_value",
            hover_data={
                "workbench_maturity_label": True,
                "trade_size_bucket": True,
                "par_amount": ":,.0f",
                "trade_count": ":,",
                "average_yield": ":.3f",
                "average_spread_bps": ":.1f",
                "display_value": False,
                plot_col: False,
            },
            labels={
                plot_col: "Par Amount ($MM)" if metric_col == "par_amount" else "Trade Count",
                "band_label": "",
                "workbench_maturity_label": "Maturity",
                "trade_size_bucket": "Trade Size Bucket",
                "par_amount": "Par Amount",
                "trade_count": "Trade Count",
                "average_yield": "Average Yield",
                "average_spread_bps": "Average Spread",
            },
        )
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(
            height=430,
            showlegend=False,
            xaxis_title="Par Amount ($MM)" if metric_col == "par_amount" else "Trade Count",
            yaxis_title="",
            yaxis=dict(categoryorder="array", categoryarray=chart_df["band_label"].tolist(), automargin=True),
            margin=dict(l=8, r=82, t=16, b=44),
        )
        fig.update_xaxes(tickangle=0)
        safe_plotly_chart(fig, width="stretch")


def _render_participation(filtered_df: pd.DataFrame):
    with st.container(border=True):
        st.markdown("<div class='workbench-block-title'>Dealer vs Customer</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='workbench-block-caption'>Shows who is driving traded par inside the active filter scope.</div>",
            unsafe_allow_html=True,
        )
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
        top_participant = part.iloc[0] if not part.empty else None
        if top_participant is not None:
            m1, m2 = st.columns(2)
            with m1:
                clean_metric_card("Top Participant", str(top_participant["participant_group"]), size="small")
            with m2:
                clean_metric_card("Share of Par", _fmt_pct(float(top_participant["par_share"])), size="small")

        fig = px.pie(
            part,
            values="par_traded",
            names="participant_group",
            hole=0.58,
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_traces(textinfo="percent", textposition="inside", hovertemplate="%{label}<br>%{percent}<br>%{value:,.0f}<extra></extra>")
        fig.update_layout(
            height=285,
            showlegend=False,
            margin=dict(l=8, r=8, t=8, b=8),
        )
        safe_plotly_chart(fig, width="stretch")

        table = part.assign(
            **{
                "Par Traded": part["par_traded"].map(_fmt_mm),
                "Par Share": part["par_share"].map(_fmt_pct),
            }
        )
        table = table.rename(columns={"participant_group": "Participant", "trade_count": "Trades"})
        safe_dataframe(
            table[["Participant", "Par Traded", "Trades", "Par Share"]],
            hide_index=True,
            auto_collapse=False,
            width="stretch",
        )


def _render_liquidity_dashboard(filtered_df: pd.DataFrame):
    st.subheader("Liquidity Dashboard")
    if filtered_df.empty:
        st.info("No trades match the selected filters.")
        return
    max_date = pd.to_datetime(filtered_df["trade_date"], errors="coerce").max()
    grouped = (
        filtered_df.groupby(["workbench_maturity_label", "trade_size_bucket"], dropna=False)
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
    grouped = grouped[grouped["workbench_maturity_label"].isin(MATURITY_YEAR_OPTIONS)]
    grouped = grouped[grouped["trade_size_bucket"].isin([x for x in TRADE_SIZE_BUCKETS if x != "All"])]
    if grouped.empty:
        st.info("No liquidity bands match the selected filters.")
        return
    grouped["bucket_label"] = grouped["workbench_maturity_label"].astype(str) + " / " + grouped["trade_size_bucket"].astype(str)
    grouped = grouped.sort_values(metric_col, ascending=(metric_col == "days_since_last_trade"))
    fig = px.bar(
        grouped.head(18),
        x=metric_col,
        y="bucket_label",
        orientation="h",
        color="workbench_maturity_label",
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
            "workbench_maturity_label": "Maturity Year",
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
    maturity_options = ["Current filters"] + sorted(
        [x for x in filtered_df["workbench_maturity_label"].dropna().unique().tolist() if x != "Unknown"],
        key=lambda label: _parse_maturity_year(label),
    )
    size_options = ["Current filters"] + sorted([x for x in filtered_df["trade_size_bucket"].dropna().unique().tolist() if x != "Unknown"])
    with c1:
        drill_maturity = st.selectbox("Drilldown maturity", maturity_options)
    with c2:
        drill_size = st.selectbox("Drilldown trade size", size_options)

    drill_df = filtered_df.copy()
    if drill_maturity != "Current filters":
        drill_df = drill_df[drill_df["workbench_maturity_label"] == drill_maturity]
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
        cusip_options = detail["cusip"].astype(str).tolist()
        current_cusip = str(st.session_state.get("workbench_selected_cusip") or "")
        default_idx = cusip_options.index(current_cusip) if current_cusip in cusip_options else 0
        if current_cusip in cusip_options and st.session_state.get("workbench_security_path_select") != current_cusip:
            st.session_state["workbench_security_path_select"] = current_cusip
        elif st.session_state.get("workbench_security_path_select") not in cusip_options:
            st.session_state["workbench_security_path_select"] = cusip_options[default_idx]
        selected_cusip = st.selectbox("CUSIP path", cusip_options, index=default_idx, key="workbench_security_path_select")
        st.session_state["workbench_selected_cusip"] = selected_cusip
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
    top_maturity = _top_bucket(filtered_df, "workbench_maturity_label", "trade_count")
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

    long_years = pd.to_numeric(filtered_df.get("workbench_maturity_year"), errors="coerce")
    long_end = filtered_df[long_years >= 20] if not filtered_df.empty else pd.DataFrame()
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
        ["Coupon", _filter_summary_label(selection.coupon_values, "All")],
        ["CUSIP", _filter_summary_label(selection.cusips, "All")],
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

    _render_inspector_css()
    _render_workbench_command_bar(prepared)

    section_anchor("workbench-issuer-selection", "1. Workbench Lens")
    st.markdown(
        "<div class='focus-band'><b>Lens:</b> one object and one filter scope drive every chart, table, CUSIP drilldown, RV view, and export below.</div>",
        unsafe_allow_html=True,
    )
    sector_options = ["All"] + sorted([x for x in prepared["sector"].dropna().astype(str).unique().tolist() if x and x != "nan"])
    all_issuer_options = sorted(prepared["issuer"].dropna().astype(str).unique().tolist())
    desired_sector = st.session_state.get("workbench_selected_sector", "All")
    desired_issuer = st.session_state.get("workbench_selected_issuer")
    if desired_sector not in sector_options:
        st.session_state["workbench_selected_sector"] = "All"
        desired_sector = "All"
    if desired_issuer in all_issuer_options and desired_sector != "All":
        issuers_in_sector = set(prepared.loc[prepared["sector"].astype(str) == str(desired_sector), "issuer"].astype(str))
        if str(desired_issuer) not in issuers_in_sector:
            st.session_state["workbench_selected_sector"] = "All"
            desired_sector = "All"

    c1, c2, c3 = st.columns([0.28, 0.34, 0.38])
    with c1:
        selected_sector = st.selectbox(
            "Sector",
            sector_options,
            index=sector_options.index(desired_sector),
            key="workbench_selected_sector",
        )
    issuer_pool = prepared if selected_sector == "All" else prepared[prepared["sector"].astype(str) == selected_sector]
    issuer_options = sorted(issuer_pool["issuer"].dropna().astype(str).unique().tolist())
    if not issuer_options:
        issuer_options = all_issuer_options
    if st.session_state.get("workbench_selected_issuer") not in issuer_options:
        st.session_state["workbench_selected_issuer"] = issuer_options[0] if issuer_options else ""
    with c2:
        selected_issuer = st.selectbox(
            "Issuer",
            issuer_options,
            index=issuer_options.index(st.session_state.get("workbench_selected_issuer")) if st.session_state.get("workbench_selected_issuer") in issuer_options else 0,
            key="workbench_selected_issuer",
        )
    issuer_base = prepared[prepared["issuer"].astype(str) == selected_issuer].copy()
    current_cusip = st.session_state.get("workbench_selected_cusip")
    if current_cusip and "cusip" in issuer_base.columns:
        if not issuer_base["cusip"].dropna().astype(str).eq(str(current_cusip)).any():
            st.session_state["workbench_selected_cusip"] = ""
    with c3:
        desired_date = st.session_state.get("workbench_date_range_label", "1 Year")
        if desired_date not in DATE_RANGE_OPTIONS:
            desired_date = "1 Year"
            st.session_state["workbench_date_range_label"] = desired_date
        date_option = st.selectbox(
            "Date Range",
            DATE_RANGE_OPTIONS,
            index=DATE_RANGE_OPTIONS.index(desired_date),
            key="workbench_date_range_label",
        )
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
    st.caption("Set one or more filters, then apply them together to avoid reloading after every checkbox.")
    coupon_available = _has_filter_values(issuer_base, "workbench_coupon")
    cusip_available = _has_filter_values(issuer_base, "cusip")
    with st.form("workbench_trading_filter_form", border=False):
        mf1, mf2, mf3 = st.columns([1, 1, 1.25])
        with mf1:
            current_year_labels = [
                label for label in st.session_state.get("workbench_maturity_years", []) if label in MATURITY_YEAR_OPTIONS
            ]
            if st.session_state.get("workbench_maturity_years") != current_year_labels:
                st.session_state["workbench_maturity_years"] = current_year_labels
            maturity_year_labels = tuple(
                st.multiselect(
                    "Maturity Years",
                    MATURITY_YEAR_OPTIONS,
                    key="workbench_maturity_years",
                    placeholder="All maturity years",
                    help="Leave empty to include all maturity years.",
                )
            )
            maturity_years = tuple(int(str(label).replace("Y", "")) for label in maturity_year_labels)
            maturity_bucket = _filter_summary_label(tuple(maturity_year_labels), "All")

        with mf2:
            coupon_options, coupon_limited = _search_filter_options(
                issuer_base,
                "workbench_coupon",
                "",
                sort_key=_coupon_sort_key,
            )
            current_coupons = _filter_existing_values(
                issuer_base,
                "workbench_coupon",
                st.session_state.get("workbench_coupon_values", []),
            )
            coupon_options = list(dict.fromkeys(current_coupons + coupon_options))
            if st.session_state.get("workbench_coupon_values") != current_coupons:
                st.session_state["workbench_coupon_values"] = current_coupons
            coupon_values: tuple[str, ...] = ()
            if coupon_available:
                coupon_values = tuple(
                    st.multiselect(
                        "Coupons",
                        coupon_options,
                        key="workbench_coupon_values",
                        placeholder="All coupons",
                        help="Leave empty to include all coupon values.",
                    )
                )
                if coupon_limited:
                    st.caption(f"Showing first {MAX_DYNAMIC_FILTER_OPTIONS:,} coupon values.")
            else:
                st.multiselect("Coupons", [], key="workbench_coupon_values_empty", placeholder="No coupons", disabled=True)

        with mf3:
            cusip_search = st.text_input(
                "Search CUSIP",
                key="workbench_cusip_search",
                placeholder="Type at least 2 characters",
                disabled=not cusip_available,
            )
            current_cusips = _filter_existing_values(
                issuer_base,
                "cusip",
                st.session_state.get("workbench_cusips", []),
            )
            if len(str(cusip_search or "").strip()) >= 2:
                cusip_matches, cusip_limited = _search_filter_options(issuer_base, "cusip", cusip_search)
            else:
                cusip_matches, cusip_limited = [], False
            cusip_options = list(dict.fromkeys(current_cusips + cusip_matches))
            if st.session_state.get("workbench_cusips") != current_cusips:
                st.session_state["workbench_cusips"] = current_cusips
            cusips: tuple[str, ...] = ()
            if not cusip_available:
                st.multiselect("CUSIPs", [], key="workbench_cusips_empty", placeholder="No CUSIPs", disabled=True)
            else:
                cusips = tuple(
                    st.multiselect(
                        "CUSIPs",
                        cusip_options,
                        key="workbench_cusips",
                        placeholder="All CUSIPs",
                        help="Leave empty to include all CUSIPs. Search first to load matching options.",
                    )
                )
                if not cusip_options:
                    st.caption("Search at least 2 CUSIP characters to load dropdown options.")
                elif cusip_limited:
                    st.caption(f"Showing first {MAX_DYNAMIC_FILTER_OPTIONS:,} CUSIP matches. Type more to narrow.")

        f1, f2, f3 = st.columns(3)
        with f1:
            if st.session_state.get("workbench_trade_size_bucket") not in TRADE_SIZE_BUCKETS:
                st.session_state["workbench_trade_size_bucket"] = "All"
            trade_size_bucket = st.selectbox("Trade Size Filters", TRADE_SIZE_BUCKETS, key="workbench_trade_size_bucket")
        with f2:
            observed_types = [x for x in TRADE_TYPE_BUCKETS if x == "All" or x in issuer_base["trade_type_bucket"].unique()]
            trade_type_options = observed_types or TRADE_TYPE_BUCKETS
            if st.session_state.get("workbench_trade_type_bucket") not in trade_type_options:
                st.session_state["workbench_trade_type_bucket"] = "All"
            trade_type_bucket = st.selectbox("Trade Type Filters", trade_type_options, key="workbench_trade_type_bucket")
        with f3:
            if st.session_state.get("workbench_lot_bucket") not in LOT_BUCKETS:
                st.session_state["workbench_lot_bucket"] = "All"
            lot_bucket = st.selectbox("Lot / Block Filter", LOT_BUCKETS, key="workbench_lot_bucket")

        apply_col, note_col = st.columns([0.18, 0.82])
        with apply_col:
            st.form_submit_button("Apply filters", type="primary")
        with note_col:
            st.caption("Empty multi-select means All. For CUSIP, search first to load matching dropdown options.")

    selection = WorkbenchSelection(
        sector=selected_sector,
        issuer=selected_issuer,
        date_range_label=date_option,
        date_range=date_range,
        maturity_bucket=maturity_bucket,
        trade_size_bucket=trade_size_bucket,
        trade_type_bucket=trade_type_bucket,
        lot_bucket=lot_bucket,
        maturity_years=maturity_years,
        coupon_values=coupon_values,
        cusips=cusips,
    )
    filtered_universe = _apply_workbench_filters(prepared, selection, issuer=None)
    filtered_issuer = _apply_workbench_filters(prepared, selection, issuer=selected_issuer)
    cusip_summary = _build_cusip_summary(filtered_issuer)
    selected_cusip = _best_cusip_for_profile(filtered_issuer, cusip_summary)
    if selected_cusip:
        st.session_state["workbench_selected_cusip"] = selected_cusip

    _render_object_status_bar(selection, filtered_issuer, benchmark_source_mode)
    _active_filter_summary(selection)
    _render_summary_cards(filtered_issuer)

    section_anchor("workbench-market-analytics", "3. Market Analytics")
    st.caption(f"Benchmark source retained for spread calculations: {benchmark_source_mode}. Trading analysis is driven by the uploaded trade tape.")
    _render_volume_overview(filtered_issuer)
    a1, a2 = st.columns([0.58, 0.42], gap="large")
    with a1:
        _render_activity_concentration_map(filtered_issuer)
    with a2:
        _render_participation(filtered_issuer)
    _render_liquidity_dashboard(filtered_issuer)

    security_detail = _render_security_drilldown(filtered_issuer)
    cusip_summary = _build_cusip_summary(filtered_issuer)
    peer_metrics, peer_issuers = _render_peer_comparison(prepared, selection, filtered_issuer)

    section_anchor("workbench-narrative-insights", "5B. Narrative Read-Through")
    observations = _build_narrative(filtered_issuer, security_detail, selection)
    for obs in observations:
        st.markdown(f"<div class='methodology-note'>{_html_escape(obs)}</div>", unsafe_allow_html=True)

    selected_cusip = _best_cusip_for_profile(filtered_issuer, cusip_summary)
    issuer_profile = _build_issuer_profile(filtered_issuer, selection, benchmark_source_mode)
    with st.sidebar:
        selected_cusip, security_profile = _render_right_inspector(selection, issuer_profile, filtered_issuer, cusip_summary)

    return {
        "selection": selection,
        "prepared_df": prepared,
        "filtered_universe_df": filtered_universe,
        "filtered_issuer_df": filtered_issuer,
        "security_detail_df": security_detail,
        "cusip_summary_df": cusip_summary,
        "issuer_profile": issuer_profile,
        "security_profile": security_profile,
        "peer_metrics_df": peer_metrics,
        "observations": observations,
        "selected_issuer": selected_issuer,
        "selected_cusip": selected_cusip,
        "selected_sector": selected_sector,
        "peer_issuers": peer_issuers,
        "filtered_rows": len(filtered_issuer),
        "security_rows": len(security_detail),
        "peer_rows": len(peer_metrics),
    }
