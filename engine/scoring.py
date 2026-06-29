from __future__ import annotations

import numpy as np
import pandas as pd


def add_workflow_spread_bps(df: pd.DataFrame) -> pd.DataFrame:
    """Add a best-effort spread_bps column for focused workflow charts."""
    out = df.copy()
    if out.empty:
        out["spread_bps"] = pd.Series(dtype="float64")
        return out

    if "spread_bps" in out.columns:
        out["spread_bps"] = pd.to_numeric(out["spread_bps"], errors="coerce")
        return out

    if "spread" in out.columns and pd.to_numeric(out["spread"], errors="coerce").notna().any():
        raw = pd.to_numeric(out["spread"], errors="coerce")
        median_abs = raw.abs().dropna().median()
        out["spread_bps"] = raw * 100 if pd.notna(median_abs) and median_abs <= 10 else raw
    elif {"yield", "index_rate"}.issubset(out.columns):
        out["spread_bps"] = (
            pd.to_numeric(out["yield"], errors="coerce")
            - pd.to_numeric(out["index_rate"], errors="coerce")
        ) * 100
    else:
        out["spread_bps"] = pd.NA
    return out


def workflow_date_range_text(df: pd.DataFrame) -> str:
    if df.empty or "trade_date" not in df.columns:
        return "No dates"
    dates = pd.to_datetime(df["trade_date"], errors="coerce").dropna()
    if dates.empty:
        return "No dates"
    return f"{dates.min():%m/%d/%Y} - {dates.max():%m/%d/%Y}"


def build_workflow_cusip_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Small deterministic CUSIP summary used by focused pages."""
    if df.empty or "cusip" not in df.columns:
        return pd.DataFrame()

    base = add_workflow_spread_bps(df.copy())
    if "trade_date" in base.columns:
        base["trade_date"] = pd.to_datetime(base["trade_date"], errors="coerce")
    if "trade_amount" in base.columns:
        base["trade_amount"] = pd.to_numeric(base["trade_amount"], errors="coerce").fillna(0)
    else:
        base["trade_amount"] = 0
    if "yield" in base.columns:
        base["yield"] = pd.to_numeric(base["yield"], errors="coerce")
    if "price" in base.columns:
        base["price"] = pd.to_numeric(base["price"], errors="coerce")

    agg_spec = {
        "trade_count": ("trade_date", "count") if "trade_date" in base.columns else ("cusip", "count"),
        "total_trade_amount": ("trade_amount", "sum"),
        "latest_trade": ("trade_date", "max") if "trade_date" in base.columns else ("cusip", "count"),
        "current_spread_bps": ("spread_bps", "median"),
    }
    if "yield" in base.columns:
        agg_spec["avg_yield"] = ("yield", "mean")
    if "price" in base.columns:
        agg_spec["avg_price"] = ("price", "mean")
    if "maturity_bucket" in base.columns:
        agg_spec["maturity_bucket"] = ("maturity_bucket", "first")

    summary = base.groupby("cusip", dropna=False, observed=True).agg(**agg_spec).reset_index()
    if "latest_trade" in summary.columns:
        latest_dates = pd.to_datetime(summary["latest_trade"], errors="coerce")
        latest_anchor = pd.to_datetime(base.get("trade_date"), errors="coerce").max() if "trade_date" in base.columns else pd.Timestamp.today()
        summary["days_since_last_trade"] = (latest_anchor - latest_dates).dt.days
    else:
        summary["days_since_last_trade"] = pd.NA

    spread_rank = pd.to_numeric(summary["current_spread_bps"], errors="coerce").rank(pct=True)
    count_rank = pd.to_numeric(summary["trade_count"], errors="coerce").rank(pct=True)
    amount_rank = pd.to_numeric(summary["total_trade_amount"], errors="coerce").rank(pct=True)
    recency_rank = (1 - pd.to_numeric(summary["days_since_last_trade"], errors="coerce").rank(pct=True)).fillna(0.5)
    summary["liquidity_score"] = (count_rank * 35 + amount_rank * 35 + recency_rank * 30).round(1)
    summary["rv_score"] = (spread_rank.fillna(0.5) * 55 + summary["liquidity_score"].rank(pct=True).fillna(0.5) * 45).round(1)
    summary["signal"] = np.select(
        [
            (summary["rv_score"] >= 70) & (summary["liquidity_score"] >= 55),
            summary["current_spread_bps"].fillna(-999) >= summary["current_spread_bps"].quantile(0.75),
            summary["liquidity_score"] >= 70,
        ],
        ["Wide + Liquid", "Wide / Review", "Liquid / Monitor"],
        default="Monitor",
    )
    return summary.sort_values(["rv_score", "liquidity_score", "trade_count"], ascending=False)


def focused_trade_side(value: object) -> str:
    """Best-effort buy/sell side classifier for focused CUSIP flow views."""
    text_val = str(value).strip().lower()
    if not text_val or text_val in {"nan", "none"}:
        return "Unknown"
    if any(token in text_val for token in ["sell", "sold", "sld", "customer sell", "cust sell", "cs"]):
        return "Sell"
    if any(token in text_val for token in ["buy", "bought", "purchase", "customer buy", "cust buy", "cb"]):
        return "Buy"
    if text_val == "s":
        return "Sell"
    if text_val == "b":
        return "Buy"
    return "Other / Unknown"


def focused_summary_with_peer_gaps(summary: pd.DataFrame) -> pd.DataFrame:
    """Add same-bucket peer spread gaps used by RV, watchlist, and report output."""
    if summary is None or summary.empty:
        return pd.DataFrame()
    out = summary.copy()
    if "maturity_bucket" in out.columns and "current_spread_bps" in out.columns:
        out["peer_median_spread_bps"] = out.groupby("maturity_bucket", observed=True)["current_spread_bps"].transform("median")
        out["peer_median_gap_bps"] = (
            pd.to_numeric(out["current_spread_bps"], errors="coerce")
            - pd.to_numeric(out["peer_median_spread_bps"], errors="coerce")
        )
    elif "peer_median_gap_bps" not in out.columns:
        out["peer_median_gap_bps"] = pd.NA
    return out
