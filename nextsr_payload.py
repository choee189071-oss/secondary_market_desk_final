from __future__ import annotations

import math
import re
from typing import Any

import pandas as pd


MAX_MATURITY_YEAR = 40
MATURITY_BUCKET_ORDER = [f"{year}Y" for year in range(1, MAX_MATURITY_YEAR + 1)]


def _json_value(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _round_or_none(value: Any, digits: int = 2) -> float | None:
    value = _json_value(value)
    if value is None:
        return None
    try:
        return round(float(value), digits)
    except Exception:
        return None


def _nearest_benchmark_tenor(year: int) -> str:
    if year <= 2:
        return f"{year}Y"
    if year <= 7:
        return "5Y"
    if year <= 15:
        return "10Y"
    if year <= 25:
        return "20Y"
    return "30Y"


def _parse_trade_index_tenor(index_value: object) -> str | None:
    if pd.isna(index_value):
        return None
    text = str(index_value).upper().strip()
    match = re.search(r"(\d{1,2})\s*Y?$", text)
    if not match:
        return None
    year = int(match.group(1))
    if year < 1:
        return None
    return f"{min(year, MAX_MATURITY_YEAR)}Y"


def prepare_market_frame(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Return a model-ready trade frame for nextsr payload generation."""
    if trades_df is None or trades_df.empty:
        return pd.DataFrame()

    out = trades_df.copy()
    for col in ["trade_date", "trade_datetime", "maturity"]:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")
    for col in ["yield", "price", "trade_amount", "index_rate", "spread", "coupon"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if "issuer" not in out.columns:
        if "source_issuer_guess" in out.columns:
            out["issuer"] = out["source_issuer_guess"]
        elif "source_file" in out.columns:
            out["issuer"] = out["source_file"].astype(str).str.replace(r"\.[^.]+$", "", regex=True)
        else:
            out["issuer"] = "Unknown"
    out["issuer"] = out["issuer"].fillna("Unknown").astype(str).str.strip()

    if {"maturity", "trade_date"}.issubset(out.columns):
        out["years_to_maturity"] = (out["maturity"] - out["trade_date"]).dt.days / 365.25
        years = pd.to_numeric(out["years_to_maturity"], errors="coerce")
        out["maturity_year"] = years.apply(lambda v: math.ceil(v) if pd.notna(v) else pd.NA)
        out["maturity_bucket"] = out["maturity_year"].apply(
            lambda v: f"{int(v)}Y" if pd.notna(v) and 1 <= int(v) <= MAX_MATURITY_YEAR else pd.NA
        )

    return out


def build_trade_index_curve(market_df: pd.DataFrame) -> pd.DataFrame:
    """Build a daily benchmark curve from trade tape index/index_rate columns."""
    required = {"trade_date", "index", "index_rate"}
    if market_df is None or market_df.empty or not required.issubset(market_df.columns):
        return pd.DataFrame()

    tmp = market_df[list(required)].copy()
    tmp["trade_date"] = pd.to_datetime(tmp["trade_date"], errors="coerce").dt.normalize()
    tmp["index_rate"] = pd.to_numeric(tmp["index_rate"], errors="coerce")
    tmp["tenor"] = tmp["index"].apply(_parse_trade_index_tenor)
    tmp = tmp.dropna(subset=["trade_date", "index_rate", "tenor"])
    if tmp.empty:
        return pd.DataFrame()

    curve = tmp.groupby(["trade_date", "tenor"], as_index=False)["index_rate"].median()
    curve = curve.rename(columns={"trade_date": "date", "index_rate": "benchmark_yield"})
    curve["benchmark_source"] = "Trade Sheet Index / Index Rate"
    return curve.sort_values(["date", "tenor"]).reset_index(drop=True)


def build_spread_observations(market_df: pd.DataFrame, benchmark_curve: pd.DataFrame) -> pd.DataFrame:
    """Build issuer/date/maturity spread observations in basis points."""
    required_market = {"issuer", "trade_date", "maturity_bucket", "yield"}
    required_curve = {"date", "tenor", "benchmark_yield"}
    if (
        market_df is None
        or benchmark_curve is None
        or market_df.empty
        or benchmark_curve.empty
        or not required_market.issubset(market_df.columns)
        or not required_curve.issubset(benchmark_curve.columns)
    ):
        return pd.DataFrame()

    work = market_df.copy()
    work["trade_date"] = pd.to_datetime(work["trade_date"], errors="coerce").dt.normalize()
    work["yield"] = pd.to_numeric(work["yield"], errors="coerce")
    work = work.dropna(subset=["issuer", "trade_date", "maturity_bucket", "yield"])
    work = work[work["maturity_bucket"].isin(MATURITY_BUCKET_ORDER)]
    if work.empty:
        return pd.DataFrame()

    daily = (
        work.groupby(["issuer", "trade_date", "maturity_bucket"], as_index=False)
        .agg(
            avg_yield=("yield", "mean"),
            trade_count=("yield", "count"),
            total_trade_amount=("trade_amount", "sum") if "trade_amount" in work.columns else ("yield", "count"),
        )
    )
    daily["tenor"] = daily["maturity_bucket"].str.replace("Y", "", regex=False).astype(int).apply(_nearest_benchmark_tenor)

    curve = benchmark_curve.copy()
    curve["date"] = pd.to_datetime(curve["date"], errors="coerce").dt.normalize()
    merged = daily.merge(curve, left_on=["trade_date", "tenor"], right_on=["date", "tenor"], how="inner")
    if merged.empty:
        return pd.DataFrame()

    merged["spread_to_benchmark_bps"] = (merged["avg_yield"] - merged["benchmark_yield"]) * 100
    return merged.sort_values(["issuer", "maturity_bucket", "trade_date"]).reset_index(drop=True)


def _liquidity_signal(trades: pd.DataFrame, latest_date: pd.Timestamp, period_days: int) -> dict[str, Any]:
    if trades.empty:
        return {"liquidity_score": None, "trade_count": 0, "total_trade_amount": 0.0, "days_since_last_trade": None}
    work = trades.copy()
    work["trade_date"] = pd.to_datetime(work["trade_date"], errors="coerce").dt.normalize()
    if "trade_amount" in work.columns:
        work["trade_amount"] = pd.to_numeric(work["trade_amount"], errors="coerce").fillna(0.0)
    else:
        work["trade_amount"] = 0.0
    work = work.dropna(subset=["trade_date"])
    if work.empty:
        return {"liquidity_score": None, "trade_count": 0, "total_trade_amount": 0.0, "days_since_last_trade": None}

    latest_trade_date = work["trade_date"].max()
    window = work[work["trade_date"] >= latest_date - pd.Timedelta(days=int(period_days))]
    trade_count = int(len(window))
    total_amount = float(window["trade_amount"].sum()) if not window.empty else 0.0
    days_since_last = max(0, int((latest_date - latest_trade_date).days))
    liquidity_score = (
        min(trade_count / 10, 1) * 35
        + min(total_amount / 5_000_000, 1) * 35
        + max(0, 1 - min(days_since_last / 180, 1)) * 30
    )
    return {
        "liquidity_score": round(float(liquidity_score), 1),
        "trade_count": trade_count,
        "total_trade_amount": round(total_amount, 0),
        "days_since_last_trade": days_since_last,
    }


def _flow_signal(trades: pd.DataFrame, latest_date: pd.Timestamp, period_days: int) -> dict[str, Any]:
    if trades.empty or "trade_type" not in trades.columns:
        return {"sell_buy_imbalance": None, "classified_buy_amount": 0.0, "classified_sell_amount": 0.0}

    work = trades.copy()
    work["trade_date"] = pd.to_datetime(work["trade_date"], errors="coerce").dt.normalize()
    work["trade_amount"] = pd.to_numeric(work.get("trade_amount", 0.0), errors="coerce").fillna(0.0)
    work = work[work["trade_date"] >= latest_date - pd.Timedelta(days=int(period_days))]

    def classify(value: object) -> str:
        text = str(value).strip().lower()
        if text in {"s", "sell", "sold", "sld", "customer sell", "cust sell", "cs"}:
            return "Sell"
        if text in {"b", "buy", "bought", "purchase", "customer buy", "cust buy", "cb"}:
            return "Buy"
        return "Other"

    work["flow_side"] = work["trade_type"].map(classify)
    buy_amount = float(work.loc[work["flow_side"] == "Buy", "trade_amount"].sum())
    sell_amount = float(work.loc[work["flow_side"] == "Sell", "trade_amount"].sum())
    denom = buy_amount + sell_amount
    imbalance = (sell_amount - buy_amount) / denom if denom > 0 else None
    return {
        "sell_buy_imbalance": _round_or_none(imbalance, 4),
        "classified_buy_amount": round(buy_amount, 0),
        "classified_sell_amount": round(sell_amount, 0),
    }


def _label_signal(spread_change: float | None, percentile: float | None, liquidity: float | None, flow_imbalance: float | None) -> str:
    score = 0.0
    if spread_change is not None and spread_change >= 15:
        score += 1
    if percentile is not None and percentile >= 75:
        score += 1
    if liquidity is not None and liquidity >= 60:
        score += 1
    if flow_imbalance is not None and flow_imbalance >= 0.25:
        score += 0.5

    if score >= 3:
        return "Potential Relative Value Candidate"
    if score >= 2:
        return "Watchlist Candidate"
    if score <= 0.5 and percentile is not None and percentile <= 25:
        return "Potentially Rich / Lower Priority"
    return "Neutral / Needs More Evidence"


def build_nextsr_payload(
    trades_df: pd.DataFrame,
    issuer: str | None = None,
    maturity_bucket: str | None = None,
    period_days: int = 30,
) -> dict[str, Any]:
    """Build the stable JSON-style payload intended for nextsr/nexjr input."""
    market = prepare_market_frame(trades_df)
    benchmark_curve = build_trade_index_curve(market)
    spread_obs = build_spread_observations(market, benchmark_curve)

    payload: dict[str, Any] = {
        "schema_version": "nextsr_payload.v1",
        "issuer": issuer,
        "maturity_bucket": maturity_bucket,
        "as_of_date": None,
        "universe": {
            "trade_rows": int(len(market)),
            "cusip_count": int(market["cusip"].nunique()) if "cusip" in market.columns else 0,
            "benchmark_source": "Trade Sheet Index / Index Rate" if not benchmark_curve.empty else None,
        },
        "signals": {
            "spread": {
                "current_spread_bps": None,
                "spread_change_bps": None,
                "historical_percentile_1y": None,
                "latest_spread_date": None,
            },
            "liquidity": {
                "liquidity_score": None,
                "trade_count": 0,
                "total_trade_amount": 0.0,
                "days_since_last_trade": None,
            },
            "flow": {
                "sell_buy_imbalance": None,
                "classified_buy_amount": 0.0,
                "classified_sell_amount": 0.0,
            },
        },
        "label": "Neutral / Needs More Evidence",
        "evidence": [],
    }

    if market.empty:
        payload["evidence"].append("No model-ready trade rows were available.")
        return payload

    if issuer is None:
        issuer = str(market["issuer"].dropna().mode().iloc[0]) if market["issuer"].notna().any() else "Unknown"
        payload["issuer"] = issuer

    issuer_market = market[market["issuer"].astype(str) == str(issuer)].copy()
    if maturity_bucket is None and not issuer_market.empty and "maturity_bucket" in issuer_market.columns:
        bucket_counts = issuer_market["maturity_bucket"].dropna().value_counts()
        maturity_bucket = str(bucket_counts.index[0]) if not bucket_counts.empty else None
        payload["maturity_bucket"] = maturity_bucket

    if spread_obs.empty:
        payload["evidence"].append("No overlapping trade/index-rate observations were available for spread signals.")
        return payload

    obs = spread_obs[spread_obs["issuer"].astype(str) == str(issuer)].copy()
    if maturity_bucket is not None:
        obs = obs[obs["maturity_bucket"].astype(str) == str(maturity_bucket)]
    obs = obs.dropna(subset=["trade_date", "spread_to_benchmark_bps"]).sort_values("trade_date")
    if obs.empty:
        payload["evidence"].append("No spread observations matched the selected issuer and maturity bucket.")
        return payload

    latest = obs.iloc[-1]
    latest_date = pd.to_datetime(latest["trade_date"]).normalize()
    current_spread = float(latest["spread_to_benchmark_bps"])
    target_date = latest_date - pd.Timedelta(days=int(period_days))
    hist_candidates = obs[obs["trade_date"] <= target_date]
    spread_change = None
    if not hist_candidates.empty:
        spread_change = current_spread - float(hist_candidates.iloc[-1]["spread_to_benchmark_bps"])

    one_year = obs[obs["trade_date"] >= latest_date - pd.Timedelta(days=365)]
    hist_values = pd.to_numeric(one_year["spread_to_benchmark_bps"], errors="coerce").dropna()
    percentile = float((hist_values <= current_spread).mean() * 100) if len(hist_values) >= 2 else None

    payload["as_of_date"] = latest_date.strftime("%Y-%m-%d")
    payload["signals"]["spread"] = {
        "current_spread_bps": _round_or_none(current_spread, 2),
        "spread_change_bps": _round_or_none(spread_change, 2),
        "historical_percentile_1y": _round_or_none(percentile, 1),
        "latest_spread_date": latest_date.strftime("%Y-%m-%d"),
    }

    selected_trades = issuer_market.copy()
    if maturity_bucket is not None and "maturity_bucket" in selected_trades.columns:
        selected_trades = selected_trades[selected_trades["maturity_bucket"].astype(str) == str(maturity_bucket)]
    payload["signals"]["liquidity"] = _liquidity_signal(selected_trades, latest_date, period_days)
    payload["signals"]["flow"] = _flow_signal(selected_trades, latest_date, period_days)

    payload["label"] = _label_signal(
        payload["signals"]["spread"]["spread_change_bps"],
        payload["signals"]["spread"]["historical_percentile_1y"],
        payload["signals"]["liquidity"]["liquidity_score"],
        payload["signals"]["flow"]["sell_buy_imbalance"],
    )

    payload["evidence"].append(
        f"Current spread is {payload['signals']['spread']['current_spread_bps']:+.1f} bps "
        f"as of {payload['signals']['spread']['latest_spread_date']}."
    )
    if payload["signals"]["spread"]["spread_change_bps"] is not None:
        payload["evidence"].append(
            f"{period_days}-day spread movement is {payload['signals']['spread']['spread_change_bps']:+.1f} bps."
        )
    if payload["signals"]["liquidity"]["liquidity_score"] is not None:
        payload["evidence"].append(
            f"Liquidity score is {payload['signals']['liquidity']['liquidity_score']:.1f} "
            f"from {payload['signals']['liquidity']['trade_count']} trade(s) in the selected window."
        )

    return payload
