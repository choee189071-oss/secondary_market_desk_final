from __future__ import annotations

import re

import pandas as pd


try:
    import streamlit as st
except Exception:  # pragma: no cover - lets pure engine tests import without Streamlit installed.
    class _StreamlitShim:
        @staticmethod
        def cache_data(*_args, **_kwargs):
            def decorator(func):
                return func
            return decorator

    st = _StreamlitShim()


RATING_SPREADS: dict[str, dict[str, float]] = {
    "AAA": {"5Y": 0.00, "10Y": 0.00, "20Y": 0.00, "30Y": 0.00},
    "AA+": {"5Y": 0.08, "10Y": 0.10, "20Y": 0.12, "30Y": 0.15},
    "AA": {"5Y": 0.10, "10Y": 0.14, "20Y": 0.17, "30Y": 0.20},
    "AA-": {"5Y": 0.14, "10Y": 0.18, "20Y": 0.22, "30Y": 0.28},
    "A+": {"5Y": 0.22, "10Y": 0.28, "20Y": 0.35, "30Y": 0.42},
    "A": {"5Y": 0.30, "10Y": 0.38, "20Y": 0.48, "30Y": 0.58},
    "A-": {"5Y": 0.42, "10Y": 0.55, "20Y": 0.68, "30Y": 0.82},
    "BBB": {"5Y": 0.60, "10Y": 0.80, "20Y": 1.00, "30Y": 1.20},
}

MAX_MATURITY_YEAR = 40
MATURITY_BUCKET_ORDER = [f"{y}Y" for y in range(1, MAX_MATURITY_YEAR + 1)]
MATURITY_BUCKET_OPTIONS = ["All"] + MATURITY_BUCKET_ORDER
MATURITY_BUCKET_RENAME = {
    "Short": "5Y",
    "Intermediate": "10Y",
    "Long": "20Y",
    "Extended Long": "30Y",
    "10Y": "10Y",
    "20Y": "20Y",
    "30Y": "30Y",
}


def maturity_year_sort_key(value: object) -> int:
    """Sort labels like 1Y, 10Y, 30Y numerically instead of alphabetically."""
    try:
        text = str(value).strip().upper().replace("Y", "")
        return int(float(text))
    except Exception:
        return 9999


def observed_maturity_years(
    df: pd.DataFrame,
    bucket_col: str = "maturity_bucket",
    min_observations: int = 1,
) -> list[str]:
    """Return only maturity years that actually exist in the data."""
    if df is None or df.empty or bucket_col not in df.columns:
        return []

    counts = (
        df.dropna(subset=[bucket_col])
        .assign(**{bucket_col: lambda x: x[bucket_col].astype(str)})
        .groupby(bucket_col)
        .size()
    )
    valid = [b for b, n in counts.items() if n >= min_observations and b in MATURITY_BUCKET_ORDER]
    return sorted(valid, key=maturity_year_sort_key)


def compact_maturity_table(table: pd.DataFrame) -> pd.DataFrame:
    """Drop all-empty maturity rows and sort annual maturity labels numerically."""
    if table is None or table.empty:
        return table
    out = table.dropna(how="all")
    if out.empty:
        return out
    return out.loc[sorted(out.index, key=maturity_year_sort_key)]


def nearest_benchmark_tenor(year: int) -> str:
    """Map an annual maturity year to the closest available benchmark tenor."""
    if year <= 2:
        return f"{year}Y"
    if year <= 7:
        return "5Y"
    if year <= 15:
        return "10Y"
    if year <= 25:
        return "20Y"
    return "30Y"


MMD_BUCKET_MAP = {f"{y}Y": nearest_benchmark_tenor(y) for y in range(1, MAX_MATURITY_YEAR + 1)}
MMD_BUCKET_MAP["All"] = "10Y"
BENCHMARK_RATINGS = list(RATING_SPREADS.keys())


def curve_column_key(name: object) -> str:
    """Normalize curve column names for flexible matching."""
    text = str(name).strip().lower()
    text = text.replace("+", " plus ").replace("-", " minus ")
    keep = []
    for ch in text:
        keep.append(ch if ch.isalnum() else " ")
    return " ".join("".join(keep).split()).replace(" ", "")


def rating_key(rating: str) -> str:
    return curve_column_key(rating)


def find_uploaded_benchmark_column(mmd_df: pd.DataFrame, tenor: str, rating: str) -> str | None:
    """Find an explicitly uploaded benchmark column for rating + tenor."""
    normalized = {curve_column_key(c): c for c in mmd_df.columns}
    r = rating_key(rating)
    t = curve_column_key(tenor)

    candidates = [
        f"{r}{t}",
        f"{r}curve{t}",
        f"{r}yield{t}",
        f"{r}muni{t}",
        f"{r}mmd{t}",
        f"{t}{r}",
    ]

    if rating == "AAA":
        candidates.extend([
            t,
            f"mmd{t}",
            f"mmdaaa{t}",
            f"aaammd{t}",
            f"aaacurve{t}",
        ])

    for key in candidates:
        if key in normalized:
            return normalized[key]
    return None


def get_benchmark_curve(mmd_plot: pd.DataFrame, tenor: str, rating: str) -> tuple[pd.Series, dict] | tuple[None, dict]:
    """Return benchmark yield and metadata."""
    explicit_col = find_uploaded_benchmark_column(mmd_plot, tenor, rating)
    if explicit_col is not None:
        return pd.to_numeric(mmd_plot[explicit_col], errors="coerce"), {
            "benchmark_source": "Uploaded curve",
            "source_column": explicit_col,
            "rating_spread_bps": 0.0,
        }

    base_col = find_uploaded_benchmark_column(mmd_plot, tenor, "AAA")
    if base_col is None:
        return None, {
            "benchmark_source": "Unavailable",
            "source_column": None,
            "rating_spread_bps": pd.NA,
        }

    base_curve = pd.to_numeric(mmd_plot[base_col], errors="coerce")
    spread_adjustment = RATING_SPREADS.get(rating, RATING_SPREADS["AAA"]).get(tenor, 0.00)
    return base_curve + spread_adjustment, {
        "benchmark_source": "Modeled from MMD + spread assumption" if rating != "AAA" else "Uploaded MMD / AAA curve",
        "source_column": base_col,
        "rating_spread_bps": spread_adjustment * 100,
    }


def benchmark_curve_from_mmd(mmd_plot: pd.DataFrame, mmd_col: str, rating: str) -> pd.Series:
    """Backward-compatible wrapper used by older chart blocks."""
    curve, _meta = get_benchmark_curve(mmd_plot, mmd_col, rating)
    if curve is None:
        return pd.Series([pd.NA] * len(mmd_plot), index=mmd_plot.index, dtype="float")
    return curve


def rating_spread_table() -> pd.DataFrame:
    """User-facing spread assumption table in both percentage points and bps."""
    rows = []
    for rating, tenors in RATING_SPREADS.items():
        row = {"Rating": rating}
        for tenor, spread_pct in tenors.items():
            row[f"{tenor} Spread"] = spread_pct
            row[f"{tenor} Spread (bps)"] = round(spread_pct * 100, 1)
        rows.append(row)
    return pd.DataFrame(rows)


def detect_mmd_date_column(mmd_df: pd.DataFrame) -> str | None:
    """Find the MMD date column across common naming variants."""
    if "Date" in mmd_df.columns:
        return "Date"
    if "date" in mmd_df.columns:
        return "date"
    return None


@st.cache_data(show_spinner=False, max_entries=32)
def make_benchmark_long(mmd_df: pd.DataFrame, rating: str) -> pd.DataFrame:
    """Convert MMD wide curve data into long benchmark data by maturity year."""
    if mmd_df.empty:
        return pd.DataFrame()

    date_col = detect_mmd_date_column(mmd_df)
    if date_col is None:
        return pd.DataFrame()

    frames = []
    mmd_base = mmd_df.copy()
    mmd_base[date_col] = pd.to_datetime(mmd_base[date_col], errors="coerce")
    mmd_base = mmd_base.dropna(subset=[date_col])

    for bucket, tenor in MMD_BUCKET_MAP.items():
        if bucket == "All":
            continue
        benchmark_yield, meta = get_benchmark_curve(mmd_base, tenor, rating)
        if benchmark_yield is None:
            continue
        frames.append(
            pd.DataFrame(
                {
                    "trade_date": mmd_base[date_col].dt.normalize(),
                    "maturity_bucket": bucket,
                    "benchmark_rating": rating,
                    "mmd_tenor": tenor,
                    "benchmark_yield": benchmark_yield,
                    "rating_spread_bps": meta.get("rating_spread_bps"),
                    "benchmark_source": meta.get("benchmark_source"),
                    "source_column": meta.get("source_column"),
                }
            )
        )

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


@st.cache_data(show_spinner=False, max_entries=32)
def build_spread_observations(
    market_df: pd.DataFrame,
    mmd_df: pd.DataFrame,
    issuer: str,
    rating: str,
) -> pd.DataFrame:
    """Build daily issuer spread observations by maturity year."""
    required_cols = {"issuer", "trade_date", "maturity_bucket", "yield"}
    if market_df.empty or mmd_df.empty or not required_cols.issubset(set(market_df.columns)):
        return pd.DataFrame()

    issuer_df = market_df[market_df["issuer"] == issuer].copy()
    issuer_df = issuer_df[issuer_df["maturity_bucket"].isin(MATURITY_BUCKET_ORDER)]
    if issuer_df.empty:
        return pd.DataFrame()

    issuer_df["trade_date"] = pd.to_datetime(issuer_df["trade_date"], errors="coerce").dt.normalize()
    issuer_df["yield"] = pd.to_numeric(issuer_df["yield"], errors="coerce")
    issuer_df = issuer_df.dropna(subset=["trade_date", "yield", "maturity_bucket"])

    daily_issuer = (
        issuer_df.groupby(["trade_date", "maturity_bucket"], as_index=False)
        .agg(
            avg_yield=("yield", "mean"),
            trade_count=("yield", "count"),
            total_trade_amount=("trade_amount", "sum") if "trade_amount" in issuer_df.columns else ("yield", "count"),
        )
    )

    benchmark_long = make_benchmark_long(mmd_df, rating)
    if benchmark_long.empty:
        return pd.DataFrame()

    spread_obs = daily_issuer.merge(
        benchmark_long,
        on=["trade_date", "maturity_bucket"],
        how="inner",
    )
    if spread_obs.empty:
        return pd.DataFrame()

    spread_obs["spread_to_benchmark_bps"] = (
        spread_obs["avg_yield"] - spread_obs["benchmark_yield"]
    ) * 100
    return spread_obs.sort_values(["maturity_bucket", "trade_date"])


@st.cache_data(show_spinner=False, max_entries=32)
def build_spread_movement_ladder_data(
    spread_obs: pd.DataFrame,
    windows: dict[str, int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ladder table and audit table for spread movement."""
    if windows is None:
        windows = {"1W": 7, "1M": 30, "3M": 90, "6M": 180, "1Y": 365}

    audit_rows = []

    if spread_obs.empty:
        return pd.DataFrame(columns=list(windows.keys()), dtype="float"), pd.DataFrame(audit_rows)

    obs = spread_obs.copy()
    obs["trade_date"] = pd.to_datetime(obs["trade_date"], errors="coerce").dt.normalize()
    obs = obs.dropna(subset=["trade_date", "spread_to_benchmark_bps", "maturity_bucket"])

    maturity_order = observed_maturity_years(obs, min_observations=1)
    table = pd.DataFrame(index=maturity_order, columns=list(windows.keys()), dtype="float")

    for bucket in maturity_order:
        bucket_obs = obs[obs["maturity_bucket"] == bucket].sort_values("trade_date")
        if bucket_obs.empty:
            continue

        latest_row = bucket_obs.iloc[-1]
        latest_date = latest_row["trade_date"]
        latest_spread = latest_row["spread_to_benchmark_bps"]

        for label, days in windows.items():
            target_date = latest_date - pd.Timedelta(days=days)
            historical_candidates = bucket_obs[bucket_obs["trade_date"] <= target_date]
            if historical_candidates.empty:
                audit_rows.append(
                    {
                        "maturity_bucket": bucket,
                        "window": label,
                        "latest_date": latest_date,
                        "latest_spread_bps": latest_spread,
                        "target_date": target_date,
                        "historical_date": pd.NaT,
                        "historical_spread_bps": pd.NA,
                        "spread_movement_bps": pd.NA,
                        "note": "No historical observation at or before target date",
                    }
                )
                continue

            historical_row = historical_candidates.iloc[-1]
            historical_date = historical_row["trade_date"]
            historical_spread = historical_row["spread_to_benchmark_bps"]
            movement = latest_spread - historical_spread
            table.loc[bucket, label] = movement
            audit_rows.append(
                {
                    "maturity_bucket": bucket,
                    "window": label,
                    "latest_date": latest_date,
                    "latest_spread_bps": latest_spread,
                    "target_date": target_date,
                    "historical_date": historical_date,
                    "historical_spread_bps": historical_spread,
                    "spread_movement_bps": movement,
                    "note": "Positive = widening; negative = tightening",
                }
            )

    return compact_maturity_table(table), pd.DataFrame(audit_rows)


@st.cache_data(show_spinner=False, max_entries=32)
def build_spread_level_data(
    market_df: pd.DataFrame,
    mmd_df: pd.DataFrame,
    issuer: str,
    ratings: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return current spread level table and audit table."""
    maturity_order = observed_maturity_years(market_df, min_observations=1) or MATURITY_BUCKET_ORDER
    clean_ratings = [r for r in ratings if r in BENCHMARK_RATINGS]
    table = pd.DataFrame(index=maturity_order, columns=clean_ratings, dtype="float")
    audit_rows: list[dict] = []

    if not clean_ratings or market_df.empty or mmd_df.empty:
        return table, pd.DataFrame(audit_rows)

    for rating in clean_ratings:
        spread_obs = build_spread_observations(
            market_df=market_df,
            mmd_df=mmd_df,
            issuer=issuer,
            rating=rating,
        )
        if spread_obs.empty:
            continue

        spread_obs = spread_obs.copy()
        spread_obs["trade_date"] = pd.to_datetime(spread_obs["trade_date"], errors="coerce").dt.normalize()
        spread_obs = spread_obs.dropna(subset=["trade_date", "spread_to_benchmark_bps"])

        for bucket in maturity_order:
            bucket_obs = spread_obs[spread_obs["maturity_bucket"] == bucket].sort_values("trade_date")
            if bucket_obs.empty:
                audit_rows.append(
                    {
                        "maturity_bucket": bucket,
                        "benchmark_rating": rating,
                        "latest_date": pd.NaT,
                        "avg_yield": pd.NA,
                        "benchmark_yield": pd.NA,
                        "spread_to_benchmark_bps": pd.NA,
                        "mmd_tenor": MMD_BUCKET_MAP.get(bucket),
                        "rating_spread_bps": RATING_SPREADS.get(rating, RATING_SPREADS["AAA"]).get(MMD_BUCKET_MAP.get(bucket, "10Y"), 0.00) * 100,
                        "benchmark_source": "No matching benchmark/date",
                        "source_column": pd.NA,
                        "trade_count": pd.NA,
                        "total_trade_amount": pd.NA,
                        "note": "No overlapping issuer trade and benchmark observation",
                    }
                )
                continue

            latest = bucket_obs.iloc[-1]
            spread_level = latest["spread_to_benchmark_bps"]
            table.loc[bucket, rating] = spread_level
            audit_rows.append(
                {
                    "maturity_bucket": bucket,
                    "benchmark_rating": rating,
                    "latest_date": latest["trade_date"],
                    "avg_yield": latest.get("avg_yield"),
                    "benchmark_yield": latest.get("benchmark_yield"),
                    "spread_to_benchmark_bps": spread_level,
                    "mmd_tenor": latest.get("mmd_tenor"),
                    "rating_spread_bps": latest.get("rating_spread_bps"),
                    "benchmark_source": latest.get("benchmark_source"),
                    "source_column": latest.get("source_column"),
                    "trade_count": latest.get("trade_count"),
                    "total_trade_amount": latest.get("total_trade_amount"),
                    "note": "Latest available spread observation for maturity year and benchmark",
                }
            )

    return compact_maturity_table(table), pd.DataFrame(audit_rows)


@st.cache_data(show_spinner=False, max_entries=32)
def build_issuer_curve_snapshot(
    market_df: pd.DataFrame,
    mmd_df: pd.DataFrame,
    issuer: str,
    ratings: list[str],
    as_of_date: pd.Timestamp,
    lookback_days: int,
    aggregation_method: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build issuer yield curve vs benchmark curves by maturity year."""
    maturity_order = MATURITY_BUCKET_ORDER
    clean_ratings = [r for r in ratings if r in BENCHMARK_RATINGS]

    if market_df.empty or mmd_df.empty or not clean_ratings:
        return pd.DataFrame(), pd.DataFrame()

    required_cols = {"issuer", "trade_date", "maturity_bucket", "yield"}
    if not required_cols.issubset(set(market_df.columns)):
        return pd.DataFrame(), pd.DataFrame()

    as_of_date = pd.to_datetime(as_of_date).normalize()
    issuer_df = market_df[market_df["issuer"] == issuer].copy()
    issuer_df = issuer_df[issuer_df["maturity_bucket"].isin(maturity_order)]
    issuer_df["trade_date"] = pd.to_datetime(issuer_df["trade_date"], errors="coerce").dt.normalize()
    issuer_df["yield"] = pd.to_numeric(issuer_df["yield"], errors="coerce")
    issuer_df = issuer_df.dropna(subset=["trade_date", "yield", "maturity_bucket"])
    issuer_df = issuer_df[issuer_df["trade_date"] <= as_of_date]
    if issuer_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    if aggregation_method == "Latest trade per bucket":
        latest_rows = (
            issuer_df.sort_values(["maturity_bucket", "trade_date"])
            .groupby("maturity_bucket", as_index=False)
            .tail(1)
        )
        issuer_curve = latest_rows[["maturity_bucket", "trade_date", "yield"]].rename(
            columns={"trade_date": "issuer_observation_date", "yield": "issuer_yield"}
        )
        counts = issuer_df.groupby("maturity_bucket", as_index=False).agg(trade_count=("yield", "count"))
        issuer_curve = issuer_curve.merge(counts, on="maturity_bucket", how="left")
        issuer_curve["aggregation_method"] = aggregation_method
        issuer_curve["lookback_start"] = pd.NaT
        issuer_curve["lookback_end"] = as_of_date
    else:
        lookback_start = as_of_date - pd.Timedelta(days=int(lookback_days))
        window_df = issuer_df[(issuer_df["trade_date"] >= lookback_start) & (issuer_df["trade_date"] <= as_of_date)].copy()
        if window_df.empty:
            return pd.DataFrame(), pd.DataFrame()
        agg_dict = {
            "issuer_yield": ("yield", "mean"),
            "trade_count": ("yield", "count"),
            "issuer_observation_date": ("trade_date", "max"),
        }
        if "trade_amount" in window_df.columns:
            agg_dict["total_trade_amount"] = ("trade_amount", "sum")
        issuer_curve = window_df.groupby("maturity_bucket", as_index=False).agg(**agg_dict)
        issuer_curve["aggregation_method"] = f"Average last {lookback_days} days"
        issuer_curve["lookback_start"] = lookback_start
        issuer_curve["lookback_end"] = as_of_date

    issuer_curve["maturity_bucket"] = pd.Categorical(
        issuer_curve["maturity_bucket"], categories=maturity_order, ordered=True
    )
    issuer_curve = issuer_curve.sort_values("maturity_bucket")

    date_col = detect_mmd_date_column(mmd_df)
    if date_col is None:
        return pd.DataFrame(), pd.DataFrame()

    mmd_base = mmd_df.copy()
    mmd_base[date_col] = pd.to_datetime(mmd_base[date_col], errors="coerce").dt.normalize()
    mmd_base = mmd_base.dropna(subset=[date_col])
    mmd_base = mmd_base[mmd_base[date_col] <= as_of_date]
    if mmd_base.empty:
        return pd.DataFrame(), pd.DataFrame()

    rows = []
    for rating in clean_ratings:
        for bucket in maturity_order:
            tenor = MMD_BUCKET_MAP.get(bucket, "10Y")
            y, meta = get_benchmark_curve(mmd_base, tenor, rating)
            if y is None:
                continue
            tmp = mmd_base[[date_col]].copy()
            tmp["benchmark_yield"] = pd.to_numeric(y, errors="coerce")
            tmp = tmp.dropna(subset=["benchmark_yield"])
            if tmp.empty:
                continue
            latest_bench = tmp.iloc[-1]
            rows.append(
                {
                    "maturity_bucket": bucket,
                    "benchmark_rating": rating,
                    "benchmark_date": latest_bench[date_col],
                    "benchmark_yield": latest_bench["benchmark_yield"],
                    "mmd_tenor": tenor,
                    "benchmark_source": meta.get("benchmark_source"),
                    "source_column": meta.get("source_column"),
                    "rating_spread_bps": meta.get("rating_spread_bps"),
                }
            )

    benchmark_curve = pd.DataFrame(rows)
    if benchmark_curve.empty:
        return pd.DataFrame(), pd.DataFrame()

    curve_data = issuer_curve.merge(benchmark_curve, on="maturity_bucket", how="inner")
    if curve_data.empty:
        return pd.DataFrame(), pd.DataFrame()

    curve_data["spread_to_benchmark_bps"] = (
        curve_data["issuer_yield"] - curve_data["benchmark_yield"]
    ) * 100

    issuer_line = issuer_curve[["maturity_bucket", "issuer_yield", "trade_count", "issuer_observation_date"]].copy()
    issuer_line = issuer_line.rename(columns={"issuer_yield": "yield_value"})
    issuer_line["curve"] = f"{issuer} issuer curve"
    issuer_line["curve_type"] = "Issuer"

    benchmark_line = benchmark_curve.rename(columns={"benchmark_yield": "yield_value"}).copy()
    benchmark_line["curve"] = benchmark_line["benchmark_rating"].astype(str) + " benchmark curve"
    benchmark_line["curve_type"] = "Benchmark"
    benchmark_line["trade_count"] = pd.NA
    benchmark_line["issuer_observation_date"] = pd.NaT

    plot_df = pd.concat(
        [
            issuer_line[["maturity_bucket", "yield_value", "curve", "curve_type", "trade_count", "issuer_observation_date"]],
            benchmark_line[["maturity_bucket", "yield_value", "curve", "curve_type", "trade_count", "issuer_observation_date"]],
        ],
        ignore_index=True,
    )
    plot_df["maturity_bucket"] = pd.Categorical(plot_df["maturity_bucket"], categories=maturity_order, ordered=True)
    plot_df = plot_df.sort_values(["curve_type", "curve", "maturity_bucket"])

    return plot_df, curve_data
