from __future__ import annotations

import re

import pandas as pd


DATA_MODEL: dict[str, list[str]] = {
    "issuer": ["issuer", "Issuer", "issuer_name", "obligor"],
    "sector": ["sector", "Sector", "industry"],
    "cusip": ["cusip", "CUSIP", "cusip9", "CUSIP9"],
    "trade_date": ["trade_date", "Trade Date", "Trade Date/Time", "trade_datetime", "date", "Date"],
    "maturity": ["maturity", "Maturity", "Maturity Date", "maturity_date", "maturity_trade"],
    "maturity_bucket": ["maturity_bucket", "maturity_year", "maturity_zone", "bucket", "tenor", "Maturity Year"],
    "maturity_year": ["maturity_year", "maturity_bucket", "Maturity Year", "tenor"],
    "yield": ["yield", "Yield", "avg_yield", "issuer_yield", "current_avg_yield"],
    "spread_bps": ["spread_to_benchmark_bps", "spread_bps", "current_spread_bps", "spread", "Spread"],
    "benchmark_yield": ["benchmark_yield", "Index Rate", "index_rate"],
    "benchmark_rating": ["benchmark_rating", "Benchmark Rating", "rating", "Rating"],
    "liquidity_score": ["liquidity_score", "Liquidity Score"],
    "trade_amount": ["trade_amount", "Trade Amount", "total_trade_amount", "volume"],
    "trade_count": ["trade_count", "Trade Count", "recent_90d_trades"],
    "peer_gap_bps": ["peer_gap_bps", "Peer Gap", "peer_gap"],
    "rv_score": ["rv_score", "RV Score"],
}


def resolve_model_col(df: pd.DataFrame, concept: str, required: bool = False) -> str | None:
    """Return the actual dataframe column for a central dashboard concept."""
    if df is None or not isinstance(df, pd.DataFrame):
        if required:
            raise KeyError(f"No dataframe supplied while resolving required concept: {concept}")
        return None
    candidates = DATA_MODEL.get(concept, [concept])
    exact = {str(c): c for c in df.columns}
    lowered = {str(c).strip().lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate in exact:
            return exact[candidate]
        key = str(candidate).strip().lower()
        if key in lowered:
            return lowered[key]
    if required:
        raise KeyError(f"Required concept '{concept}' not found. Available columns: {list(df.columns)}")
    return None


def coerce_maturity_label(value: object) -> str | pd.NA:
    """Normalize maturity labels to dashboard labels such as 5Y or 13-20Y."""
    if pd.isna(value):
        return pd.NA
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "unknown"}:
        return pd.NA
    zone_values = {"1-3Y", "4-7Y", "8-12Y", "13-20Y", "21Y+"}
    if text.upper() in {z.upper() for z in zone_values}:
        return text.upper().replace("Y+", "Y+")
    m = re.search(r"(\d{1,2})", text)
    if m:
        year = int(m.group(1))
        max_year = 40
        if 1 <= year <= max_year:
            return f"{year}Y"
    return text


def ensure_model_columns(df: pd.DataFrame, concepts: list[str] | None = None) -> pd.DataFrame:
    """Return a copy with central model columns added when aliases exist."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame() if df is None else df
    out = df.copy()
    concepts = concepts or list(DATA_MODEL.keys())
    canonical_map = {
        "spread_bps": "spread_to_benchmark_bps",
        "maturity_bucket": "maturity_bucket",
        "maturity_year": "maturity_year",
        "benchmark_rating": "benchmark_rating",
        "liquidity_score": "liquidity_score",
        "peer_gap_bps": "peer_gap_bps",
        "rv_score": "rv_score",
    }
    for concept in concepts:
        target = canonical_map.get(concept, concept)
        if target in out.columns:
            continue
        src = resolve_model_col(out, concept, required=False)
        if src is not None and src in out.columns:
            out[target] = out[src]
    if "maturity_bucket" in out.columns:
        out["maturity_bucket"] = out["maturity_bucket"].apply(coerce_maturity_label)
    return out


def safe_melt_by_maturity(
    df: pd.DataFrame,
    value_name: str = "spread_to_benchmark_bps",
    var_name: str = "benchmark_rating",
    maturity_concept: str = "maturity_bucket",
    value_vars: list[str] | None = None,
    id_vars: str | list[str] | None = None,
) -> pd.DataFrame:
    """Melt wide matrices defensively using the central maturity concept."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()

    out = df.copy()
    out = out.loc[:, ~out.columns.duplicated()].copy()

    explicit_id_vars = []
    if id_vars is not None:
        explicit_id_vars = [id_vars] if isinstance(id_vars, str) else list(id_vars)
        explicit_id_vars = [c for c in explicit_id_vars if c in out.columns]

    maturity_col = explicit_id_vars[0] if explicit_id_vars else resolve_model_col(out, maturity_concept, required=False)

    if maturity_col is None:
        index_name = out.index.name or "maturity_bucket"
        out = out.reset_index().rename(columns={index_name: "maturity_bucket", "index": "maturity_bucket"})
        maturity_col = resolve_model_col(out, maturity_concept, required=False)

    if maturity_col is None or maturity_col not in out.columns:
        return pd.DataFrame()

    if maturity_col != "maturity_bucket":
        out = out.rename(columns={maturity_col: "maturity_bucket"})
        maturity_col = "maturity_bucket"
    out["maturity_bucket"] = out["maturity_bucket"].apply(coerce_maturity_label)

    if value_vars is None:
        value_vars = [c for c in out.columns if c != maturity_col]
    value_vars = [c for c in value_vars if c in out.columns and c != maturity_col]
    if not value_vars:
        return pd.DataFrame()
    melted = out.melt(
        id_vars=maturity_col,
        value_vars=value_vars,
        var_name=var_name,
        value_name=value_name,
    )
    melted[value_name] = pd.to_numeric(melted[value_name], errors="coerce")
    melted = melted.dropna(subset=[value_name, maturity_col])
    return melted
