from __future__ import annotations

import io
import hashlib
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from data_utils import (
    read_uploaded_file,
    standardize_bonds,
    standardize_issuer_mapping,
    standardize_mmd,
    standardize_trades,
)
from engine.benchmark import (
    MATURITY_BUCKET_RENAME,
    MAX_MATURITY_YEAR,
    detect_mmd_date_column,
    make_benchmark_long,
)
from engine.validation import normalize_col_name


try:
    import streamlit as st
except Exception:  # pragma: no cover
    class _StreamlitShim:
        @staticmethod
        def cache_data(*_args, **_kwargs):
            def decorator(func):
                return func
            return decorator

    st = _StreamlitShim()


def infer_issuer_from_description(description: object, fallback: str = "Unknown") -> str:
    """Conservative issuer extraction from MuniPro security descriptions."""
    text = "" if pd.isna(description) else str(description).strip()
    if not text:
        return fallback or "Unknown"

    upper = text.upper()
    cut_patterns = [
        r"\s+--\s+", r"\s+-\s+",
        r"\s+GO\s", r"\s+REV\s", r"\s+REF\s", r"\s+SER\s", r"\s+SERIES\s",
        r"\s+BONDS?\s", r"\s+CAP\s+APP", r"\s+VARIOUS\s+PURPOSE",
        r"\s+20\d{2}\b", r"\s+19\d{2}\b",
    ]
    cut_positions = []
    for pattern in cut_patterns:
        m = re.search(pattern, upper)
        if m and m.start() >= 4:
            cut_positions.append(m.start())
    if cut_positions:
        text = text[: min(cut_positions)].strip(" ,-")
    return text or fallback or "Unknown"


def issuer_from_source_file(source_file: object) -> str:
    """Use the uploaded MuniPro trade filename as the issuer name."""
    if pd.isna(source_file):
        return "Unknown"

    name = Path(str(source_file)).stem.strip()
    if not name:
        return "Unknown"

    name = re.sub(r"(?i)([_\-\s]+)?(trade|trades|trade[_\-\s]*history|munipro|export|secondary|market|history|data)$", "", name).strip()
    name = re.sub(r"(?i)([_\-\s]+)?(trade|trades|trade[_\-\s]*history|munipro|export|secondary|market|history|data)$", "", name).strip()
    name = re.sub(r"[_\-]+", " ", name)
    name = re.sub(r"\s+", " ", name).strip(" -_.,")

    if not name:
        return "Unknown"

    if len(name) <= 8 and name.replace(" ", "").isupper():
        return name

    return name.title().replace(" Ca ", " CA ").replace(" Usd", " USD ").replace(" Go ", " GO ")


@st.cache_data(show_spinner=False, max_entries=32)
def ensure_trade_only_fields(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Make standardized trade exports self-sufficient for dashboard analytics."""
    if trades_df is None or trades_df.empty:
        return pd.DataFrame()

    out = trades_df.copy()

    if "trade_date" in out.columns:
        out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce")
    if "trade_datetime" in out.columns:
        out["trade_datetime"] = pd.to_datetime(out["trade_datetime"], errors="coerce")
    elif "trade_date" in out.columns:
        out["trade_datetime"] = out["trade_date"]

    if "maturity" not in out.columns:
        if "maturity_trade" in out.columns:
            out["maturity"] = out["maturity_trade"]
        elif "maturity_date" in out.columns:
            out["maturity"] = out["maturity_date"]
    if "maturity_trade" not in out.columns and "maturity" in out.columns:
        out["maturity_trade"] = out["maturity"]
    if "maturity_bond" not in out.columns and "maturity" in out.columns:
        out["maturity_bond"] = out["maturity"]
    if "maturity" in out.columns:
        out["maturity"] = pd.to_datetime(out["maturity"], errors="coerce")
        out["maturity_trade"] = pd.to_datetime(out.get("maturity_trade", out["maturity"]), errors="coerce")
        out["maturity_bond"] = pd.to_datetime(out.get("maturity_bond", out["maturity"]), errors="coerce")

    for col in ["yield", "price", "trade_amount", "spread", "index_rate", "coupon"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if "coupon_trade" not in out.columns and "coupon" in out.columns:
        out["coupon_trade"] = out["coupon"]

    if "source_file" in out.columns:
        out["issuer"] = out["source_file"].apply(issuer_from_source_file)
    elif "issuer" not in out.columns:
        if "description" in out.columns:
            out["issuer"] = out["description"].apply(infer_issuer_from_description)
        else:
            out["issuer"] = "Unknown"
    else:
        missing_issuer = out["issuer"].isna() | (out["issuer"].astype(str).str.strip() == "") | (out["issuer"].astype(str).str.lower() == "unknown")
        if "description" in out.columns:
            out.loc[missing_issuer, "issuer"] = out.loc[missing_issuer, "description"].apply(infer_issuer_from_description)

    out["issuer"] = out["issuer"].fillna("Unknown").astype(str).str.strip()

    if "years_to_maturity" not in out.columns and {"maturity", "trade_date"}.issubset(out.columns):
        out["years_to_maturity"] = (out["maturity"] - out["trade_date"]).dt.days / 365.25

    if "maturity_year" not in out.columns and "years_to_maturity" in out.columns:
        y_for_year = pd.to_numeric(out["years_to_maturity"], errors="coerce")
        out["maturity_year"] = pd.Series(np.ceil(y_for_year), index=out.index).where(y_for_year.notna())
        out["maturity_year"] = out["maturity_year"].astype("Int64")

    if "maturity_year" in out.columns:
        yy = pd.to_numeric(out["maturity_year"], errors="coerce")
        out["maturity_bucket"] = yy.apply(lambda v: f"{int(v)}Y" if pd.notna(v) and int(v) >= 1 and int(v) <= MAX_MATURITY_YEAR else "Unknown")
    elif "maturity_bucket" in out.columns:
        out["maturity_bucket"] = out["maturity_bucket"].replace(MATURITY_BUCKET_RENAME).fillna("Unknown")

    if "sector" not in out.columns:
        out["sector"] = "Unknown"
    if "primary_type" not in out.columns:
        out["primary_type"] = pd.NA

    placeholder_defaults = {
        "description": "",
        "price": pd.NA,
        "trade_amount": 0,
        "coupon_bond": out["coupon_trade"] if "coupon_trade" in out.columns else pd.NA,
        "maturity_bond": out["maturity"] if "maturity" in out.columns else pd.NaT,
        "outstanding_amount": pd.NA,
        "call_date": pd.NaT,
        "call_price": pd.NA,
        "lien": pd.NA,
        "fed_tax": pd.NA,
        "amt": pd.NA,
        "secondary_credit": pd.NA,
        "series": pd.NA,
        "election": pd.NA,
        "term": pd.NA,
    }
    for col, default in placeholder_defaults.items():
        if col not in out.columns:
            out[col] = default

    for num_col in ["price", "trade_amount", "outstanding_amount", "call_price"]:
        if num_col in out.columns:
            out[num_col] = pd.to_numeric(out[num_col], errors="coerce")
    for date_col in ["maturity_bond", "call_date"]:
        if date_col in out.columns:
            out[date_col] = pd.to_datetime(out[date_col], errors="coerce")

    return out


def parse_trade_index_tenor(index_value: object) -> str | None:
    """Parse MuniPro Index labels such as AAA-5, AAA-10, AAA-20."""
    if pd.isna(index_value):
        return None
    text = str(index_value).upper().strip()
    m = re.search(r"(\d{1,2})\s*Y?$", text)
    if not m:
        return None
    year = int(m.group(1))
    if year < 1:
        return None
    return f"{min(year, MAX_MATURITY_YEAR)}Y"


@st.cache_data(show_spinner=False, max_entries=32)
def build_benchmark_curve_from_trade_index(market_df: pd.DataFrame) -> pd.DataFrame:
    """Build a benchmark curve table directly from trade-file Index / Index Rate columns."""
    required = {"trade_date", "index", "index_rate"}
    if market_df is None or market_df.empty or not required.issubset(set(market_df.columns)):
        return pd.DataFrame()

    tmp = market_df[list(required)].copy()
    tmp["trade_date"] = pd.to_datetime(tmp["trade_date"], errors="coerce").dt.normalize()
    tmp["index_rate"] = pd.to_numeric(tmp["index_rate"], errors="coerce")
    tmp["tenor"] = tmp["index"].apply(parse_trade_index_tenor)
    tmp = tmp.dropna(subset=["trade_date", "index_rate", "tenor"])
    if tmp.empty:
        return pd.DataFrame()

    daily = tmp.groupby(["trade_date", "tenor"], as_index=False)["index_rate"].median()
    wide = daily.pivot(index="trade_date", columns="tenor", values="index_rate").reset_index()
    wide = wide.rename(columns={"trade_date": "Date"})

    for tenor in [f"{y}Y" for y in range(1, MAX_MATURITY_YEAR + 1)]:
        if tenor in wide.columns:
            wide[f"AAA_{tenor}"] = wide[tenor]

    wide.attrs["benchmark_source_mode"] = "Trade Sheet Index / Index Rate"
    return wide.sort_values("Date").reset_index(drop=True)


def derive_source_spread_bps(df: pd.DataFrame) -> pd.Series:
    """Preserve source trade spread before active benchmark governance is applied."""
    if df is None or df.empty:
        return pd.Series(dtype="float64")

    if "spread_bps" in df.columns:
        return pd.to_numeric(df["spread_bps"], errors="coerce")

    if "spread" in df.columns and pd.to_numeric(df["spread"], errors="coerce").notna().any():
        raw = pd.to_numeric(df["spread"], errors="coerce")
        median_abs = raw.abs().dropna().median()
        return raw * 100 if pd.notna(median_abs) and median_abs <= 10 else raw

    if {"yield", "index_rate"}.issubset(df.columns):
        return (
            pd.to_numeric(df["yield"], errors="coerce")
            - pd.to_numeric(df["index_rate"], errors="coerce")
        ) * 100

    return pd.Series(pd.NA, index=df.index, dtype="Float64")


def attach_active_benchmark_spreads(
    market_df: pd.DataFrame,
    mmd_df: pd.DataFrame,
    benchmark_source_mode: str,
) -> pd.DataFrame:
    """Attach the governed benchmark spread used by workflow charts and RV scoring."""
    if market_df is None or market_df.empty:
        return market_df

    out = market_df.copy()
    out["source_spread_bps"] = derive_source_spread_bps(out)
    out["active_benchmark_yield"] = pd.NA
    out["active_benchmark_source"] = pd.NA
    out["active_benchmark_tenor"] = pd.NA
    out["active_benchmark_rating"] = "AAA"
    out["benchmark_spread_bps"] = pd.NA

    required_cols = {"trade_date", "maturity_bucket", "yield"}
    if (
        mmd_df is None
        or mmd_df.empty
        or not required_cols.issubset(set(out.columns))
    ):
        out["spread_bps"] = out["source_spread_bps"]
        out["spread_bps_source"] = "Source spread / trade Index Rate"
        return out

    benchmark_long = make_benchmark_long(mmd_df, "AAA")
    if benchmark_long.empty:
        out["spread_bps"] = out["source_spread_bps"]
        out["spread_bps_source"] = "Source spread / trade Index Rate"
        return out

    benchmark_cols = [
        "trade_date",
        "maturity_bucket",
        "benchmark_yield",
        "benchmark_source",
        "mmd_tenor",
        "benchmark_rating",
    ]
    benchmark_lookup = benchmark_long[[c for c in benchmark_cols if c in benchmark_long.columns]].copy()
    benchmark_lookup["trade_date"] = pd.to_datetime(benchmark_lookup["trade_date"], errors="coerce").dt.normalize()
    benchmark_lookup = benchmark_lookup.dropna(subset=["trade_date", "maturity_bucket", "benchmark_yield"])
    benchmark_lookup = benchmark_lookup.drop_duplicates(["trade_date", "maturity_bucket"], keep="last")

    working = out.copy()
    working["_row_id"] = np.arange(len(working))
    working["trade_date"] = pd.to_datetime(working["trade_date"], errors="coerce").dt.normalize()
    merged = working.merge(
        benchmark_lookup,
        on=["trade_date", "maturity_bucket"],
        how="left",
        suffixes=("", "_active"),
    )
    merged = merged.sort_values("_row_id").drop(columns=["_row_id"])

    merged["active_benchmark_yield"] = pd.to_numeric(merged.get("benchmark_yield"), errors="coerce")
    merged["active_benchmark_source"] = merged.get("benchmark_source")
    merged["active_benchmark_tenor"] = merged.get("mmd_tenor")
    merged["active_benchmark_rating"] = merged.get("benchmark_rating", "AAA")
    merged["benchmark_spread_bps"] = (
        pd.to_numeric(merged.get("yield"), errors="coerce")
        - pd.to_numeric(merged["active_benchmark_yield"], errors="coerce")
    ) * 100

    benchmark_spread = pd.to_numeric(merged["benchmark_spread_bps"], errors="coerce")
    source_spread = pd.to_numeric(merged["source_spread_bps"], errors="coerce")
    if benchmark_source_mode == "Uploaded MMD / AAA Curve":
        merged["spread_bps"] = benchmark_spread
        merged["spread_bps_source"] = np.where(
            benchmark_spread.notna(),
            "Uploaded MMD / AAA Curve",
            "Unavailable - uploaded MMD date/tenor gap",
        )
    elif benchmark_source_mode == "Trade Sheet Index / Index Rate":
        merged["spread_bps"] = benchmark_spread.combine_first(source_spread)
        merged["spread_bps_source"] = np.where(
            benchmark_spread.notna(),
            "Trade Sheet Index / Index Rate",
            "Source spread / trade Index Rate",
        )
    else:
        merged["spread_bps"] = source_spread
        merged["spread_bps_source"] = "Source spread / trade Index Rate"

    return merged.drop(columns=["benchmark_yield", "benchmark_source", "mmd_tenor", "benchmark_rating"], errors="ignore")


@st.cache_data(show_spinner=False, max_entries=32)
def build_issuer_master_from_trades(market_df: pd.DataFrame, issuer_mapping_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build issuer / sector reference from trades and optional mapping."""
    if market_df is None or market_df.empty or "issuer" not in market_df.columns:
        base = pd.DataFrame(columns=["issuer", "sector", "primary_type"])
    else:
        agg_cols = {"sector": "first", "primary_type": "first"}
        present = {k: v for k, v in agg_cols.items() if k in market_df.columns}
        if present:
            base = market_df.groupby("issuer", as_index=False).agg(present)
        else:
            base = market_df[["issuer"]].drop_duplicates().copy()
            base["sector"] = "Unknown"
            base["primary_type"] = pd.NA

    if issuer_mapping_df is not None and not issuer_mapping_df.empty:
        mapping = issuer_mapping_df.copy()
        if "issuer" in mapping.columns:
            keep_cols = [c for c in ["issuer", "sector", "primary_type"] if c in mapping.columns]
            mapping = mapping[keep_cols].drop_duplicates("issuer", keep="first")
            base = base.merge(mapping, on="issuer", how="left", suffixes=("", "_map"))
            for col in ["sector", "primary_type"]:
                map_col = f"{col}_map"
                if map_col in base.columns:
                    if col not in base.columns:
                        base[col] = base[map_col]
                    else:
                        base[col] = base[map_col].combine_first(base[col])
                    base = base.drop(columns=[map_col])

    if "sector" not in base.columns:
        base["sector"] = "Unknown"
    base["sector"] = base["sector"].fillna("Unknown").replace({"": "Unknown", "nan": "Unknown"})
    if "primary_type" not in base.columns:
        base["primary_type"] = pd.NA
    return base.sort_values("issuer").reset_index(drop=True)


@st.cache_data(show_spinner=False, max_entries=32)
def build_security_reference_from_trades(market_df: pd.DataFrame, optional_bonds_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Create a security reference table from the trade tape, enriched by optional bond data."""
    if market_df is None or market_df.empty or "cusip" not in market_df.columns:
        return pd.DataFrame()

    candidate_cols = [
        "issuer", "sector", "primary_type", "cusip", "description", "trade_date", "maturity", "maturity_trade",
        "coupon", "coupon_trade", "ratings_m_s_f", "rating", "index", "trade_type"
    ]
    cols = [c for c in candidate_cols if c in market_df.columns]
    ref = market_df[cols].dropna(subset=["cusip"]).copy()
    ref = ref.sort_values([c for c in ["cusip", "trade_date"] if c in market_df.columns]) if "trade_date" in market_df.columns else ref.sort_values("cusip")
    ref = ref.drop_duplicates("cusip", keep="last")

    trade_agg = market_df.groupby("cusip", as_index=False).agg(
        trade_count=("cusip", "count"),
        first_trade=("trade_date", "min") if "trade_date" in market_df.columns else ("cusip", "count"),
        latest_trade=("trade_date", "max") if "trade_date" in market_df.columns else ("cusip", "count"),
        total_trade_amount=("trade_amount", "sum") if "trade_amount" in market_df.columns else ("cusip", "count"),
    )
    ref = ref.merge(trade_agg, on="cusip", how="left")

    if optional_bonds_df is not None and not optional_bonds_df.empty and "cusip" in optional_bonds_df.columns:
        bond_enrich = optional_bonds_df.drop_duplicates("cusip", keep="first").copy()
        enrich_cols = [c for c in [
            "cusip", "lien", "election", "series", "secondary_credit", "term", "par_amount",
            "outstanding_amount", "call_date", "call_price", "fed_tax", "amt"
        ] if c in bond_enrich.columns]
        if len(enrich_cols) > 1:
            ref = ref.merge(bond_enrich[enrich_cols], on="cusip", how="left")

    return ref.reset_index(drop=True)


MMD_FALLBACK_LOOKBACK_YEARS = 2
MMD_MAX_TENOR_YEAR = 40
ANALYTICS_CACHE_DIR = Path(tempfile.gettempdir()) / "secondary_market_desk_analytics_cache"
ANALYTICS_CATEGORY_COLUMNS = {
    "issuer",
    "sector",
    "primary_type",
    "cusip",
    "maturity_bucket",
    "trade_type",
    "ratings_m_s_f",
    "rating",
    "index",
    "active_benchmark_source",
    "active_benchmark_tenor",
    "active_benchmark_rating",
    "spread_bps_source",
}


def optimize_market_frame_for_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce memory pressure after upload so reruns/groupbys stay responsive."""
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    for col in out.columns:
        if col in {"trade_date", "trade_datetime", "maturity", "maturity_trade", "maturity_bond", "call_date"}:
            out[col] = pd.to_datetime(out[col], errors="coerce")
        elif col in {
            "yield",
            "price",
            "trade_amount",
            "spread",
            "spread_bps",
            "source_spread_bps",
            "benchmark_spread_bps",
            "index_rate",
            "coupon",
            "coupon_trade",
            "coupon_bond",
            "outstanding_amount",
            "call_price",
        }:
            out[col] = pd.to_numeric(out[col], errors="coerce", downcast="float")
        elif col in {"maturity_year"}:
            out[col] = pd.to_numeric(out[col], errors="coerce", downcast="integer")
        elif col in ANALYTICS_CATEGORY_COLUMNS:
            out[col] = out[col].astype("string").fillna("Unknown").astype("category")
    return out


def _payload_digest(payloads: list[tuple[str, bytes]] | tuple[tuple[str, bytes], ...]) -> str:
    digest = hashlib.sha256()
    for name, payload in payloads:
        digest.update(str(name).encode("utf-8", errors="ignore"))
        digest.update(str(len(payload or b"")).encode("ascii"))
        digest.update(payload or b"")
    return digest.hexdigest()[:24]


def materialize_market_parquet(market_df: pd.DataFrame, digest: str) -> str:
    """Persist the processed market tape as a temp parquet analytics layer."""
    if market_df is None or market_df.empty or not digest:
        return ""
    try:
        ANALYTICS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = ANALYTICS_CACHE_DIR / f"market_{digest}.parquet"
        if not path.exists():
            market_df.to_parquet(path, index=False)
        return str(path)
    except Exception:
        return ""


def is_date_like_col(col: object) -> bool:
    key = normalize_col_name(col)
    return key in {"date", "trade date", "pricing date", "curve date", "mmd date"}


def is_mmd_tenor_col(col: object, max_year: int = MMD_MAX_TENOR_YEAR) -> bool:
    """Return True for tenor columns like 1Y, 1-Yr, 01Y, AAA_10Y, MMD 30Y."""
    text = str(col).strip().upper()
    match = re.search(
        r"(?:^|[^0-9])0?([1-9]|[1-3][0-9]|40)\s*[-_ ]?\s*(?:Y|YR|YEAR|YEARS)(?:[^0-9]|$)",
        text,
    )
    if not match:
        return False
    year = int(match.group(1))
    return 1 <= year <= max_year


@st.cache_data(show_spinner=False, max_entries=32)
def trim_mmd_frame(mmd_df: pd.DataFrame, lookback_years: int = MMD_FALLBACK_LOOKBACK_YEARS) -> pd.DataFrame:
    """Keep only recent dates and needed benchmark tenor columns."""
    if mmd_df is None or mmd_df.empty:
        return pd.DataFrame()

    out = mmd_df.copy()
    date_col = detect_mmd_date_column(out)
    if date_col is None:
        return out

    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.dropna(subset=[date_col])
    if not out.empty:
        cutoff = out[date_col].max() - pd.DateOffset(years=int(lookback_years))
        out = out[out[date_col] >= cutoff].copy()

    keep_cols = [date_col]
    for col in out.columns:
        if col == date_col:
            continue
        if is_mmd_tenor_col(col):
            keep_cols.append(col)

    keep_cols = [c for c in keep_cols if c in out.columns]
    return out[keep_cols].reset_index(drop=True) if keep_cols else out.reset_index(drop=True)


@st.cache_data(show_spinner=False, max_entries=32)
def read_external_mmd_fallback_file(
    file_name: str,
    payload: bytes,
    lookback_years: int = MMD_FALLBACK_LOOKBACK_YEARS,
) -> pd.DataFrame:
    """Memory-aware MMD reader for the uploaded AAA benchmark curve."""
    suffix = Path(file_name).suffix.lower()

    if suffix == ".csv":
        header = pd.read_csv(io.BytesIO(payload), nrows=0)
        usecols = [c for c in header.columns if is_date_like_col(c) or is_mmd_tenor_col(c)]
        if not usecols:
            raw_mmd = read_uploaded_file(io.BytesIO(payload), file_name)
        else:
            raw_mmd = pd.read_csv(io.BytesIO(payload), usecols=usecols)
    else:
        raw_mmd = read_uploaded_file(io.BytesIO(payload), file_name)

    standardized = standardize_mmd(raw_mmd)
    return trim_mmd_frame(standardized, lookback_years=lookback_years)


@st.cache_data(show_spinner="Processing uploaded data...")
def process_uploads(
    trade_payloads: list[tuple[str, bytes]],
    issuer_mapping_payload: tuple[str, bytes] | None,
    mmd_payload: tuple[str, bytes] | None,
    bond_payload: tuple[str, bytes] | None = None,
):
    """Process a trade-first dashboard dataset."""
    optional_bonds_df = pd.DataFrame()
    if bond_payload is not None:
        try:
            bond_name, bond_bytes = bond_payload
            raw_bonds = read_uploaded_file(io.BytesIO(bond_bytes), bond_name)
            optional_bonds_df = standardize_bonds(raw_bonds)
        except Exception:
            optional_bonds_df = pd.DataFrame()

    issuer_mapping_df = pd.DataFrame()
    if issuer_mapping_payload is not None:
        name, payload = issuer_mapping_payload
        raw_mapping = read_uploaded_file(io.BytesIO(payload), name)
        issuer_mapping_df = standardize_issuer_mapping(raw_mapping)

    trade_frames = []
    failed_files = []
    for trade_name, trade_bytes in trade_payloads:
        try:
            raw_trade = read_uploaded_file(io.BytesIO(trade_bytes), trade_name)
            standardized = standardize_trades(raw_trade, source_file=trade_name)
            standardized = ensure_trade_only_fields(standardized)
            trade_frames.append(standardized)
        except Exception as exc:
            failed_files.append((trade_name, str(exc)))

    trades_df = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()

    before_dedup = len(trades_df)
    trades_df = trades_df.drop_duplicates().reset_index(drop=True)
    duplicates_removed = before_dedup - len(trades_df)

    market_df = ensure_trade_only_fields(trades_df)

    if not optional_bonds_df.empty and "cusip" in optional_bonds_df.columns and "cusip" in market_df.columns:
        enrich = optional_bonds_df.drop_duplicates("cusip", keep="first").copy()
        enrich_cols = [c for c in [
            "cusip", "issuer", "sector", "primary_type", "maturity", "coupon", "lien", "election",
            "series", "secondary_credit", "term", "par_amount", "outstanding_amount", "call_date",
            "call_price", "fed_tax", "amt"
        ] if c in enrich.columns]
        if len(enrich_cols) > 1:
            market_df = market_df.merge(enrich[enrich_cols], on="cusip", how="left", suffixes=("", "_bondref"))
            for col in ["sector", "primary_type", "maturity", "coupon", "issuer"]:
                ref_col = f"{col}_bondref"
                if ref_col in market_df.columns:
                    if col in ["issuer"]:
                        missing = market_df[col].isna() | (market_df[col].astype(str).str.strip() == "") | (market_df[col].astype(str).str.lower() == "unknown")
                        market_df.loc[missing, col] = market_df.loc[missing, ref_col]
                    else:
                        market_df[col] = market_df[col].combine_first(market_df[ref_col])
                    market_df = market_df.drop(columns=[ref_col])

    issuer_master = build_issuer_master_from_trades(market_df, issuer_mapping_df)

    if not issuer_master.empty and "issuer" in market_df.columns:
        map_cols = [c for c in ["issuer", "sector", "primary_type"] if c in issuer_master.columns]
        market_df = market_df.drop(columns=[c for c in ["sector", "primary_type"] if c in market_df.columns], errors="ignore")
        market_df = market_df.merge(issuer_master[map_cols], on="issuer", how="left")
        market_df["sector"] = market_df.get("sector", pd.Series(index=market_df.index, dtype="object")).fillna("Unknown")

    bonds_df = build_security_reference_from_trades(market_df, optional_bonds_df)
    trade_index_curve_df = build_benchmark_curve_from_trade_index(market_df)

    uploaded_mmd_df = pd.DataFrame()
    if mmd_payload is not None:
        name, payload = mmd_payload
        uploaded_mmd_df = read_external_mmd_fallback_file(
            file_name=name,
            payload=payload,
            lookback_years=MMD_FALLBACK_LOOKBACK_YEARS,
        )

    trade_index_available = not trade_index_curve_df.empty
    uploaded_mmd_available = not uploaded_mmd_df.empty

    if uploaded_mmd_available:
        mmd_df = uploaded_mmd_df
        mmd_df.attrs["benchmark_source_mode"] = "Uploaded MMD / AAA Curve"
        mmd_df.attrs["benchmark_source_priority"] = "Primary"
        mmd_df.attrs["uploaded_mmd_available"] = True
        mmd_df.attrs["trade_index_available"] = trade_index_available
        mmd_df.attrs["benchmark_conflict_policy"] = (
            "Uploaded MMD is the active AAA benchmark; trade-sheet Index / Index Rate is retained for audit but not used as the benchmark curve."
        )
    elif trade_index_available:
        mmd_df = trade_index_curve_df
        mmd_df.attrs["benchmark_source_mode"] = "Trade Sheet Index / Index Rate"
        mmd_df.attrs["benchmark_source_priority"] = "Fallback"
        mmd_df.attrs["uploaded_mmd_available"] = uploaded_mmd_available
        mmd_df.attrs["trade_index_available"] = True
        mmd_df.attrs["benchmark_conflict_policy"] = "No uploaded MMD was provided; using trade-sheet Index / Index Rate fallback."
    else:
        mmd_df = uploaded_mmd_df
        mmd_df.attrs["benchmark_source_mode"] = "None"
        mmd_df.attrs["benchmark_source_priority"] = "Unavailable"
        mmd_df.attrs["uploaded_mmd_available"] = False
        mmd_df.attrs["trade_index_available"] = False
        mmd_df.attrs["benchmark_conflict_policy"] = "No uploaded MMD or usable trade-sheet Index / Index Rate benchmark was found."

    market_df = attach_active_benchmark_spreads(market_df, mmd_df, mmd_df.attrs.get("benchmark_source_mode", "None"))
    market_df = optimize_market_frame_for_analysis(market_df)
    analytics_payloads = list(trade_payloads)
    for optional_payload in [issuer_mapping_payload, mmd_payload, bond_payload]:
        if optional_payload is not None:
            analytics_payloads.append(optional_payload)
    analytics_digest = _payload_digest(tuple(analytics_payloads))
    market_df.attrs["analytics_store_path"] = materialize_market_parquet(market_df, analytics_digest)
    market_df.attrs["analytics_store_format"] = "parquet"

    return bonds_df, trades_df, issuer_master, market_df, mmd_df, failed_files, duplicates_removed
