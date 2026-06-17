from __future__ import annotations

import re
from pathlib import Path
from typing import BinaryIO, Optional

import pandas as pd


# ============================================================
# Basic cleaning helpers
# ============================================================

def clean_colname(col: object) -> str:
    """Normalize uploaded column names into snake_case."""
    text = str(col).strip().lower()
    text = re.sub(r"[^0-9a-z]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def clean_money_series(s: pd.Series) -> pd.Series:
    """Remove common money / percent formatting before numeric conversion."""
    return (
        s.astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip()
        .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "NaN": pd.NA})
    )


def clean_numeric(s: pd.Series) -> pd.Series:
    """Convert formatted numeric-like series to float."""
    return pd.to_numeric(clean_money_series(s), errors="coerce")


def clean_cusip(s: pd.Series) -> pd.Series:
    """Clean CUSIP / CUSIP9 fields while preserving leading zeros."""
    return (
        s.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .replace({"nan": pd.NA, "": pd.NA, "None": pd.NA, "NaN": pd.NA})
    )


def infer_issuer_from_filename(filename: str | None) -> object:
    """
    Infer issuer from trade file name.

    Example:
        LADWP_Trade.xlsx -> LADWP
        State_of_CA_Trades.csv -> State of CA
    """
    if not filename:
        return pd.NA
    stem = Path(filename).stem
    stem = re.sub(r"[_\-\s]*(Trade|Trades|trade|trades)\s*$", "", stem)
    return stem.replace("_", " ").strip()


def read_uploaded_file(uploaded_file: BinaryIO, filename: str) -> pd.DataFrame:
    """Read CSV/XLS/XLSX uploaded through Streamlit."""
    suffix = Path(filename).suffix.lower()
    if suffix in [".xlsx", ".xls"]:
        try:
            return pd.read_excel(uploaded_file, sheet_name="ag-grid", dtype=str)
        except Exception:
            uploaded_file.seek(0)
            return pd.read_excel(uploaded_file, dtype=str)
    return pd.read_csv(uploaded_file, dtype=str)


# ============================================================
# Bond file standardization
# Bonds are optional. The app should still work trades-only.
# ============================================================

def standardize_bonds(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [clean_colname(c) for c in df.columns]

    rename_map = {
        "cusip": "cusip",
        "cusip9": "cusip",
        "issuer": "issuer",
        "secondary_credit": "secondary_credit",
        "maturity": "maturity",
        "maturity_date": "maturity",
        "par_amount": "par_amount",
        "outstanding_amount": "outstanding_amount",
        "coupon": "coupon",
        "call_date": "call_date",
        "call_price": "call_price",
        "fed_tax": "fed_tax",
        "tax_status": "fed_tax",
        "amt": "amt",
        "series": "series",
        "election": "election",
        "type": "type",
        "lien": "lien",
        "term": "term",
        "sector": "sector",
        "primary_type": "primary_type",
    }
    df = df.rename(columns={c: rename_map.get(c, c) for c in df.columns})

    required_cols = [
        "issuer", "type", "lien", "election", "series", "cusip",
        "secondary_credit", "term", "maturity", "par_amount",
        "outstanding_amount", "coupon", "call_date", "call_price",
        "fed_tax", "amt", "sector", "primary_type",
    ]
    for col in required_cols:
        if col not in df.columns:
            df[col] = pd.NA

    df["cusip"] = clean_cusip(df["cusip"])
    df["issuer"] = df["issuer"].astype(str).str.strip().replace({"nan": pd.NA, "": pd.NA})
    df["sector"] = df["sector"].astype(str).str.strip().replace({"nan": pd.NA, "": pd.NA})
    df["primary_type"] = df["primary_type"].astype(str).str.strip().replace({"nan": pd.NA, "": pd.NA})

    for col in ["series", "secondary_credit", "term", "type", "lien", "election", "fed_tax", "amt"]:
        df[col] = df[col].astype(str).str.strip().replace({"nan": pd.NA, "": pd.NA})

    df["maturity"] = pd.to_datetime(df["maturity"], errors="coerce")
    df["call_date"] = pd.to_datetime(df["call_date"], errors="coerce")

    for col in ["par_amount", "outstanding_amount", "coupon", "call_price"]:
        df[col] = clean_numeric(df[col])

    df = df[df["cusip"].notna()].copy()
    df = df[df["maturity"].notna()].copy()

    today = pd.Timestamp.today().normalize()
    df["years_to_maturity"] = (df["maturity"] - today).dt.days / 365.25
    return df[required_cols + ["years_to_maturity"]]


# ============================================================
# Trade file standardization
# ============================================================

def standardize_trades(df: pd.DataFrame, source_file: Optional[str] = None) -> pd.DataFrame:
    df = df.copy()
    df.columns = [clean_colname(c) for c in df.columns]

    rename_map = {
        # Common MuniPro export headers
        "td_time": "trade_datetime",
        "trade_date_time": "trade_datetime",
        "trade_date_time": "trade_datetime",
        "trade_datetime": "trade_datetime",
        "trade_time": "trade_datetime",
        "datetime": "trade_datetime",
        "cusip9": "cusip",
        "cusip": "cusip",
        "security_id": "cusip",
        "security_description": "description",
        "bond_description": "description",
        "description": "description",
        "mty": "maturity",
        "maturity_date": "maturity",
        "maturity": "maturity",
        "trade_date": "trade_date",
        "date": "trade_date",
        "transaction_date": "trade_date",
        "settle_date": "settlement_date",
        "settlement_date": "settlement_date",
        "cpn": "coupon",
        "coupon": "coupon",
        "coupon_rate": "coupon",
        "ytw": "yield",
        "ytm": "yield",
        "yt_par": "yield",
        "yt_prm": "yield",
        "yt_sink": "yield",
        "msrb_yld": "yield",
        "yield_to_worst": "yield",
        "yield_to_maturity": "yield",
        "yield": "yield",
        "yield_": "yield",
        "price": "price",
        "trade_price": "price",
        "execution_price": "price",
        "qty_m": "trade_amount",
        "quantity": "trade_amount",
        "amount": "trade_amount",
        "par_traded": "trade_amount",
        "trade_amount": "trade_amount",
        "par_amount": "trade_amount",
        "calculation_date": "calculation_date",
        "calculation_price": "calculation_price",
        "bnch_year": "index",
        "benchmark_year": "index",
        "benchmark": "index",
        "index": "index",
        "bnch_rate": "index_rate",
        "benchmark_rate": "index_rate",
        "index_rate": "index_rate",
        "spread_bp": "spread",
        "spread_bps": "spread",
        "spread_to_benchmark": "spread",
        "spread": "spread",
        "tde_type": "trade_type",
        "side": "trade_type",
        "buy_sell": "trade_type",
        "trade_type": "trade_type",
        "m_s_f": "ratings_m_s_f",
        "ratings_m_s_f": "ratings_m_s_f",
        "ratings": "ratings_m_s_f",
        "rating": "ratings_m_s_f",
    }
    amount_was_qty_m = "qty_m" in df.columns and "trade_amount" not in df.columns
    df = df.rename(columns={c: rename_map.get(c, c) for c in df.columns})
    if df.columns.duplicated().any():
        df = df.T.groupby(level=0, sort=False).first().T
    if "ratings_m_s_f" not in df.columns and {"m", "s", "f"}.issubset(df.columns):
        def _clean_rating_part(value: object) -> str:
            if pd.isna(value):
                return ""
            text = str(value).strip()
            return "" if text.lower() in {"", "nan", "none", "<na>"} else text

        rating_parts = df[["m", "s", "f"]].apply(lambda col: col.map(_clean_rating_part))
        df["ratings_m_s_f"] = rating_parts.apply(
            lambda row: "/".join([part for part in row.tolist() if part]),
            axis=1,
        ).replace({"": pd.NA})

    required_cols = [
        "trade_datetime", "cusip", "description", "maturity", "trade_date",
        "settlement_date", "coupon", "yield", "price", "trade_amount",
        "calculation_date", "calculation_price", "index", "index_rate",
        "spread", "trade_type", "ratings_m_s_f",
    ]
    for col in required_cols:
        if col not in df.columns:
            df[col] = pd.NA

    df["cusip"] = clean_cusip(df["cusip"])

    # Prefer explicit trade_date. If missing, fall back to trade_datetime.
    for col in ["trade_datetime", "maturity", "trade_date", "settlement_date", "calculation_date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    df["trade_date"] = df["trade_date"].fillna(df["trade_datetime"])

    for col in ["coupon", "yield", "price", "trade_amount", "calculation_price", "index_rate", "spread"]:
        df[col] = clean_numeric(df[col])
    if amount_was_qty_m:
        df["trade_amount"] = df["trade_amount"] * 1000

    df["source_file"] = source_file or pd.NA
    df["source_issuer_guess"] = infer_issuer_from_filename(source_file)

    df = df[df["cusip"].notna()].copy()
    df = df[df["trade_date"].notna()].copy()

    return df[required_cols + ["source_file", "source_issuer_guess"]]


# ============================================================
# Issuer mapping / master
# ============================================================

def standardize_issuer_mapping(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [clean_colname(c) for c in df.columns]

    for col in ["issuer", "sector", "primary_type", "notes"]:
        if col not in df.columns:
            df[col] = pd.NA

    df["issuer"] = df["issuer"].astype(str).str.strip().replace({"nan": pd.NA, "": pd.NA})
    df["sector"] = (
        df["sector"]
        .astype(str)
        .str.strip()
        .replace({"nan": "Unassigned", "": "Unassigned", "None": "Unassigned"})
    )
    df["primary_type"] = df["primary_type"].astype(str).str.strip().replace({"nan": pd.NA, "": pd.NA})
    df["notes"] = df["notes"].astype(str).str.strip().replace({"nan": pd.NA, "": pd.NA})

    return (
        df[df["issuer"].notna()][["issuer", "sector", "primary_type", "notes"]]
        .drop_duplicates("issuer", keep="last")
    )


def build_issuer_master(
    bonds_df: Optional[pd.DataFrame] = None,
    issuer_mapping_df: Optional[pd.DataFrame] = None,
    trades_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Build issuer master from any available layer.

    Priority:
    1. issuer_mapping_df if supplied
    2. bonds_df issuer / sector / primary_type if supplied
    3. trades_df source_issuer_guess if running trades-only
    """
    pieces: list[pd.DataFrame] = []

    if bonds_df is not None and not bonds_df.empty and "issuer" in bonds_df.columns:
        cols = ["issuer", "sector", "primary_type"]
        from_bonds = bonds_df[[c for c in cols if c in bonds_df.columns]].drop_duplicates("issuer").copy()
        if "sector" not in from_bonds.columns:
            from_bonds["sector"] = "Unassigned"
        if "primary_type" not in from_bonds.columns:
            from_bonds["primary_type"] = pd.NA
        from_bonds["notes"] = pd.NA
        pieces.append(from_bonds[["issuer", "sector", "primary_type", "notes"]])

    if trades_df is not None and not trades_df.empty and "source_issuer_guess" in trades_df.columns:
        from_trades = (
            trades_df[["source_issuer_guess"]]
            .rename(columns={"source_issuer_guess": "issuer"})
            .dropna()
            .drop_duplicates()
            .copy()
        )
        from_trades["sector"] = "Unassigned"
        from_trades["primary_type"] = pd.NA
        from_trades["notes"] = "Inferred from uploaded trade file name"
        pieces.append(from_trades[["issuer", "sector", "primary_type", "notes"]])

    if issuer_mapping_df is not None and not issuer_mapping_df.empty:
        pieces.append(issuer_mapping_df[["issuer", "sector", "primary_type", "notes"]].copy())

    if not pieces:
        return pd.DataFrame(columns=["issuer", "sector", "primary_type", "notes"])

    combined = pd.concat(pieces, ignore_index=True)

    combined["issuer"] = combined["issuer"].astype(str).str.strip().replace({"nan": pd.NA, "": pd.NA})
    combined["sector"] = (
        combined["sector"]
        .astype(str)
        .str.strip()
        .replace({"nan": "Unassigned", "": "Unassigned", "None": "Unassigned"})
    )
    combined["primary_type"] = combined["primary_type"].astype(str).str.strip().replace({"nan": pd.NA, "": pd.NA})
    combined["notes"] = combined["notes"].astype(str).str.strip().replace({"nan": pd.NA, "": pd.NA})

    combined = combined[combined["issuer"].notna()].drop_duplicates("issuer", keep="last")
    return combined.sort_values(["sector", "issuer"])[["issuer", "sector", "primary_type", "notes"]]


# ============================================================
# Maturity logic
# ============================================================

def assign_maturity_bucket(years: float) -> object:
    """Legacy maturity buckets kept for backward compatibility."""
    if pd.isna(years):
        return pd.NA
    if years <= 7:
        return "Short"
    if years <= 15:
        return "10Y"
    if years <= 25:
        return "20Y"
    return "30Y"


def assign_maturity_year(years: float) -> object:
    """Annual maturity bucket: 1Y, 2Y, ..., 40Y."""
    if pd.isna(years):
        return pd.NA
    year = int(round(float(years)))
    year = max(1, min(40, year))
    return f"{year}Y"


def assign_maturity_zone(years: float) -> object:
    """Desk-friendly grouped maturity zones."""
    if pd.isna(years):
        return pd.NA
    years = float(years)
    if years <= 3:
        return "1-3Y"
    if years <= 7:
        return "4-7Y"
    if years <= 12:
        return "8-12Y"
    if years <= 20:
        return "13-20Y"
    return "21Y+"


# ============================================================
# Market merge
# ============================================================

def merge_market_data(
    bonds_df: Optional[pd.DataFrame],
    trades_df: pd.DataFrame,
    issuer_master: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Merge trades with optional bond/reference data.

    Works in both:
    - trades + bonds mode
    - trades-only mode
    """
    if trades_df is None or trades_df.empty:
        return pd.DataFrame()

    trades = trades_df.copy()

    # Trades-only base
    market_df = trades.copy()
    market_df["issuer"] = market_df.get("source_issuer_guess", pd.NA)
    market_df["sector"] = "Unassigned"
    market_df["primary_type"] = pd.NA

    # Optional bond enrichment
    if bonds_df is not None and not bonds_df.empty and "cusip" in bonds_df.columns:
        bonds = bonds_df.copy()

        if issuer_master is not None and not issuer_master.empty and "issuer" in bonds.columns:
            bonds = bonds.drop(columns=["sector", "primary_type"], errors="ignore").merge(
                issuer_master[["issuer", "sector", "primary_type"]],
                on="issuer",
                how="left",
            )

        market_df = trades.merge(
            bonds,
            on="cusip",
            how="left",
            suffixes=("_trade", "_bond"),
        )

        if "issuer" not in market_df.columns:
            market_df["issuer"] = market_df.get("source_issuer_guess", pd.NA)
        else:
            market_df["issuer"] = market_df["issuer"].fillna(market_df.get("source_issuer_guess", pd.NA))

        if "sector" not in market_df.columns:
            market_df["sector"] = "Unassigned"
        else:
            market_df["sector"] = market_df["sector"].fillna("Unassigned")

        if "primary_type" not in market_df.columns:
            market_df["primary_type"] = pd.NA

    # Optional issuer mapping overlay
    if issuer_master is not None and not issuer_master.empty and "issuer" in market_df.columns:
        market_df = market_df.drop(columns=["sector", "primary_type"], errors="ignore").merge(
            issuer_master[["issuer", "sector", "primary_type"]],
            on="issuer",
            how="left",
        )
        market_df["sector"] = market_df["sector"].fillna("Unassigned")

    # Resolve maturity from bond if available, otherwise trade file maturity.
    maturity_bond = market_df["maturity_bond"] if "maturity_bond" in market_df.columns else pd.Series(pd.NaT, index=market_df.index)
    maturity_trade = market_df["maturity_trade"] if "maturity_trade" in market_df.columns else (
        market_df["maturity"] if "maturity" in market_df.columns else pd.Series(pd.NaT, index=market_df.index)
    )

    maturity_final = pd.to_datetime(maturity_bond, errors="coerce").fillna(pd.to_datetime(maturity_trade, errors="coerce"))
    market_df["maturity"] = maturity_final

    trade_date = pd.to_datetime(market_df["trade_date"], errors="coerce")
    market_df["years_to_maturity_at_trade"] = (market_df["maturity"] - trade_date).dt.days / 365.25

    market_df["maturity_bucket"] = market_df["years_to_maturity_at_trade"].apply(assign_maturity_bucket)
    market_df["maturity_year"] = market_df["years_to_maturity_at_trade"].apply(assign_maturity_year)
    market_df["maturity_zone"] = market_df["years_to_maturity_at_trade"].apply(assign_maturity_zone)

    # Normalize common bond/trade fields after merge.
    for col in ["coupon", "yield", "price", "trade_amount", "spread", "index_rate"]:
        if col in market_df.columns:
            market_df[col] = pd.to_numeric(market_df[col], errors="coerce")

    return market_df


# ============================================================
# MMD / benchmark curve standardization
# ============================================================

def standardize_mmd(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize MMD-style curve sheet.

    Accepts headers like:
        Date, 1-Yr, 2-Yr, 10-Yr
        date, 1Yr, 2Yr
        Date, 1Y, 2Y
    """
    df = df.copy()

    new_cols = []
    for c in df.columns.astype(str):
        col = c.strip()
        col = col.replace("-Yr", "Y").replace("Yr", "Y")
        col = col.replace("-YR", "Y").replace("YR", "Y")
        col = col.replace("-yr", "Y").replace("yr", "Y")
        col = col.replace("-", "")
        new_cols.append(col)

    df.columns = new_cols

    date_col = "Date" if "Date" in df.columns else "date" if "date" in df.columns else None
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        if date_col != "Date":
            df = df.rename(columns={date_col: "Date"})

    # Convert tenor columns to numeric where possible.
    for c in df.columns:
        if c != "Date":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "Date" in df.columns:
        df = df[df["Date"].notna()].copy()
        df = df.sort_values("Date")

    return df
