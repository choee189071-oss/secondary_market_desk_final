from __future__ import annotations

import pandas as pd


COLUMN_ALIASES: dict[str, list[str]] = {
    "cusip": ["cusip", "cusip9", "cusip 9", "cusip_9", "security id", "security_id"],
    "issuer": ["issuer", "issuer name", "issuer_name", "obligor", "borrower"],
    "sector": ["sector", "industry", "sector name", "sector_name"],
    "primary_type": ["type", "primary type", "primary_type", "bond type"],
    "lien": ["lien"],
    "election": ["election"],
    "series": ["series"],
    "secondary_credit": ["secondary credit", "secondary_credit", "credit", "credit enhancement"],
    "term": ["term"],
    "maturity": ["maturity", "maturity date", "maturity_date"],
    "par_amount": ["par amount", "par_amount", "par", "amount issued"],
    "outstanding_amount": ["outstanding amount", "outstanding_amount", "amount outstanding", "current amount outstanding"],
    "coupon": ["coupon", "coupon rate", "coupon_rate"],
    "call_date": ["call date", "call_date", "first call date", "first_call_date"],
    "call_price": ["call price", "call_price"],
    "fed_tax": ["fed tax", "fed_tax", "tax status", "tax_status"],
    "amt": ["amt", "alternative minimum tax"],
    "rating": ["rating", "ratings", "ratings m/s/f", "ratings_m_s_f", "moody/s&p/fitch"],
    "trade_datetime": ["trade date/time", "trade datetime", "trade_datetime", "datetime"],
    "trade_date": ["trade date", "trade_date", "date", "transaction date"],
    "settlement_date": ["settlement date", "settlement_date", "settle date"],
    "description": ["description", "security description", "bond description"],
    "maturity_trade": ["maturity date", "maturity_date", "maturity"],
    "yield": ["yield", "yield to worst", "ytw", "yield_to_worst", "yield to maturity", "ytm"],
    "price": ["price", "trade price", "execution price"],
    "trade_amount": ["trade amount", "trade_amount", "par traded", "par amount", "amount", "quantity"],
    "calculation_date": ["calculation date", "calculation_date"],
    "calculation_price": ["calculation price", "calculation_price"],
    "index": ["index", "benchmark", "bnch year", "bench year", "benchmark year", "benchmark index", "m index", "m(index)", "maturity index"],
    "index_rate": ["index rate", "index_rate", "benchmark rate", "bnch rate", "bench rate", "benchmark yield"],
    "spread": ["spread", "g spread", "z spread", "spread to benchmark"],
    "trade_type": ["trade type", "trade_type", "side", "buy/sell"],
    "1Y": ["1y", "1 yr", "1 year", "1-year", "1-yr"],
    "2Y": ["2y", "2 yr", "2 year", "2-year", "2-yr"],
    "5Y": ["5y", "5 yr", "5 year", "5-year", "5-yr"],
    "10Y": ["10y", "10 yr", "10 year", "10-year", "10-yr"],
    "20Y": ["20y", "20 yr", "20 year", "20-year", "20-yr"],
    "30Y": ["30y", "30 yr", "30 year", "30-year", "30-yr"],
}

BOND_REQUIRED = ["cusip", "issuer", "maturity"]
BOND_RECOMMENDED = [
    "coupon", "outstanding_amount", "call_date", "call_price", "sector", "secondary_credit", "fed_tax", "amt"
]
BOND_OPTIONAL = ["primary_type", "lien", "election", "series", "term", "par_amount", "rating"]

TRADE_REQUIRED = ["cusip", "trade_date", "yield"]
TRADE_RECOMMENDED = ["price", "trade_amount", "trade_type", "spread", "settlement_date", "rating"]
TRADE_OPTIONAL = ["trade_datetime", "description", "maturity_trade", "calculation_date", "calculation_price", "index", "index_rate"]

MMD_REQUIRED = ["date"]
MMD_RECOMMENDED = ["1Y", "2Y", "5Y", "10Y", "20Y", "30Y"]
CURVE_TEMPLATE_COLUMNS = [
    "date", "5Y", "10Y", "20Y", "30Y",
    "AA+_5Y", "AA+_10Y", "AA+_20Y", "AA+_30Y",
    "AA_5Y", "AA_10Y", "AA_20Y", "AA_30Y",
    "AA-_5Y", "AA-_10Y", "AA-_20Y", "AA-_30Y",
    "A_5Y", "A_10Y", "A_20Y", "A_30Y",
    "BBB_5Y", "BBB_10Y", "BBB_20Y", "BBB_30Y",
]


def normalize_col_name(name: object) -> str:
    """Normalize external column names so Munipro/Excel variants can be detected."""
    text = str(name).strip().lower()
    for ch in ["_", "-", "/", "\\", "\n", "\t"]:
        text = text.replace(ch, " ")
    return " ".join(text.split())


def find_column(df: pd.DataFrame, canonical_name: str) -> str | None:
    """Return the actual uploaded column matching a canonical internal field."""
    normalized_columns = {normalize_col_name(c): c for c in df.columns}
    aliases = COLUMN_ALIASES.get(canonical_name, [canonical_name])
    for alias in aliases:
        hit = normalized_columns.get(normalize_col_name(alias))
        if hit is not None:
            return hit
    return None


def build_column_mapping(df: pd.DataFrame, expected_fields: list[str]) -> dict[str, str | None]:
    return {field: find_column(df, field) for field in expected_fields}


def validate_dataset(
    df: pd.DataFrame,
    dataset_name: str,
    required_fields: list[str],
    recommended_fields: list[str],
    optional_fields: list[str] | None = None,
) -> dict:
    """Create a file-readiness report without blocking on non-critical fields."""
    optional_fields = optional_fields or []
    all_fields = required_fields + recommended_fields + optional_fields
    mapping = build_column_mapping(df, all_fields)

    missing_required = [field for field in required_fields if mapping.get(field) is None]
    missing_recommended = [field for field in recommended_fields if mapping.get(field) is None]
    detected_required = [field for field in required_fields if mapping.get(field) is not None]
    detected_recommended = [field for field in recommended_fields if mapping.get(field) is not None]

    return {
        "dataset": dataset_name,
        "can_run": len(missing_required) == 0,
        "row_count": len(df),
        "column_count": len(df.columns),
        "mapping": mapping,
        "missing_required": missing_required,
        "missing_recommended": missing_recommended,
        "detected_required": detected_required,
        "detected_recommended": detected_recommended,
    }


def validate_basic_values(df: pd.DataFrame, mapping: dict[str, str | None], dataset_type: str) -> list[str]:
    """Soft data-quality checks. These generate warnings instead of killing the app."""
    warnings: list[str] = []

    cusip_col = mapping.get("cusip")
    if cusip_col and cusip_col in df.columns:
        blank_cusips = df[cusip_col].isna().sum() + (df[cusip_col].astype(str).str.strip() == "").sum()
        if blank_cusips:
            warnings.append(f"{blank_cusips:,} row(s) have blank CUSIP values.")

    date_field = "maturity" if dataset_type == "bond" else "trade_date"
    date_col = mapping.get(date_field)
    if date_col and date_col in df.columns:
        parsed = pd.to_datetime(df[date_col], errors="coerce")
        bad_dates = parsed.isna().sum()
        if bad_dates:
            warnings.append(f"{bad_dates:,} row(s) have invalid or blank {date_field} values.")

    yield_col = mapping.get("yield")
    if yield_col and yield_col in df.columns:
        parsed_yield = pd.to_numeric(df[yield_col], errors="coerce")
        bad_yields = parsed_yield.isna().sum()
        extreme_yields = ((parsed_yield < -5) | (parsed_yield > 30)).sum()
        if bad_yields:
            warnings.append(f"{bad_yields:,} row(s) have non-numeric yield values.")
        if extreme_yields:
            warnings.append(f"{extreme_yields:,} row(s) have yield values outside the expected -5% to 30% range.")

    amount_col = mapping.get("trade_amount") or mapping.get("outstanding_amount")
    if amount_col and amount_col in df.columns:
        parsed_amount = pd.to_numeric(df[amount_col], errors="coerce")
        negative_amounts = (parsed_amount < 0).sum()
        if negative_amounts:
            warnings.append(f"{negative_amounts:,} row(s) have negative amount values.")

    return warnings
