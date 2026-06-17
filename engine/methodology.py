from __future__ import annotations

import pandas as pd


def _first_existing_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None
    lowered = {str(col).lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
        match = lowered.get(candidate.lower())
        if match is not None:
            return match
    return None


def _nonnull_rate(df: pd.DataFrame, col: str | None) -> float | None:
    if not isinstance(df, pd.DataFrame) or df.empty or not col or col not in df.columns:
        return None
    return float(df[col].notna().mean() * 100)


def _numeric_rate(df: pd.DataFrame, col: str | None) -> float | None:
    if not isinstance(df, pd.DataFrame) or df.empty or not col or col not in df.columns:
        return None
    return float(pd.to_numeric(df[col], errors="coerce").notna().mean() * 100)


def _status_from_rate(rate: float | None, good_threshold: float = 80, warn_threshold: float = 40) -> str:
    if rate is None:
        return "bad"
    if rate >= good_threshold:
        return "good"
    if rate >= warn_threshold:
        return "warn"
    return "bad"


def _format_rate(rate: float | None) -> str:
    return "N/A" if rate is None else f"{rate:.1f}%"


def _date_range_text(df: pd.DataFrame) -> str:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return "N/A"
    date_col = _first_existing_col(df, ["trade_date", "date", "sale_date"])
    if not date_col:
        return "N/A"
    dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
    if dates.empty:
        return "N/A"
    return f"{dates.min().date()} to {dates.max().date()}"


def methodology_evidence_summary(
    market_df: pd.DataFrame,
    issuer_df: pd.DataFrame | None,
    mmd_df: pd.DataFrame,
    benchmark_source_mode: str,
    benchmark_priority: str,
    benchmark_conflict_policy: str,
) -> dict[str, object]:
    """Compact evidence layer used consistently across workflow pages."""
    issuer_df = issuer_df if isinstance(issuer_df, pd.DataFrame) and not issuer_df.empty else market_df
    benchmark_source = benchmark_source_mode or "No active benchmark"
    benchmark_status = "good" if benchmark_source in {"Trade Sheet Index / Index Rate", "Uploaded MMD fallback"} else "bad"

    yield_col = _first_existing_col(issuer_df, ["yield", "yield_percent", "trade_yield"])
    index_col = _first_existing_col(issuer_df, ["index_rate", "index yield", "index_yield"])
    spread_col = _first_existing_col(issuer_df, ["spread_bps", "current_spread_bps", "spread"])
    cusip_col = _first_existing_col(issuer_df, ["cusip", "cusip9"])
    maturity_col = _first_existing_col(issuer_df, ["maturity_bucket", "maturity_date", "maturity"])
    rating_col = _first_existing_col(issuer_df, ["ratings_m_s_f", "rating", "ratings", "benchmark_rating"])
    sector_col = _first_existing_col(issuer_df, ["sector", "sector_name", "security_sector"])
    trade_amount_col = _first_existing_col(issuer_df, ["trade_amount", "par_amount", "par", "principal_amount"])

    yield_rate = _numeric_rate(issuer_df, yield_col)
    index_rate = _numeric_rate(issuer_df, index_col)
    spread_rate = _numeric_rate(issuer_df, spread_col)
    cusip_rate = _nonnull_rate(issuer_df, cusip_col)
    maturity_rate = _nonnull_rate(issuer_df, maturity_col)
    rating_rate = _nonnull_rate(issuer_df, rating_col)
    sector_rate = _nonnull_rate(issuer_df, sector_col)
    amount_rate = _numeric_rate(issuer_df, trade_amount_col)

    spread_status = "good" if (yield_rate or 0) >= 80 and ((index_rate or 0) >= 50 or (spread_rate or 0) >= 50 or benchmark_status == "good") else "warn"
    if (yield_rate or 0) < 40 and (spread_rate or 0) < 40:
        spread_status = "bad"

    peer_status = "good"
    peer_value = f"Rating {_format_rate(rating_rate)}"
    if rating_rate is None or rating_rate < 50:
        peer_status = "warn" if (sector_rate or 0) >= 50 and (maturity_rate or 0) >= 70 else "bad"
        peer_value = f"Sector {_format_rate(sector_rate)} / maturity {_format_rate(maturity_rate)}"

    liquidity_status = "good" if (amount_rate or 0) >= 70 and (cusip_rate or 0) >= 90 else "warn"
    if (cusip_rate or 0) < 60:
        liquidity_status = "bad"

    date_range = _date_range_text(issuer_df)
    cards = [
        {
            "status": benchmark_status,
            "kicker": "Benchmark",
            "title": "Active source",
            "value": benchmark_source,
            "detail": benchmark_conflict_policy or "One benchmark source is used per run.",
        },
        {
            "status": spread_status,
            "kicker": "Spread",
            "title": "Traceability",
            "value": f"Yield {_format_rate(yield_rate)} / index {_format_rate(index_rate)}",
            "detail": "Spread uses yield minus active benchmark, or supplied spread when available.",
        },
        {
            "status": peer_status,
            "kicker": "Peer RV",
            "title": "Grouping basis",
            "value": peer_value,
            "detail": "Ratings are preferred; missing ratings fall back to sector and maturity.",
        },
        {
            "status": liquidity_status,
            "kicker": "CUSIP",
            "title": "Detail reliability",
            "value": f"CUSIP {_format_rate(cusip_rate)}",
            "detail": "CUSIP, par amount, and recency drive drilldown, liquidity, and watchlist confidence.",
        },
    ]

    evidence_rows = [
        {
            "Control": "Benchmark source",
            "Current reading": benchmark_source,
            "Status": benchmark_status,
            "Evidence": f"Priority: {benchmark_priority or 'N/A'}; benchmark rows: {len(mmd_df) if isinstance(mmd_df, pd.DataFrame) else 0:,}",
            "Reviewer action": "Confirm this is the expected desk benchmark for the selected issuer and date range.",
        },
        {
            "Control": "Spread calculation",
            "Current reading": f"Yield numeric {_format_rate(yield_rate)}; Index Rate numeric {_format_rate(index_rate)}; Spread numeric {_format_rate(spread_rate)}",
            "Status": spread_status,
            "Evidence": "Formula: issuer yield minus active benchmark yield, expressed in bps.",
            "Reviewer action": "Spot-check several rows against source trade yield and benchmark/index rate.",
        },
        {
            "Control": "Data scope",
            "Current reading": f"{len(issuer_df):,} issuer rows; {len(market_df) if isinstance(market_df, pd.DataFrame) else 0:,} filtered market rows",
            "Status": "good" if len(issuer_df) > 0 else "bad",
            "Evidence": f"Trade date range: {date_range}",
            "Reviewer action": "Confirm the active global filters match the intended review window.",
        },
        {
            "Control": "Peer grouping",
            "Current reading": f"Ratings {_format_rate(rating_rate)}; sector {_format_rate(sector_rate)}; maturity {_format_rate(maturity_rate)}",
            "Status": peer_status,
            "Evidence": "Ratings drive peer grouping when present; otherwise sector and maturity are used.",
            "Reviewer action": "Review peer set quality before relying on RV ranking.",
        },
        {
            "Control": "Liquidity inputs",
            "Current reading": f"CUSIP {_format_rate(cusip_rate)}; trade amount numeric {_format_rate(amount_rate)}",
            "Status": liquidity_status,
            "Evidence": "Liquidity blends trade count, total par, and recency.",
            "Reviewer action": "Treat liquidity as directional when par amount or CUSIP quality is weak.",
        },
    ]

    return {
        "cards": cards,
        "evidence": pd.DataFrame(evidence_rows),
    }


def methodology_trust_layers(
    benchmark_source_mode: str,
    benchmark_priority: str,
    benchmark_conflict_policy: str,
) -> dict[str, pd.DataFrame]:
    """Structured methodology tables shown in the trust layer and reports."""
    benchmark_rows = [
        {
            "Topic": "Benchmark hierarchy",
            "Current policy": benchmark_priority or "Trade Sheet Index / Index Rate first; uploaded MMD fallback second",
            "Why it matters": "Prevents mixing two curve sources with different dates, tenors, and rounding conventions.",
            "Analyst validation question": "Is this the benchmark source the desk expects for this issuer/date window?",
        },
        {
            "Topic": "Active benchmark source",
            "Current policy": benchmark_source_mode or "Unknown",
            "Why it matters": "Every spread/RV conclusion must disclose which benchmark produced the spread.",
            "Analyst validation question": "Do the displayed spreads reconcile to the uploaded trade Index Rate or approved MMD file?",
        },
        {
            "Topic": "MMD treatment",
            "Current policy": "Uploaded MMD is treated as AAA and used only when external MMD fallback is active.",
            "Why it matters": "Rating, sector, liquidity, and callable effects remain separate from the benchmark curve.",
            "Analyst validation question": "Is the uploaded MMD file the correct AAA curve for this analysis date range?",
        },
        {
            "Topic": "Conflict policy",
            "Current policy": benchmark_conflict_policy or "Do not blend trade-sheet Index Rate and uploaded MMD in one run.",
            "Why it matters": "Blended benchmarks can create false spread movement or peer gaps.",
            "Analyst validation question": "Should any exception to the no-mixing rule be documented?",
        },
    ]

    scoring_rows = [
        {
            "Score": "Spread",
            "Formula": "Spread bps = (yield - active benchmark yield) x 100, or uploaded spread when already supplied.",
            "Inputs": "yield, index_rate/spread, trade_date, maturity_bucket",
            "Validation cue": "Spot-check several trades against source Index Rate and uploaded MMD, if active.",
        },
        {
            "Score": "Liquidity",
            "Formula": "Percentile blend of trade count, total par, and recency.",
            "Inputs": "cusip, trade_date, trade_amount",
            "Validation cue": "High scores should correspond to observable recent trades and meaningful par amount.",
        },
        {
            "Score": "RV",
            "Formula": "Screening blend of spread rank and liquidity rank.",
            "Inputs": "current_spread_bps, liquidity_score, maturity_bucket",
            "Validation cue": "Top candidates should be reviewed at CUSIP detail before any recommendation.",
        },
        {
            "Score": "Peer gap",
            "Formula": "CUSIP spread minus same-maturity-bucket median spread.",
            "Inputs": "cusip summary, maturity_bucket, current_spread_bps",
            "Validation cue": "Same-bucket peer set should be large enough and economically comparable.",
        },
    ]

    fallback_rows = [
        {
            "Missing / weak input": "Ratings",
            "Fallback": "Use sector and maturity bucket for peer context.",
            "Disclosure": "Rating effects are not embedded into benchmark spread when ratings are missing.",
        },
        {
            "Missing / weak input": "Index Rate",
            "Fallback": "Use uploaded MMD only when external MMD fallback is enabled and active.",
            "Disclosure": "Spread/RV views are degraded if neither source is available.",
        },
        {
            "Missing / weak input": "CUSIP",
            "Fallback": "Run issuer-level analytics; suppress or weaken CUSIP drilldown/watchlist confidence.",
            "Disclosure": "CUSIP quality card should be reviewed before trusting security-level output.",
        },
        {
            "Missing / weak input": "Trade amount",
            "Fallback": "Liquidity uses observable trade count and recency but loses par support.",
            "Disclosure": "Liquidity score should be treated as incomplete.",
        },
    ]

    safety_rows = [
        {
            "Area": "Raw data",
            "Control": "Users upload authorized files during the active session; real MuniPro exports should not be committed to GitHub.",
            "Reviewer note": "Confirm deployment is private before sharing with external analysts.",
        },
        {
            "Area": "Exports",
            "Control": "Markdown, HTML, CSV, PDF, PPTX, and bundle downloads are user-triggered deliverables.",
            "Reviewer note": "Review export contents before distributing outside the team.",
        },
        {
            "Area": "AI commentary",
            "Control": "Current workflow is rule-based plus structured context; OpenAI is optional and not required for core analytics.",
            "Reviewer note": "Keep AI disabled for methodology validation unless explicitly testing commentary.",
        },
        {
            "Area": "Regression",
            "Control": "Golden sample expected outputs can block changes that drift from approved methodology.",
            "Reviewer note": "Analyst-approved expected values should be updated deliberately, not opportunistically.",
        },
    ]

    return {
        "Benchmark Policy": pd.DataFrame(benchmark_rows),
        "Scoring Rules": pd.DataFrame(scoring_rows),
        "Fallback Logic": pd.DataFrame(fallback_rows),
        "Data Safety": pd.DataFrame(safety_rows),
    }


def analyst_review_items(context: dict) -> pd.DataFrame:
    """Default checklist shown in Analyst Review Mode."""
    metrics = {}
    try:
        metrics = {row["Metric"]: row["Value"] for _, row in context.get("metrics", pd.DataFrame()).iterrows()}
    except Exception:
        metrics = {}

    top = context.get("top_opportunities", pd.DataFrame())
    top_cusip = "N/A" if top is None or top.empty else str(top.iloc[0].get("cusip", "N/A"))

    return pd.DataFrame(
        [
            {
                "Review item": "Benchmark source",
                "Current output": metrics.get("Benchmark source", context.get("benchmark_source_mode", "N/A")),
                "Suggested expected value": metrics.get("Benchmark source", context.get("benchmark_source_mode", "N/A")),
                "Why review": "Benchmark source drives spread, RV, and methodology disclosures.",
            },
            {
                "Review item": "Trade rows",
                "Current output": metrics.get("Trade rows", "N/A"),
                "Suggested expected value": metrics.get("Trade rows", "N/A"),
                "Why review": "Row count confirms upload/dedup logic did not unexpectedly change.",
            },
            {
                "Review item": "CUSIPs",
                "Current output": metrics.get("CUSIPs", "N/A"),
                "Suggested expected value": metrics.get("CUSIPs", "N/A"),
                "Why review": "CUSIP count affects drilldown, liquidity, and watchlist coverage.",
            },
            {
                "Review item": "Median spread",
                "Current output": metrics.get("Median spread", "N/A"),
                "Suggested expected value": metrics.get("Median spread", "N/A"),
                "Why review": "Spread level is a core desk conclusion and should reconcile to source data.",
            },
            {
                "Review item": "Median liquidity",
                "Current output": metrics.get("Median liquidity", "N/A"),
                "Suggested expected value": metrics.get("Median liquidity", "N/A"),
                "Why review": "Liquidity scoring can be sensitive to recency and par amount assumptions.",
            },
            {
                "Review item": "Top CUSIP",
                "Current output": top_cusip,
                "Suggested expected value": top_cusip,
                "Why review": "Top candidate determines analyst attention and report read-through.",
            },
        ]
    )
