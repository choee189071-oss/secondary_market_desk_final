from __future__ import annotations

import pandas as pd


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
