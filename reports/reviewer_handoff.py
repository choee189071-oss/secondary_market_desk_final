from __future__ import annotations


def reviewer_handoff_markdown(selected_issuer: str = "Selected issuer") -> str:
    """One-page reviewer handoff used by docs and the Streamlit export page."""
    issuer = selected_issuer or "Selected issuer"
    return f"""# Secondary Market Workstation Reviewer Handoff

Issuer for demo: `{issuer}`

## Demo Flow

1. Upload trusted trade file.
2. Select sector, issuer, and date range in `Issuer Selection`.
3. Apply maturity, trade-size, trade-type, and lot/block filters in `Trading Filters`.
4. Review `Market Analytics`: volume overview, activity concentration map, participation, and ranked liquidity bands.
5. Open `Security Drilldown`; inspect the CUSIPs driving the filtered activity.
6. Review `Peer Comparison` under the same filters.
7. Save candidate CUSIPs into `RV / Watchlist`, then move each item through `New`, `Reviewing`, `Need Data Check`, `Approved`, or `Rejected`.
8. Read `Narrative Insights` and confirm each observation is supported by the filtered data.
9. Use `Export / Methodology` to download filtered trades, security drilldown, watchlist review output, Excel workbook, PDF summary, or PPTX outline.

## Analyst Review Checklist

- Does the selected issuer/date range match the source file?
- Do trade rows, CUSIP count, and top CUSIP reconcile to the uploaded data?
- Do the maturity, trade-size, trade-type, and lot/block filters behave as expected?
- Do total par, trade count, and average trade size reconcile to the filtered source data?
- Does the activity concentration map correctly show where trading is concentrated?
- Does dealer/customer/interdealer participation look correct?
- Do liquidity metrics by maturity and trade size look reasonable?
- Does the selected CUSIP trade path reconcile to source trades?
- Are peer issuers compared under equivalent trading conditions?
- Does each saved candidate have the correct review stage, decision, reviewer note, and next step?
- Are report exports clear enough to send for internal review?

## Known Limitations / Questions

- Trade type classification is inferred from source text such as Customer Bought, Customer Sold, Dealer, and Inter-Dealer. Confirm naming conventions.
- Trade-size buckets assume trade amount is reported as par amount in dollars. Confirm source scaling.
- Odd Lot / Round Lot / Block Trade is inferred from par size. Confirm preferred thresholds.
- Liquidity is currently displayed from trade frequency, par amount, spread/yield, and recency; confirm whether a formal liquidity score is needed.
- Uploaded MMD is the primary AAA benchmark when provided; trade-sheet Index / Index Rate is fallback and audit evidence.
- Watchlist review status is session-based until exported; confirm whether production needs permanent user accounts and database-backed review history.

## Feedback Format

- Mark each checklist item as `Correct`, `Needs Review`, or `Wrong`.
- For `Wrong`, provide the expected value and source evidence.
- For methodology questions, state the preferred rule and whether it should become a regression check.
- For UI feedback, identify whether the issue blocks analysis or is only a layout preference.
"""
