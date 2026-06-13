# Reviewer Handoff Pack

Use this page when asking a professional analyst to review the Streamlit workstation. It is intentionally short: the reviewer should spend time on the workflow and outputs, not on reading documentation.

## Demo Flow

1. Upload trusted trade file and AAA MMD file.
2. Open `Upload / Data Audit`; confirm ready status and benchmark source.
3. Open `Desk Snapshot`; review headline metrics and top names.
4. Open `Core Charts`; review Spread Trend first, then add Volume or Curve if needed.
5. Open `CUSIP Drilldown`; inspect top CUSIP trade path and same-bucket peers.
6. Save any candidate to `RV / Watchlist`.
7. Open `Export / Methodology`; complete Analyst Review Mode and download outputs.

## Analyst Review Checklist

- Does the selected issuer/date range match the source file?
- Do trade rows, CUSIP count, and top CUSIP reconcile to the uploaded data?
- Is the active benchmark source correct for this review?
- Do median spread and liquidity look reasonable?
- Does Spread Trend tell the right market story?
- Does Volume support or weaken the signal?
- Does Issuer Curve placement make sense versus reference lines?
- Does the selected CUSIP trade path reconcile to source trades?
- Are same-bucket peers economically comparable?
- Are watchlist candidates names an analyst would actually investigate?
- Are report exports clear enough to send for internal review?

## Known Limitations / Questions

- Benchmark priority: should Trade Sheet Index / Index Rate remain primary when uploaded MMD is also available?
- MMD treatment: uploaded MMD is currently treated as AAA; confirm this is correct for the source file.
- Rating fallback: when ratings are missing, the app falls back to sector and maturity. Confirm whether this is acceptable.
- Liquidity score: current score blends trade count, par amount, and recency. Confirm weighting.
- RV score: current score is a screening blend of spread rank and liquidity rank. Confirm weighting and thresholds.
- Callable, sector, and liquidity effects are displayed separately, not embedded into benchmark spread. Confirm this methodology.
- Watchlist saves candidates for the active Streamlit session; confirm whether persistent storage is needed.

## Feedback Format

- Mark each checklist item as `Correct`, `Needs Review`, or `Wrong`.
- For `Wrong`, provide the expected value and source evidence.
- For methodology questions, state the preferred rule and whether it should become a regression check.
- For UI feedback, identify whether the issue blocks analysis or is only a layout preference.
