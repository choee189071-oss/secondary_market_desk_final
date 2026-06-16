# Data Safety And Deployment Notes

This project should be treated as a private analysis workstation while it uses MuniPro exports or other proprietary trade data.

## Data Handling

- Do not commit real MuniPro workbooks, proprietary trade exports, or analyst notes to GitHub.
- Use `data/golden/ladwp_expected.json` for approved output values, not raw source data.
- Keep source files local and upload them during the active Streamlit session.
- Review report downloads before sending them outside the team.

## Benchmark Policy

- The default Trading Workbench does not require an uploaded MMD file.
- Trade Sheet `Index Rate` / `Spread` can still support spread calculations when available.
- Benchmark file management is hidden from the normal user flow so analysts focus on trading activity, liquidity, participation, and CUSIP-level investigation.
- Ratings, sector, liquidity, and callable effects should remain disclosed separately from any benchmark spread.

## Deployment Checklist

Before sharing with professional analysts:

1. Run `scripts/regression_check.py` against the LADWP golden sample.
2. Confirm PDF and PPTX helper checks pass.
3. Confirm Streamlit startup smoke passes.
4. Confirm no proprietary source files are staged in Git.
5. Confirm Streamlit Cloud or internal hosting is private.
6. Confirm OpenAI features remain optional and are disabled unless explicitly being tested.

## Analyst Sharing Mode

For review sessions, ask analysts to use the Trading Workbench in order: Issuer Selection, Trading Filters, Market Analytics, Security Drilldown, Peer Comparison, Narrative Insights, and Export.
