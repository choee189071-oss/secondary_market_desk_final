# Data Safety And Deployment Notes

This project should be treated as a private analysis workstation while it uses MuniPro exports or other proprietary trade data.

## Data Handling

- Do not commit real MuniPro workbooks, proprietary trade exports, or analyst notes to GitHub.
- Use `data/golden/ladwp_expected.json` for approved output values, not raw source data.
- Keep source files local and upload them during the active Streamlit session.
- Review report downloads before sending them outside the team.

## Benchmark Policy

- Uploaded MMD is treated as the AAA curve.
- Trade Sheet `Index Rate` / `Spread` remains the active benchmark source when available.
- Uploaded MMD is used as fallback only when trade-sheet benchmark data is unavailable.
- Ratings, sector, liquidity, and callable effects are disclosed separately instead of embedded into the benchmark spread.

## Deployment Checklist

Before sharing with professional analysts:

1. Run `scripts/regression_check.py` against the LADWP golden sample.
2. Confirm PDF and PPTX helper checks pass.
3. Confirm Streamlit startup smoke passes.
4. Confirm no proprietary source files are staged in Git.
5. Confirm Streamlit Cloud or internal hosting is private.
6. Confirm OpenAI features remain optional and are disabled unless explicitly being tested.

## Analyst Sharing Mode

For review sessions, ask analysts to focus on the Export / Methodology page after using the workflow. Analyst Review Mode captures approved expected values and notes without requiring code changes.
