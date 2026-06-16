# Streamlit Secondary Market Workstation

This repository is the Streamlit-focused trading analysis workstation for municipal secondary-market review. It keeps the workflow centered on uploaded trade files, interactive trading filters, liquidity analysis, CUSIP drilldown, peer comparison, and reportable analyst output.

The goal is to make the Streamlit version accurate, visual, and analyst-friendly before deciding what should ever be productized elsewhere.

## Repository Scope

Included here:

- `streamlit_app.py`
- `streamlit_app_duckdb.py`
- `data_utils.py`
- `nextsr_payload.py`
- `requirements.txt`
- `scripts/`
- `data/`

Not included here:

- `nextsr-web/`
- Next.js / Vercel configuration
- Browser-first product UI work

The current priority is improving the Streamlit analysis product, methodology accuracy, and visual workflow before revisiting any separate web deployment.

## Run Locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

DuckDB variant:

```bash
streamlit run streamlit_app_duckdb.py
```

## Input Files

Required or primary:

- MuniPro trade-history export, preferably `.xlsx`

Optional:

- Bond master / reference file
- Issuer or sector mapping file

## Current Product Direction

The Streamlit workstation should focus on:

1. Issuer Selection
2. Trading Filters
3. Market Analytics
4. Security Drilldown
5. Peer Comparison
6. Narrative Insights
7. Export

Use `LADWP.xlsx` as the working golden trade sample when validating methodology and UI changes. Benchmark fields from the trade sheet can still support spread calculations, but MMD file management is no longer part of the default user workflow.

## UI Direction

The default experience is intentionally focused:

- Keep the Trading Workbench sections in the main page as the daily path.
- Use short indications instead of long explanatory copy.
- Put charts before raw tables wherever possible.
- Show conclusions, next actions, and key metrics before data tables.
- Keep detailed audit tables, methodology evidence, and large ranking tables inside expanders.
- Render only the chart modules the analyst selects.
- Keep legacy methodology and export code available internally, but do not make it the default reading path.

## Analyst Validation Workflow

The current review path is:

1. Upload trade files.
2. Select sector, issuer, and date range.
3. Apply maturity, trade-size, trade-type, and lot/block filters.
4. Validate market analytics: volume overview, activity heatmap, participation, and liquidity.
5. Review the CUSIP-level security drilldown.
6. Compare peer issuers under the same filters.
7. Export filtered trades, security drilldown, peer metrics, Excel workbook, or PDF summary.

More detail is in `docs/analyst_review_playbook.md`.

For external or professional review sessions, use `docs/reviewer_handoff_pack.md`.

## Golden Regression Check

Run the LADWP golden-sample check after methodology or UI changes:

```bash
python scripts/regression_check.py
```

The script uses the local trusted LADWP workbook when it exists, falls back to repo sample files when it does not, compiles project files, runs core analytics, checks exports, and performs a Streamlit startup smoke test. The default LADWP expected values live in `data/golden/ladwp_expected.json` and are only applied automatically when the selected issuer matches `LADWP`.

Useful variants:

```bash
python scripts/regression_check.py --skip-streamlit
python scripts/regression_check.py --trade-file /path/to/LADWP.xlsx --mmd-file /path/to/mmd.csv
python scripts/regression_check.py --expected-file data/golden/ladwp_expected.json
```

## Reports And Review Outputs

The Trading Workbench export section can generate:

- Filtered trades CSV
- Security drilldown CSV
- Peer comparison CSV inside the Excel workbook
- Excel workbook
- PDF summary

PDF requires `reportlab`; Excel requires `openpyxl`. Both are included in `requirements.txt`.

## NextSR Payload Utility

Generate a stable JSON payload from a MuniPro trade file without opening the dashboard:

```bash
python scripts/build_nextsr_payload.py data/processed/Trade_Output_Sample.csv \
  --output nextsr_payload.json
```

The payload contract is versioned as `nextsr_payload.v1` and includes issuer, maturity bucket, benchmark source, spread signals, liquidity signals, flow signals, a rule label, and evidence bullets.

## Privacy / Data Note

Do not commit real MuniPro or proprietary trade exports to public GitHub. This app is designed for users to upload their own authorized files during their own session.

See `docs/data_safety_and_deployment.md` for deployment and data-safety checks before sharing the app with outside reviewers.
