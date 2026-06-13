# Streamlit Secondary Market Workstation

This repository is the Streamlit-focused analysis workstation for municipal secondary-market review. It keeps the workflow centered on Python, Streamlit, uploaded trade files, benchmark curves, CUSIP drilldown, RV screening, and reportable analyst output.

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
- AAA MMD curve file, such as `mmd.csv`

Optional:

- Bond master / reference file
- Issuer or sector mapping file

## Current Product Direction

The Streamlit workstation should focus on:

1. Upload / Data Audit
2. Desk Snapshot
3. Core Charts
4. CUSIP Drilldown
5. RV / Watchlist
6. Export / Methodology

Use `LADWP.xlsx + mmd.csv` as the working golden sample when validating methodology and UI changes.

## Analyst Validation Workflow

The current review path is:

1. Upload trade files and AAA MMD.
2. Confirm Data Audit and Ready to Analyze.
3. Read Desk Snapshot before charts.
4. Validate Core Charts and CUSIP Drilldown.
5. Save watchlist candidates with notes.
6. Use Export / Methodology to review the Methodology Trust Layer, complete Analyst Review Mode, and download reports.

More detail is in `docs/analyst_review_playbook.md`.

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

The Export / Methodology page can generate:

- Markdown report
- Print HTML report
- ZIP report bundle
- Watchlist CSV
- PDF summary
- PPTX outline
- Analyst review CSV/JSON

PDF requires `reportlab`; PPTX requires `python-pptx`. Both are included in `requirements.txt`.

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
