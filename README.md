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

## NextSR Payload Utility

Generate a stable JSON payload from a MuniPro trade file without opening the dashboard:

```bash
python scripts/build_nextsr_payload.py data/processed/Trade_Output_Sample.csv \
  --output nextsr_payload.json
```

The payload contract is versioned as `nextsr_payload.v1` and includes issuer, maturity bucket, benchmark source, spread signals, liquidity signals, flow signals, a rule label, and evidence bullets.

## Privacy / Data Note

Do not commit real MuniPro or proprietary trade exports to public GitHub. This app is designed for users to upload their own authorized files during their own session.
