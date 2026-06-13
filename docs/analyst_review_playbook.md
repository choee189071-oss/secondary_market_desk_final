# Analyst Review Playbook

This workstation is designed to become analyst-validated before broader use. The review process should lock methodology first, then lock sample outputs.

## Review Order

1. Upload the trusted trade file and AAA MMD file.
2. Confirm Data Audit shows required trade fields, date coverage, CUSIP quality, and benchmark policy.
3. Review Desk Snapshot before charts.
4. Open Core Charts and confirm spread trend, volume, and issuer curve are directionally sensible.
5. Select the top CUSIP and review CUSIP Drilldown, trade path, same-bucket peers, and benchmark audit.
6. Save any candidate to the watchlist with analyst notes.
7. Open Export / Methodology, complete Analyst Review Mode, and download the review CSV/JSON.
8. Update `data/golden/ladwp_expected.json` only after the analyst explicitly approves the expected values.

## Items To Lock

- Benchmark source and hierarchy.
- Trade row count after upload processing.
- CUSIP count and CUSIP quality.
- Date range.
- Median spread and liquidity.
- Top CUSIP and top-candidate scoring.
- Benchmark audit logic for selected CUSIP.
- Report outputs: Markdown, HTML, PDF, PPTX, and bundle.

## Review Status Meaning

- `Correct`: analyst approves the current output.
- `Needs Review`: output may be reasonable but needs supporting evidence.
- `Wrong`: output should be corrected before product use.
- `Not Reviewed`: no analyst judgment has been captured.

## Expected Output Discipline

Expected values are not automatic truth. They are locked analyst decisions. If the code changes and regression fails, do not immediately update the JSON. First decide whether the code changed the methodology intentionally or exposed a bug.
