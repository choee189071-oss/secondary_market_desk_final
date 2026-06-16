# Analyst Review Playbook

This workstation is designed to become analyst-validated before broader use. The review process should lock methodology first, then lock sample outputs.

## Review Order

1. Upload the trusted trade file.
2. Confirm required trade fields, date coverage, CUSIP quality, and trade amount scaling.
3. Select sector, issuer, and date range.
4. Apply maturity, trade-size, trade-type, and lot/block filters.
5. Review market analytics: volume overview, activity heatmap, participation, and liquidity.
6. Select the top CUSIP and review security drilldown, trade path, and source rows.
7. Compare peer issuers under the same trading filters.
8. Review narrative observations and confirm each one is supported by filtered data.
9. Export CSV/Excel/PDF outputs for review.
10. Update `data/golden/ladwp_expected.json` only after the analyst explicitly approves the expected values.

## Items To Lock

- Benchmark source and hierarchy.
- Trade row count after upload processing.
- CUSIP count and CUSIP quality.
- Date range.
- Trade amount scaling and trade-size buckets.
- Trade type classification.
- Maturity bucket boundaries.
- Top CUSIP and security drilldown metrics.
- Peer comparison metrics under equivalent filters.
- Report outputs: CSV, Excel, and PDF summary.

## Review Status Meaning

- `Correct`: analyst approves the current output.
- `Needs Review`: output may be reasonable but needs supporting evidence.
- `Wrong`: output should be corrected before product use.
- `Not Reviewed`: no analyst judgment has been captured.

## Expected Output Discipline

Expected values are not automatic truth. They are locked analyst decisions. If the code changes and regression fails, do not immediately update the JSON. First decide whether the code changed the methodology intentionally or exposed a bug.
