from __future__ import annotations


PERFORMANCE_MODE = True
MAX_TABLE_ROWS = 3000
MAX_HEATMAP_ROWS = 18
SHOW_FULL_RAW_TABLES = False
ENABLE_REPORT_EXPORTS = False

TABLE_PREVIEW_ROWS = 500
LARGE_TABLE_ROW_THRESHOLD = 8
LARGE_TABLE_COL_THRESHOLD = 8

WORKFLOW_STEPS = [
    {
        "label": "1. Upload / Data Audit",
        "short": "Upload / Audit",
        "title": "Upload / Data Audit",
        "note": "Files, fields, row counts, benchmark source.",
    },
    {
        "label": "2. Desk Snapshot",
        "short": "Snapshot",
        "title": "Desk Snapshot",
        "note": "Issuer, dates, spread, liquidity, top signals.",
    },
    {
        "label": "3. Core Charts",
        "short": "Charts",
        "title": "Core Charts",
        "note": "Spread, yield, volume, issuer curve.",
    },
    {
        "label": "4. CUSIP Drilldown",
        "short": "CUSIP",
        "title": "CUSIP Drilldown",
        "note": "Security detail, trade path, same-bucket peers.",
    },
    {
        "label": "5. RV / Watchlist",
        "short": "RV / Watchlist",
        "title": "RV / Watchlist",
        "note": "Opportunity ranking, saved candidates.",
    },
    {
        "label": "6. Export / Methodology",
        "short": "Export",
        "title": "Export / Methodology",
        "note": "Reports, downloads, assumptions, benchmark audit.",
    },
]

WORKFLOW_LABELS = [step["label"] for step in WORKFLOW_STEPS]
FULL_DASHBOARD_LABEL = "Full Dashboard"
