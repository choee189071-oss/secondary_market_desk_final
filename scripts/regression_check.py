from __future__ import annotations

import argparse
import io
import json
import py_compile
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_utils import read_uploaded_file
from engine.benchmark import (
    build_issuer_curve_snapshot,
    build_spread_level_data,
    build_spread_movement_ladder_data,
    build_spread_observations,
    make_benchmark_long,
)
from engine.load_data import process_uploads
from engine.scoring import (
    add_workflow_spread_bps,
    build_workflow_cusip_summary,
    focused_summary_with_peer_gaps,
    workflow_date_range_text,
)
from engine.validation import (
    MMD_RECOMMENDED,
    MMD_REQUIRED,
    TRADE_OPTIONAL,
    TRADE_RECOMMENDED,
    TRADE_REQUIRED,
    validate_basic_values,
    validate_dataset,
)
from reports.export_center import (
    focused_core_chart_explanations,
    focused_methodology_appendix,
    focused_report_bundle_bytes,
    focused_report_html,
    focused_report_markdown,
    focused_report_markdown_table,
    focused_report_pdf_bytes,
    focused_report_pptx_bytes,
)


LOCAL_LADWP_TRADE = Path("/Users/zhouyiyi/Desktop/Intern_Muni_Data/Secondary/LADWP/2024-26/LADWP.xlsx")
LOCAL_LADWP_MMD = Path("/Users/zhouyiyi/Desktop/Intern_Muni_Data/Secondary/LADWP/2024-26/mmd.csv")
PROJECT_SAMPLE_TRADE = REPO_ROOT / "data" / "processed" / "Trade_Output_Sample.csv"
PROJECT_SAMPLE_MMD = REPO_ROOT / "data" / "processed" / "mmd.csv"

COMPILE_TARGETS = [
    "streamlit_app.py",
    "streamlit_app_duckdb.py",
    "data_utils.py",
    "nextsr_payload.py",
    "app_state.py",
    "engine/__init__.py",
    "engine/normalize.py",
    "engine/scoring.py",
    "engine/benchmark.py",
    "engine/validation.py",
    "engine/load_data.py",
    "reports/__init__.py",
    "reports/export_center.py",
    "ui/__init__.py",
    "ui/common.py",
    "ui/charts.py",
    "ui/cusip_detail.py",
    "ui/export_center.py",
    "ui/upload.py",
    "ui/snapshot.py",
    "scripts/build_duckdb.py",
    "scripts/build_nextsr_payload.py",
    "scripts/regression_check.py",
]


def first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def add_check(checks: list[dict], name: str, condition: object, detail: object = "") -> None:
    checks.append(
        {
            "check": name,
            "status": "PASS" if bool(condition) else "FAIL",
            "detail": str(detail),
        }
    )


def compile_project() -> tuple[bool, list[str]]:
    errors: list[str] = []
    for relative_path in COMPILE_TARGETS:
        path = REPO_ROOT / relative_path
        if not path.exists():
            errors.append(f"Missing compile target: {relative_path}")
            continue
        try:
            py_compile.compile(str(path), doraise=True)
        except Exception as exc:
            errors.append(f"{relative_path}: {exc}")
    return len(errors) == 0, errors


def wait_for_url(url: str, timeout_seconds: float) -> tuple[bool, str]:
    deadline = time.time() + timeout_seconds
    last_error = ""
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                body = response.read(500).decode("utf-8", errors="replace")
                return response.status == 200, body
        except Exception as exc:
            last_error = str(exc)
            time.sleep(0.5)
    return False, last_error


def streamlit_smoke(port: int, timeout_seconds: float) -> dict:
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "streamlit_app.py",
        "--global.developmentMode",
        "false",
        "--server.headless",
        "true",
        "--server.port",
        str(port),
        "--server.enableCORS",
        "false",
        "--server.enableXsrfProtection",
        "false",
    ]
    process = None
    try:
        process = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        health_ok, health_detail = wait_for_url(f"http://127.0.0.1:{port}/_stcore/health", timeout_seconds)
        root_ok = False
        root_detail = ""
        if health_ok:
            root_ok, root_detail = wait_for_url(f"http://127.0.0.1:{port}/", 8)
        return {
            "status": "PASS" if health_ok and root_ok else "FAIL",
            "health_ok": health_ok,
            "root_ok": root_ok,
            "detail": f"health={health_detail[:120]}; root={root_detail[:120]}",
        }
    except Exception as exc:
        return {"status": "FAIL", "health_ok": False, "root_ok": False, "detail": str(exc)}
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()


def build_regression_context(
    trade_path: Path,
    mmd_path: Path | None,
    issuer_override: str | None,
    ratings: list[str],
    curve_lookback_days: int,
) -> tuple[dict, dict]:
    trade_bytes = trade_path.read_bytes()
    raw_trade = read_uploaded_file(io.BytesIO(trade_bytes), trade_path.name)
    trade_report = validate_dataset(raw_trade, trade_path.name, TRADE_REQUIRED, TRADE_RECOMMENDED, TRADE_OPTIONAL)
    trade_warnings = validate_basic_values(raw_trade, trade_report["mapping"], dataset_type="trade")

    mmd_payload = None
    raw_mmd = pd.DataFrame()
    mmd_report = None
    if mmd_path is not None and mmd_path.exists():
        mmd_bytes = mmd_path.read_bytes()
        raw_mmd = read_uploaded_file(io.BytesIO(mmd_bytes), mmd_path.name)
        mmd_report = validate_dataset(raw_mmd, mmd_path.name, MMD_REQUIRED, MMD_RECOMMENDED, [])
        mmd_payload = (mmd_path.name, mmd_bytes)

    bonds_df, trades_df, issuer_master, market_df, mmd_df, failed_files, duplicates_removed = process_uploads(
        trade_payloads=[(trade_path.name, trade_bytes)],
        issuer_mapping_payload=None,
        mmd_payload=mmd_payload,
        bond_payload=None,
    )

    benchmark_source_mode = mmd_df.attrs.get("benchmark_source_mode", "Unknown")
    benchmark_priority = mmd_df.attrs.get("benchmark_source_priority", "Unknown")
    benchmark_conflict_policy = mmd_df.attrs.get("benchmark_conflict_policy", "Unknown")

    issuers = sorted(market_df["issuer"].dropna().astype(str).unique().tolist()) if "issuer" in market_df.columns else []
    selected_issuer = issuer_override or ("LADWP" if "LADWP" in issuers else (issuers[0] if issuers else "Unknown"))
    issuer_trades = market_df[market_df["issuer"].astype(str) == selected_issuer].copy() if "issuer" in market_df.columns else market_df.copy()
    issuer_base = add_workflow_spread_bps(issuer_trades)
    cusip_summary = focused_summary_with_peer_gaps(build_workflow_cusip_summary(issuer_base))

    spread_series = (
        pd.to_numeric(issuer_base.get("spread_bps"), errors="coerce")
        if "spread_bps" in issuer_base.columns
        else pd.Series(dtype="float64")
    )
    liquidity_series = (
        pd.to_numeric(cusip_summary.get("liquidity_score"), errors="coerce")
        if "liquidity_score" in cusip_summary.columns
        else pd.Series(dtype="float64")
    )
    top_opportunities = cusip_summary.head(5).copy()
    saved_watchlist = top_opportunities.head(2).copy()
    if not saved_watchlist.empty:
        saved_watchlist["note"] = ["Regression saved candidate"] * len(saved_watchlist)

    active_ratings = ratings or ["AAA", "AA"]
    benchmark_long = make_benchmark_long(mmd_df, active_ratings[0]) if not mmd_df.empty else pd.DataFrame()
    spread_obs = build_spread_observations(market_df, mmd_df, selected_issuer, active_ratings[0]) if not mmd_df.empty else pd.DataFrame()
    movement_ladder, _movement_audit = build_spread_movement_ladder_data(spread_obs)
    level_ladder, _level_audit = build_spread_level_data(market_df, mmd_df, selected_issuer, active_ratings)

    as_of_date = pd.to_datetime(issuer_base.get("trade_date"), errors="coerce").dropna().max()
    if pd.isna(as_of_date):
        as_of_date = pd.Timestamp.today().normalize()
    curve_snapshot, _curve_audit = build_issuer_curve_snapshot(
        market_df=market_df,
        mmd_df=mmd_df,
        issuer=selected_issuer,
        ratings=active_ratings,
        as_of_date=as_of_date,
        lookback_days=curve_lookback_days,
        aggregation_method=f"Average last {curve_lookback_days} days",
    )

    metrics = pd.DataFrame(
        [
            {"Metric": "Issuer", "Value": selected_issuer},
            {"Metric": "Trade date range", "Value": workflow_date_range_text(issuer_base)},
            {"Metric": "Trade rows", "Value": f"{len(issuer_base):,}"},
            {"Metric": "CUSIPs", "Value": f"{issuer_base['cusip'].nunique() if 'cusip' in issuer_base.columns else 0:,}"},
            {"Metric": "Median spread", "Value": "N/A" if spread_series.dropna().empty else f"{spread_series.median():.1f} bps"},
            {"Metric": "Median liquidity", "Value": "N/A" if liquidity_series.dropna().empty else f"{liquidity_series.median():.1f}"},
            {"Metric": "Top candidate", "Value": "N/A" if top_opportunities.empty else str(top_opportunities.iloc[0].get("cusip", "N/A"))},
            {"Metric": "Benchmark source", "Value": benchmark_source_mode},
        ]
    )

    warning_rows = pd.DataFrame(
        [
            {
                "Area": "Trade schema",
                "Status": "Pass" if trade_report["can_run"] else "Fail",
                "Value": trade_path.name,
                "Detail": ", ".join(trade_report["missing_required"]) or "All required fields detected",
            },
            {
                "Area": "MMD schema",
                "Status": "Pass" if (mmd_report or {"can_run": False})["can_run"] else "Fail",
                "Value": mmd_path.name if mmd_path else "None",
                "Detail": ", ".join((mmd_report or {"missing_required": ["missing mmd"]})["missing_required"]) or "Date field detected",
            },
            {
                "Area": "Benchmark policy",
                "Status": "Pass" if benchmark_source_mode in {"Trade Sheet Index / Index Rate", "Uploaded MMD fallback"} else "Warn",
                "Value": benchmark_source_mode,
                "Detail": benchmark_conflict_policy,
            },
        ]
    )

    takeaway_bullets = [
        f"{selected_issuer} loaded with {len(issuer_base):,} trade row(s) across {issuer_base['cusip'].nunique() if 'cusip' in issuer_base.columns else 0:,} CUSIP(s).",
        "Median spread is unavailable." if spread_series.dropna().empty else f"Median spread is {spread_series.median():.1f} bps.",
        "CUSIP liquidity scoring is unavailable."
        if liquidity_series.dropna().empty
        else f"Median liquidity score is {liquidity_series.median():.1f}; top score is {liquidity_series.max():.1f}.",
        f"Benchmark source: {benchmark_source_mode}; policy: {benchmark_conflict_policy}.",
    ]

    context = {
        "title": f"{selected_issuer} Regression Report",
        "prepared_for": "Regression smoke test",
        "analyst_note": "Generated by scripts/regression_check.py.",
        "generated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "issuer": selected_issuer,
        "sector": "Unknown",
        "metrics": metrics,
        "takeaway_bullets": takeaway_bullets,
        "takeaway_labels": {},
        "top_opportunities": top_opportunities,
        "saved_watchlist": saved_watchlist,
        "methodology": focused_methodology_appendix(mmd_df, benchmark_source_mode, benchmark_priority, benchmark_conflict_policy),
        "chart_explanations": focused_core_chart_explanations(selected_issuer, "Unknown", benchmark_source_mode),
        "warning_rows": warning_rows,
        "cusip_summary": cusip_summary,
        "benchmark_source_mode": benchmark_source_mode,
        "benchmark_priority": benchmark_priority,
        "benchmark_conflict_policy": benchmark_conflict_policy,
        "top_candidate_note": "",
    }

    diagnostics = {
        "trade_path": str(trade_path),
        "mmd_path": str(mmd_path) if mmd_path else None,
        "raw_trade_rows": len(raw_trade),
        "raw_mmd_rows": len(raw_mmd),
        "market_rows": len(market_df),
        "trades_rows": len(trades_df),
        "bonds_rows": len(bonds_df),
        "issuer_master_rows": len(issuer_master),
        "benchmark_rows": len(mmd_df),
        "duplicates_removed": int(duplicates_removed),
        "failed_files": failed_files,
        "selected_issuer": selected_issuer,
        "issuer_values": issuers,
        "date_range": workflow_date_range_text(issuer_base),
        "cusip_count": int(issuer_base["cusip"].nunique()) if "cusip" in issuer_base.columns else 0,
        "median_spread_bps": None if spread_series.dropna().empty else float(spread_series.median()),
        "median_liquidity": None if liquidity_series.dropna().empty else float(liquidity_series.median()),
        "benchmark_source_mode": benchmark_source_mode,
        "benchmark_priority": benchmark_priority,
        "benchmark_conflict_policy": benchmark_conflict_policy,
        "trade_report": trade_report,
        "trade_warnings": trade_warnings[:20],
        "mmd_report": mmd_report,
        "top_opportunities": top_opportunities.head(5).to_dict("records"),
        "benchmark_long_rows": len(benchmark_long),
        "spread_observation_rows": len(spread_obs),
        "movement_ladder_shape": list(movement_ladder.shape),
        "level_ladder_shape": list(level_ladder.shape),
        "curve_snapshot_rows": len(curve_snapshot),
    }
    return context, diagnostics


def write_report(output_dir: Path, context: dict, diagnostics: dict, checks: list[dict]) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    markdown = focused_report_markdown(context)
    html_report = focused_report_html(context)
    bundle = focused_report_bundle_bytes(context, markdown, html_report)
    pdf_bytes, pdf_error = focused_report_pdf_bytes(context)
    pptx_bytes, pptx_error = focused_report_pptx_bytes(context)

    add_check(checks, "Markdown report generated", len(markdown) > 1000, f"chars={len(markdown):,}")
    add_check(checks, "HTML report generated", len(html_report) > 1000, f"chars={len(html_report):,}")
    add_check(checks, "Report bundle generated", len(bundle) > 1000, f"bytes={len(bundle):,}")
    add_check(checks, "PDF helper available", pdf_bytes is not None and len(pdf_bytes) > 1000, pdf_error or f"bytes={len(pdf_bytes) if pdf_bytes else 0:,}")
    add_check(checks, "PPTX helper available", pptx_bytes is not None and len(pptx_bytes) > 1000, pptx_error or f"bytes={len(pptx_bytes) if pptx_bytes else 0:,}")

    markdown_path = output_dir / "regression_report.md"
    html_path = output_dir / "regression_report.html"
    bundle_path = output_dir / "regression_bundle.zip"
    pdf_path = output_dir / "regression_report.pdf"
    pptx_path = output_dir / "regression_report.pptx"
    json_path = output_dir / "regression_report.json"

    summary_lines = [
        "# Secondary Market Regression Report",
        "",
        f"Generated: {pd.Timestamp.now():%Y-%m-%d %H:%M}",
        "",
        "## Inputs",
        f"- Trade file: `{diagnostics['trade_path']}`",
        f"- MMD file: `{diagnostics.get('mmd_path')}`",
        "",
        "## Summary",
        f"- Selected issuer: `{diagnostics['selected_issuer']}`",
        f"- Market rows: `{diagnostics['market_rows']:,}`",
        f"- Raw trade rows: `{diagnostics['raw_trade_rows']:,}`",
        f"- CUSIPs: `{diagnostics['cusip_count']:,}`",
        f"- Date range: `{diagnostics['date_range']}`",
        f"- Median spread: `{diagnostics['median_spread_bps']}` bps",
        f"- Median liquidity: `{diagnostics['median_liquidity']}`",
        f"- Benchmark source: `{diagnostics['benchmark_source_mode']}`",
        f"- Benchmark policy: `{diagnostics['benchmark_conflict_policy']}`",
        "",
        "## Checks",
    ]
    for row in checks:
        summary_lines.append(f"- {row['status']}: {row['check']} - {row['detail']}")
    top = context["top_opportunities"]
    display_cols = [
        c
        for c in [
            "cusip",
            "signal",
            "maturity_bucket",
            "current_spread_bps",
            "peer_median_gap_bps",
            "liquidity_score",
            "rv_score",
            "trade_count",
            "total_trade_amount",
            "latest_trade",
        ]
        if c in top.columns
    ]
    summary_lines.extend(["", "## Top Opportunities"])
    summary_lines.append(focused_report_markdown_table(top[display_cols].head(5) if display_cols else top.head(5), max_rows=5))
    summary_lines.extend(
        [
            "",
            "## Export Smoke",
            f"- Markdown chars: `{len(markdown):,}`",
            f"- HTML chars: `{len(html_report):,}`",
            f"- Bundle bytes: `{len(bundle):,}`",
            f"- PDF bytes: `{0 if pdf_bytes is None else len(pdf_bytes):,}`",
            f"- PPTX bytes: `{0 if pptx_bytes is None else len(pptx_bytes):,}`",
        ]
    )

    markdown_path.write_text("\n".join(summary_lines), encoding="utf-8")
    html_path.write_text(html_report, encoding="utf-8")
    bundle_path.write_bytes(bundle)
    if pdf_bytes:
        pdf_path.write_bytes(pdf_bytes)
    if pptx_bytes:
        pptx_path.write_bytes(pptx_bytes)

    payload = {
        **diagnostics,
        "checks": checks,
        "markdown_chars": len(markdown),
        "html_chars": len(html_report),
        "bundle_bytes": len(bundle),
        "pdf_bytes": 0 if pdf_bytes is None else len(pdf_bytes),
        "pptx_bytes": 0 if pptx_bytes is None else len(pptx_bytes),
        "pdf_error": pdf_error,
        "pptx_error": pptx_error,
        "output_files": {
            "markdown": str(markdown_path),
            "html": str(html_path),
            "bundle": str(bundle_path),
            "pdf": str(pdf_path) if pdf_bytes else None,
            "pptx": str(pptx_path) if pptx_bytes else None,
            "json": str(json_path),
        },
    }
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the secondary market golden sample regression check.")
    parser.add_argument("--trade-file", type=Path, help="CSV/XLS/XLSX trade file. Defaults to local LADWP sample, then repo sample.")
    parser.add_argument("--mmd-file", type=Path, help="CSV/XLS/XLSX MMD file. Defaults to local LADWP MMD, then repo sample MMD.")
    parser.add_argument("--issuer", help="Issuer to validate. Defaults to LADWP when present.")
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/secondary_market_regression"), help="Directory for Markdown/JSON/export artifacts.")
    parser.add_argument("--ratings", nargs="+", default=["AAA", "AA"], help="Benchmark ratings used by regression checks.")
    parser.add_argument("--curve-lookback-days", type=int, default=60)
    parser.add_argument("--skip-compile", action="store_true", help="Skip py_compile checks.")
    parser.add_argument("--skip-streamlit", action="store_true", help="Skip Streamlit startup smoke.")
    parser.add_argument("--streamlit-port", type=int, default=8521)
    parser.add_argument("--streamlit-timeout", type=float, default=25)
    args = parser.parse_args()

    trade_path = args.trade_file or first_existing(LOCAL_LADWP_TRADE, PROJECT_SAMPLE_TRADE)
    mmd_path = args.mmd_file or first_existing(LOCAL_LADWP_MMD, PROJECT_SAMPLE_MMD)
    checks: list[dict] = []

    if not args.skip_compile:
        compile_ok, compile_errors = compile_project()
        add_check(checks, "Python files compile", compile_ok, "; ".join(compile_errors) or "all compile targets passed")

    add_check(checks, "Trade file exists", trade_path.exists(), trade_path)
    add_check(checks, "MMD file exists", mmd_path.exists(), mmd_path)
    if not trade_path.exists():
        raise SystemExit(f"Trade file not found: {trade_path}")

    context, diagnostics = build_regression_context(
        trade_path=trade_path,
        mmd_path=mmd_path if mmd_path.exists() else None,
        issuer_override=args.issuer,
        ratings=args.ratings,
        curve_lookback_days=args.curve_lookback_days,
    )

    trade_report = diagnostics["trade_report"]
    mmd_report = diagnostics["mmd_report"] or {"can_run": False, "missing_required": ["missing mmd"]}
    add_check(checks, "Trade readiness can_run", trade_report["can_run"], f"missing={trade_report['missing_required']}")
    add_check(checks, "MMD readiness can_run", mmd_report["can_run"], f"missing={mmd_report['missing_required']}")
    add_check(checks, "process_uploads returned market rows", diagnostics["market_rows"] > 0, f"rows={diagnostics['market_rows']:,}; failed={diagnostics['failed_files']}")
    add_check(checks, "No failed files", not diagnostics["failed_files"], diagnostics["failed_files"])
    add_check(checks, "Issuer detected", diagnostics["selected_issuer"] != "Unknown", diagnostics["issuer_values"][:5])
    add_check(checks, "CUSIP summary generated", len(context["cusip_summary"]) > 0, f"rows={len(context['cusip_summary']):,}")
    add_check(checks, "Top opportunities generated", len(context["top_opportunities"]) >= 1, context["top_opportunities"].head(1).to_dict("records"))
    add_check(checks, "Benchmark source resolved", diagnostics["benchmark_source_mode"] in {"Trade Sheet Index / Index Rate", "Uploaded MMD fallback"}, diagnostics["benchmark_source_mode"])
    add_check(checks, "Benchmark long generated", diagnostics["benchmark_long_rows"] > 0, f"rows={diagnostics['benchmark_long_rows']:,}")
    add_check(checks, "Spread observations generated", diagnostics["spread_observation_rows"] > 0, f"rows={diagnostics['spread_observation_rows']:,}")
    add_check(checks, "Movement ladder generated", diagnostics["movement_ladder_shape"][0] > 0, f"shape={diagnostics['movement_ladder_shape']}")
    add_check(checks, "Level ladder generated", diagnostics["level_ladder_shape"][0] > 0, f"shape={diagnostics['level_ladder_shape']}")
    add_check(checks, "Issuer curve snapshot generated", diagnostics["curve_snapshot_rows"] > 0, f"rows={diagnostics['curve_snapshot_rows']:,}")

    if not args.skip_streamlit:
        smoke = streamlit_smoke(args.streamlit_port, args.streamlit_timeout)
        add_check(checks, "Streamlit startup smoke", smoke["status"] == "PASS", smoke["detail"])

    payload = write_report(args.output_dir, context, diagnostics, checks)
    failed = [row for row in payload["checks"] if row["status"] != "PASS"]

    print(f"Regression checks: {len(payload['checks']) - len(failed)}/{len(payload['checks'])} PASS")
    print(f"Issuer: {payload['selected_issuer']}")
    print(f"Market rows: {payload['market_rows']:,}")
    print(f"CUSIPs: {payload['cusip_count']:,}")
    print(f"Benchmark: {payload['benchmark_source_mode']}")
    print(f"Top CUSIP: {payload['top_opportunities'][0]['cusip'] if payload['top_opportunities'] else 'N/A'}")
    print(f"Report JSON: {payload['output_files']['json']}")
    print(f"Report Markdown: {payload['output_files']['markdown']}")

    if failed:
        print("Failed checks:")
        for row in failed:
            print(f"- {row['check']}: {row['detail']}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
