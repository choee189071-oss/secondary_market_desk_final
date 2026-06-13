from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_utils import read_uploaded_file, standardize_trades
from nextsr_payload import build_nextsr_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a nextsr-compatible JSON payload from a MuniPro trade file.")
    parser.add_argument("trade_file", help="Path to CSV/XLS/XLSX trade-history file.")
    parser.add_argument("--issuer", help="Issuer name to include/filter in the payload.")
    parser.add_argument("--bucket", help="Maturity bucket such as 5Y, 10Y, or 25Y.")
    parser.add_argument("--period-days", type=int, default=30, help="Lookback window for movement/liquidity signals.")
    parser.add_argument("--output", default="nextsr_payload.json", help="Output JSON path.")
    args = parser.parse_args()

    trade_path = Path(args.trade_file)
    with trade_path.open("rb") as handle:
        raw_trades = read_uploaded_file(handle, trade_path.name)

    trades = standardize_trades(raw_trades, source_file=trade_path.name)
    payload = build_nextsr_payload(
        trades,
        issuer=args.issuer,
        maturity_bucket=args.bucket,
        period_days=args.period_days,
    )

    output_path = Path(args.output)
    output_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {output_path}")
    print(f"Issuer: {payload.get('issuer')}")
    print(f"Maturity bucket: {payload.get('maturity_bucket')}")
    print(f"Label: {payload.get('label')}")


if __name__ == "__main__":
    main()
