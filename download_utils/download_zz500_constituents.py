#!/usr/bin/env python3
"""Download historical CSI 500 constituents and weights from Tushare."""

import argparse
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path


TUSHARE_API_KEY = os.environ.get("TUSHARE_API_KEY", "")
INDEX_CODE = "000905.SH"
DEFAULT_START_DATE = "20050101"
DEFAULT_OUTPUT = "zz500_historical_constituents.csv"
REQUEST_INTERVAL_SECONDS = 0.5


def parse_date(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y%m%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid date {value!r}; expected YYYYMMDD") from exc


def iter_calendar_months(start: date, end: date):
    current = start.replace(day=1)
    while current <= end:
        if current.month == 12:
            next_month = current.replace(year=current.year + 1, month=1)
        else:
            next_month = current.replace(month=current.month + 1)
        yield current, next_month - timedelta(days=1)
        current = next_month


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download monthly historical constituents and weights of CSI 500."
    )
    parser.add_argument("--start-date", type=parse_date, default=parse_date(DEFAULT_START_DATE))
    parser.add_argument("--end-date", type=parse_date, default=date.today())
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    args = parser.parse_args()

    if not TUSHARE_API_KEY.strip():
        parser.error("Please set the TUSHARE_API_KEY environment variable")
    if args.start_date > args.end_date:
        parser.error("--start-date must not be later than --end-date")

    try:
        import pandas as pd
        import tushare as ts
    except ModuleNotFoundError as exc:
        parser.error(f"Missing dependency {exc.name!r}; install pandas and tushare first")

    pro = ts.pro_api(TUSHARE_API_KEY.strip())
    frames = []

    for month_start, month_end in iter_calendar_months(args.start_date, args.end_date):
        query_end = min(month_end, date.today())
        if month_start > query_end:
            continue

        print(f"Downloading {month_start:%Y-%m} ...")
        frame = pro.index_weight(
            index_code=INDEX_CODE,
            start_date=month_start.strftime("%Y%m%d"),
            end_date=query_end.strftime("%Y%m%d"),
            fields="index_code,con_code,trade_date,weight",
        )
        if not frame.empty:
            frames.append(frame)
        time.sleep(REQUEST_INTERVAL_SECONDS)

    if not frames:
        raise RuntimeError("Tushare returned no CSI 500 constituent data for the requested period")

    result = pd.concat(frames, ignore_index=True)
    result["trade_date"] = result["trade_date"].astype(str)
    start_text = args.start_date.strftime("%Y%m%d")
    end_text = args.end_date.strftime("%Y%m%d")
    result = result[result["trade_date"].between(start_text, end_text)]
    result = result.drop_duplicates(["index_code", "con_code", "trade_date"])
    result = result.sort_values(["trade_date", "con_code"], ignore_index=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Saved {len(result):,} rows to {args.output}")


if __name__ == "__main__":
    main()
