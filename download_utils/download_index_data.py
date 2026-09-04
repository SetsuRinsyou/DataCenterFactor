#!/usr/bin/env python3
"""Download daily index data into the project DuckDB database."""

import time
from pathlib import Path

import duckdb
import pandas as pd
import tushare as ts

from download_utils.download_market_data import (
    MAX_RETRIES,
    REQUEST_INTERVAL_SECONDS,
    TUSHARE_API_KEY,
)


DATABASE_PATH = Path(__file__).resolve().parents[1] / "data" / "src_data.duckdb"
INDEX_CODES = ["000905.SH"]

DAILY_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "change",
    "pct_chg",
    "vol",
    "amount",
]
DAILY_BASIC_COLUMNS = [
    "total_mv",
    "float_mv",
    "total_share",
    "float_share",
    "free_share",
    "turnover_rate",
    "turnover_rate_f",
    "pe",
    "pe_ttm",
    "pb",
]
INDEX_COLUMNS = ["trade_date", "ts_code", *DAILY_COLUMNS, *DAILY_BASIC_COLUMNS]


def create_index_table(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS "index" (
            trade_date VARCHAR NOT NULL,
            ts_code VARCHAR NOT NULL,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            pre_close DOUBLE,
            change DOUBLE,
            pct_chg DOUBLE,
            vol DOUBLE,
            amount DOUBLE,
            total_mv DOUBLE,
            float_mv DOUBLE,
            total_share DOUBLE,
            float_share DOUBLE,
            free_share DOUBLE,
            turnover_rate DOUBLE,
            turnover_rate_f DOUBLE,
            pe DOUBLE,
            pe_ttm DOUBLE,
            pb DOUBLE,
            PRIMARY KEY (trade_date, ts_code)
        )
        """
    )

    columns = [
        row[1] for row in connection.execute("PRAGMA table_info('index')").fetchall()
    ]
    if columns != INDEX_COLUMNS:
        raise RuntimeError(
            f'Existing "index" table has an unexpected schema: {columns}'
        )


def market_date_range(connection: duckdb.DuckDBPyConnection) -> tuple[str, str]:
    start_date, end_date = connection.execute(
        "SELECT MIN(trade_date), MAX(trade_date) FROM market"
    ).fetchone()
    if not start_date or not end_date:
        raise RuntimeError("market is empty")
    return start_date, end_date


def date_chunks(start_date: str, end_date: str):
    for first_year in range(int(start_date[:4]), int(end_date[:4]) + 1, 5):
        yield max(start_date, f"{first_year}0101"), min(
            end_date, f"{first_year + 4}1231"
        )


def request(pro, endpoint: str, **kwargs) -> pd.DataFrame:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            frame = getattr(pro, endpoint)(**kwargs)
            time.sleep(REQUEST_INTERVAL_SECONDS)
            return frame if frame is not None else pd.DataFrame()
        except Exception as exc:
            if attempt == MAX_RETRIES:
                raise RuntimeError(
                    f"Failed to call {endpoint} with {kwargs}"
                ) from exc
            time.sleep(attempt * 2)
    raise AssertionError("unreachable")


def fetch_index(
    pro, ts_code: str, start_date: str, end_date: str
) -> pd.DataFrame:
    daily_frames = []
    basic_frames = []
    for period_start, period_end in date_chunks(start_date, end_date):
        daily = request(
            pro,
            "index_daily",
            ts_code=ts_code,
            start_date=period_start,
            end_date=period_end,
            fields=",".join(["ts_code", "trade_date", *DAILY_COLUMNS]),
        )
        basic = request(
            pro,
            "index_dailybasic",
            ts_code=ts_code,
            start_date=period_start,
            end_date=period_end,
            fields=",".join(["ts_code", "trade_date", *DAILY_BASIC_COLUMNS]),
        )
        if not daily.empty:
            daily_frames.append(daily)
        if not basic.empty:
            basic_frames.append(basic)
        print(
            f"{ts_code} {period_start}-{period_end}: "
            f"{len(daily):,} daily, {len(basic):,} daily-basic"
        )

    if not daily_frames:
        raise RuntimeError(f"Tushare returned no index_daily rows for {ts_code}")

    keys = ["trade_date", "ts_code"]
    daily = pd.concat(daily_frames, ignore_index=True)[keys + DAILY_COLUMNS]
    if basic_frames:
        basic = pd.concat(basic_frames, ignore_index=True)[keys + DAILY_BASIC_COLUMNS]
    else:
        basic = pd.DataFrame(columns=keys + DAILY_BASIC_COLUMNS)

    for name, frame in (("index_daily", daily), ("index_dailybasic", basic)):
        if frame.duplicated(keys).any():
            raise RuntimeError(f"{name} returned duplicate keys for {ts_code}")

    result = daily.merge(basic, on=keys, how="outer", validate="one_to_one")
    result["trade_date"] = result["trade_date"].astype(str)
    result["ts_code"] = result["ts_code"].astype(str)
    for column in DAILY_COLUMNS + DAILY_BASIC_COLUMNS:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    if set(result["ts_code"]) != {ts_code}:
        raise RuntimeError(f"Tushare returned an unexpected index code for {ts_code}")
    if result["trade_date"].min() < start_date or result["trade_date"].max() > end_date:
        raise RuntimeError(f"Tushare returned dates outside the requested range for {ts_code}")
    return result[INDEX_COLUMNS].sort_values(keys, ignore_index=True)


def main() -> None:
    if not TUSHARE_API_KEY.strip():
        raise SystemExit("Please set the TUSHARE_API_KEY environment variable")
    if not DATABASE_PATH.exists():
        raise SystemExit(f"Database does not exist: {DATABASE_PATH}")

    connection = duckdb.connect(str(DATABASE_PATH))
    try:
        start_date, end_date = market_date_range(connection)
        pro = ts.pro_api(TUSHARE_API_KEY.strip())
        frames = [
            fetch_index(pro, ts_code, start_date, end_date)
            for ts_code in INDEX_CODES
        ]
        result = pd.concat(frames, ignore_index=True)
        if result.duplicated(["trade_date", "ts_code"]).any():
            raise RuntimeError("Downloaded index data contains duplicate primary keys")

        connection.register("index_batch", result)
        try:
            connection.execute("BEGIN TRANSACTION")
            create_index_table(connection)
            connection.execute(
                'DELETE FROM "index" WHERE ts_code IN (SELECT DISTINCT ts_code FROM index_batch) '
                "AND trade_date BETWEEN ? AND ?",
                [start_date, end_date],
            )
            connection.execute(
                f'INSERT INTO "index" ({", ".join(INDEX_COLUMNS)}) '
                f'SELECT {", ".join(INDEX_COLUMNS)} FROM index_batch'
            )
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.unregister("index_batch")

        connection.execute("CHECKPOINT")
        row_count, first_date, last_date = connection.execute(
            'SELECT COUNT(*), MIN(trade_date), MAX(trade_date) FROM "index" '
            "WHERE ts_code IN (SELECT UNNEST(?::VARCHAR[]))",
            [INDEX_CODES],
        ).fetchone()
        print(
            f'Done. "index" contains {row_count:,} rows for '
            f"{len(INDEX_CODES):,} index code(s), {first_date}-{last_date}."
        )
    finally:
        connection.close()


if __name__ == "__main__":
    main()
