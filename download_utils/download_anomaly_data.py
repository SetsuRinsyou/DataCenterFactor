#!/usr/bin/env python3
"""Download ST, suspension, and limit-status data for all market stocks."""

import time
import duckdb
import pandas as pd
import tushare as ts

from download_utils.download_market_data import (
    DATABASE_PATH,
    MAX_RETRIES,
    REQUEST_INTERVAL_SECONDS,
    TUSHARE_API_KEY,
)

STATUS_ORDER = ["ST", "SUSPENDED", "LIMIT_UP", "LIMIT_DOWN"]


def create_anomaly_table(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS anomaly (
            trade_date VARCHAR NOT NULL,
            ts_code VARCHAR NOT NULL,
            value VARCHAR NOT NULL,
            PRIMARY KEY (trade_date, ts_code)
        )
        """
    )


def historical_universe(connection: duckdb.DuckDBPyConnection) -> tuple[set[str], str, str]:
    codes = {
        row[0]
        for row in connection.execute("SELECT DISTINCT ts_code FROM market").fetchall()
    }
    start_date, end_date = connection.execute(
        "SELECT MIN(trade_date), MAX(trade_date) FROM market"
    ).fetchone()
    if not codes or not start_date or not end_date:
        raise RuntimeError("market is empty")
    return codes, start_date, end_date


def fetch_daily_source(
    pro,
    endpoint: str,
    trade_date: str,
    fields: list[str],
    **kwargs,
) -> pd.DataFrame:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            frame = getattr(pro, endpoint)(
                trade_date=trade_date,
                fields=",".join(fields),
                **kwargs,
            )
            if frame is None or frame.empty:
                return pd.DataFrame(columns=fields)
            missing_columns = set(fields) - set(frame.columns)
            if missing_columns:
                raise RuntimeError(
                    f"{endpoint} response for {trade_date} is missing columns: "
                    f"{sorted(missing_columns)}"
                )
            frame = frame[fields].copy()
            frame["trade_date"] = frame["trade_date"].astype(str)
            frame["ts_code"] = frame["ts_code"].astype(str)
            if set(frame["trade_date"]) != {trade_date}:
                raise RuntimeError(f"{endpoint} response date mismatch for {trade_date}")
            return frame
        except Exception as exc:
            if attempt == MAX_RETRIES:
                raise RuntimeError(
                    f"Failed to download {endpoint} for {trade_date} "
                    f"after {MAX_RETRIES} attempts"
                ) from exc
            time.sleep(attempt * 2)
    raise AssertionError("unreachable")


def main() -> None:
    if not TUSHARE_API_KEY.strip():
        raise SystemExit("Please set the TUSHARE_API_KEY environment variable")
    if not DATABASE_PATH.exists():
        raise SystemExit(f"Database does not exist: {DATABASE_PATH}")

    connection = duckdb.connect(str(DATABASE_PATH))
    try:
        create_anomaly_table(connection)
        universe, start_date, end_date = historical_universe(connection)
        trading_dates = [
            row[0]
            for row in connection.execute(
                """
                SELECT cal_date
                FROM calender
                WHERE cal_date BETWEEN ? AND ? AND is_open = 1
                ORDER BY cal_date
                """,
                [start_date, end_date],
            ).fetchall()
        ]
        if not trading_dates:
            raise RuntimeError("calender contains no open dates in the constituent range")

        pro = ts.pro_api(TUSHARE_API_KEY.strip())
        print(
            f"Universe: {len(universe):,} stocks; range: {start_date}-{end_date}; "
            f"open dates: {len(trading_dates):,}"
        )

        total_rows = 0
        empty_limit_dates = []
        for index, trade_date in enumerate(trading_dates, start=1):
            limits = fetch_daily_source(
                pro,
                "stk_limit",
                trade_date,
                ["trade_date", "ts_code", "up_limit", "down_limit"],
            )
            if limits.empty:
                empty_limit_dates.append(trade_date)
            time.sleep(REQUEST_INTERVAL_SECONDS)
            suspensions = fetch_daily_source(
                pro,
                "suspend_d",
                trade_date,
                ["trade_date", "ts_code", "suspend_type"],
                suspend_type="S",
            )
            time.sleep(REQUEST_INTERVAL_SECONDS)
            st_stocks = fetch_daily_source(
                pro,
                "stock_st",
                trade_date,
                ["trade_date", "ts_code", "type"],
            )

            statuses: dict[str, set[str]] = {}
            for ts_code in st_stocks["ts_code"]:
                if ts_code in universe:
                    statuses.setdefault(ts_code, set()).add("ST")
            for ts_code in suspensions["ts_code"]:
                if ts_code in universe:
                    statuses.setdefault(ts_code, set()).add("SUSPENDED")

            limits = limits[limits["ts_code"].isin(universe)].copy()
            limits["up_limit"] = pd.to_numeric(limits["up_limit"], errors="coerce")
            limits["down_limit"] = pd.to_numeric(limits["down_limit"], errors="coerce")
            closes = connection.execute(
                "SELECT ts_code, close FROM market WHERE trade_date = ?",
                [trade_date],
            ).df()
            price_limits = closes.merge(limits, on="ts_code", how="inner", validate="one_to_one")
            for row in price_limits.itertuples(index=False):
                if pd.notna(row.close) and pd.notna(row.up_limit):
                    if abs(row.close - row.up_limit) <= 1e-6:
                        statuses.setdefault(row.ts_code, set()).add("LIMIT_UP")
                if pd.notna(row.close) and pd.notna(row.down_limit):
                    if abs(row.close - row.down_limit) <= 1e-6:
                        statuses.setdefault(row.ts_code, set()).add("LIMIT_DOWN")

            rows = [
                (
                    trade_date,
                    ts_code,
                    "|".join(status for status in STATUS_ORDER if status in statuses[ts_code]),
                )
                for ts_code in sorted(statuses)
            ]
            batch = pd.DataFrame(rows, columns=["trade_date", "ts_code", "value"])
            connection.register("anomaly_batch", batch)
            try:
                connection.execute("BEGIN TRANSACTION")
                connection.execute("DELETE FROM anomaly WHERE trade_date = ?", [trade_date])
                connection.execute("INSERT INTO anomaly SELECT * FROM anomaly_batch")
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
            finally:
                connection.unregister("anomaly_batch")

            total_rows += len(batch)
            print(
                f"[{index:,}/{len(trading_dates):,}] {trade_date}: "
                f"saved {len(batch):,} anomalies"
            )
            if index % 100 == 0:
                connection.execute("CHECKPOINT")
            time.sleep(REQUEST_INTERVAL_SECONDS)

        connection.execute("CHECKPOINT")
        stored_rows, first_date, last_date = connection.execute(
            "SELECT COUNT(*), MIN(trade_date), MAX(trade_date) FROM anomaly"
        ).fetchone()
        print(
            f"Done. Processed {len(trading_dates):,} dates and saved {total_rows:,} rows; "
            f"anomaly now contains {stored_rows:,} rows from {first_date} to {last_date}."
        )
        if empty_limit_dates:
            print(
                f"Warning: stk_limit returned no data for {len(empty_limit_dates):,} open dates "
                f"from {empty_limit_dates[0]} to {empty_limit_dates[-1]}; "
                "no limit status was inferred for those dates."
            )
    finally:
        connection.close()


if __name__ == "__main__":
    main()
