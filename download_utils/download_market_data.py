#!/usr/bin/env python3
"""Download daily market data for all historical mainland A-share stocks."""

import os
import time
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd
import tushare as ts


TUSHARE_API_KEY = os.environ.get("TUSHARE_API_KEY", "")
DATABASE_PATH = Path(__file__).resolve().parents[1] / "data" / "src_data.duckdb"
REQUEST_INTERVAL_SECONDS = 0.13
MAX_RETRIES = 5
DAILY_ROW_LIMIT = 6000

DAILY_COLUMNS = [
    "trade_date",
    "ts_code",
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "change",
    "pct_chg",
    "vol",
    "amount",
    "ah_vol",
    "ah_amount",
]
DAILY_BASIC_COLUMNS = [
    "turnover_rate",
    "turnover_rate_f",
    "volume_ratio",
    "pe",
    "pe_ttm",
    "pb",
    "ps",
    "ps_ttm",
    "dv_ratio",
    "dv_ttm",
    "total_share",
    "float_share",
    "free_share",
    "total_mv",
    "circ_mv",
    "limit_status",
]
MARKET_COLUMNS = [*DAILY_COLUMNS, "adj_factor", *DAILY_BASIC_COLUMNS]
API_FIELDS = ",".join(["ts_code", "trade_date", *DAILY_COLUMNS[2:]])
DAILY_BASIC_FIELDS = ",".join(["ts_code", "trade_date", *DAILY_BASIC_COLUMNS])
CALENDAR_API_COLUMNS = ["exchange", "cal_date", "is_open", "pretrade_date"]
CALENDAR_COLUMNS = ["cal_date", "is_open", "pretrade_date"]
STOCK_BASIC_FIELDS = [
    "ts_code",
    "exchange",
    "curr_type",
    "list_status",
    "list_date",
    "delist_date",
]


def create_market_table(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS market (
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
            ah_vol DOUBLE,
            ah_amount DOUBLE,
            adj_factor DOUBLE,
            turnover_rate DOUBLE,
            turnover_rate_f DOUBLE,
            volume_ratio DOUBLE,
            pe DOUBLE,
            pe_ttm DOUBLE,
            pb DOUBLE,
            ps DOUBLE,
            ps_ttm DOUBLE,
            dv_ratio DOUBLE,
            dv_ttm DOUBLE,
            total_share DOUBLE,
            float_share DOUBLE,
            free_share DOUBLE,
            total_mv DOUBLE,
            circ_mv DOUBLE,
            limit_status TINYINT,
            PRIMARY KEY (trade_date, ts_code)
        )
        """
    )
    columns = {row[1] for row in connection.execute("PRAGMA table_info('market')").fetchall()}
    if "adj_factor" not in columns:
        connection.execute("ALTER TABLE market ADD COLUMN adj_factor DOUBLE")
    for column in DAILY_BASIC_COLUMNS:
        if column not in columns:
            column_type = "TINYINT" if column == "limit_status" else "DOUBLE"
            connection.execute(f"ALTER TABLE market ADD COLUMN {column} {column_type}")


def create_calendar_table(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS calender (
            cal_date VARCHAR PRIMARY KEY,
            is_open TINYINT NOT NULL CHECK (is_open IN (0, 1)),
            pretrade_date VARCHAR
        )
        """
    )


def market_date_range(connection: duckdb.DuckDBPyConnection) -> tuple[str, str]:
    start_date, end_date = connection.execute(
        "SELECT MIN(trade_date), MAX(trade_date) FROM market"
    ).fetchone()
    if not start_date or not end_date:
        raise RuntimeError("market is empty; establish its target date range first")
    return start_date, end_date


def zz500_universe(connection: duckdb.DuckDBPyConnection) -> set[str]:
    rows = connection.execute("SELECT * FROM zz500_constituents").fetchall()
    return {code for row in rows for code in row[1:] if code}


def fetch_all_a_universe(pro, start_date: str, end_date: str) -> pd.DataFrame:
    frames = []
    for list_status in ("L", "D", "P"):
        for exchange in ("SSE", "SZSE", "BSE"):
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    frame = pro.stock_basic(
                        exchange=exchange,
                        list_status=list_status,
                        fields=",".join(STOCK_BASIC_FIELDS),
                    )
                    if frame is not None and not frame.empty:
                        missing = set(STOCK_BASIC_FIELDS) - set(frame.columns)
                        if missing:
                            raise RuntimeError(
                                f"stock_basic response is missing columns: {sorted(missing)}"
                            )
                        frames.append(frame[STOCK_BASIC_FIELDS])
                    break
                except Exception as exc:
                    if attempt == MAX_RETRIES:
                        raise RuntimeError(
                            f"Failed to download stock_basic for {exchange}/{list_status}"
                        ) from exc
                    time.sleep(attempt * 2)
            time.sleep(REQUEST_INTERVAL_SECONDS)

    if not frames:
        raise RuntimeError("Tushare returned no mainland A-share stocks")
    stocks = pd.concat(frames, ignore_index=True)
    stocks = stocks[
        stocks["exchange"].isin({"SSE", "SZSE", "BSE"})
        & stocks["curr_type"].eq("CNY")
    ].copy()
    stocks["list_date"] = stocks["list_date"].fillna("").astype(str)
    stocks["delist_date"] = stocks["delist_date"].fillna("").astype(str)
    stocks = stocks[
        stocks["list_date"].str.fullmatch(r"\d{8}", na=False)
        & stocks["list_date"].le(end_date)
        & (stocks["delist_date"].eq("") | stocks["delist_date"].ge(start_date))
    ].copy()
    if stocks["ts_code"].duplicated().any():
        raise RuntimeError("stock_basic returned duplicate A-share codes")
    return stocks.sort_values("ts_code", ignore_index=True)


def fetch_calendar(pro, start_date: str, end_date: str) -> pd.DataFrame:
    frames = []
    for year in range(int(start_date[:4]), int(end_date[:4]) + 1):
        period_start = max(start_date, f"{year}0101")
        period_end = min(end_date, f"{year}1231")
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                frame = pro.trade_cal(
                    exchange="SSE",
                    start_date=period_start,
                    end_date=period_end,
                    fields=",".join(CALENDAR_API_COLUMNS),
                )
                if frame is None or frame.empty:
                    raise RuntimeError(
                        f"Tushare returned no SSE calendar for {period_start}-{period_end}"
                    )
                frames.append(frame)
                break
            except Exception as exc:
                if attempt == MAX_RETRIES:
                    raise RuntimeError(
                        f"Failed to download SSE calendar for {period_start}-{period_end}"
                    ) from exc
                time.sleep(attempt * 2)
        time.sleep(REQUEST_INTERVAL_SECONDS)

    calendar = pd.concat(frames, ignore_index=True)[CALENDAR_API_COLUMNS].copy()
    calendar["exchange"] = calendar["exchange"].astype(str)
    calendar["cal_date"] = calendar["cal_date"].astype(str)
    calendar["is_open"] = pd.to_numeric(calendar["is_open"], errors="raise").astype("int8")
    calendar["pretrade_date"] = calendar["pretrade_date"].where(
        calendar["pretrade_date"].notna(), None
    )

    expected_rows = (
        datetime.strptime(end_date, "%Y%m%d") - datetime.strptime(start_date, "%Y%m%d")
    ).days + 1
    if len(calendar) != expected_rows or calendar["cal_date"].nunique() != expected_rows:
        raise RuntimeError(
            f"Incomplete SSE calendar: received {len(calendar)} rows, expected {expected_rows}"
        )
    if set(calendar["exchange"]) != {"SSE"}:
        raise RuntimeError(f"Unexpected calendar exchanges: {sorted(set(calendar['exchange']))}")
    if calendar["cal_date"].min() != start_date or calendar["cal_date"].max() != end_date:
        raise RuntimeError("SSE calendar date range does not match the constituent range")
    return calendar[CALENDAR_COLUMNS].sort_values("cal_date", ignore_index=True)


def write_calendar(
    connection: duckdb.DuckDBPyConnection, calendar: pd.DataFrame, start_date: str, end_date: str
) -> None:
    connection.register("calendar_batch", calendar)
    try:
        connection.execute("BEGIN TRANSACTION")
        connection.execute(
            "DELETE FROM calender WHERE cal_date BETWEEN ? AND ?", [start_date, end_date]
        )
        connection.execute("INSERT INTO calender SELECT * FROM calendar_batch")
        connection.execute("COMMIT")
    except Exception:
        connection.execute("ROLLBACK")
        raise
    finally:
        connection.unregister("calendar_batch")


def fetch_daily(pro, trade_date: str) -> pd.DataFrame:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            frame = pro.daily(trade_date=trade_date, fields=API_FIELDS)
            if frame is None or frame.empty:
                raise RuntimeError(f"Tushare returned no daily data for open date {trade_date}")
            if len(frame) >= DAILY_ROW_LIMIT:
                raise RuntimeError(
                    f"Tushare returned {len(frame)} rows for {trade_date}; "
                    "the response may have reached the API limit"
                )
            return frame
        except Exception as exc:
            if attempt == MAX_RETRIES:
                raise RuntimeError(
                    f"Failed to download daily data for {trade_date} after {MAX_RETRIES} attempts"
                ) from exc
            time.sleep(attempt * 2)
    raise AssertionError("unreachable")


def fetch_adj_factor(pro, trade_date: str) -> pd.DataFrame:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            frame = pro.adj_factor(
                trade_date=trade_date,
                fields="ts_code,trade_date,adj_factor",
            )
            if frame is None or frame.empty:
                raise RuntimeError(f"Tushare returned no adj factors for open date {trade_date}")
            missing_columns = {"ts_code", "trade_date", "adj_factor"} - set(frame.columns)
            if missing_columns:
                raise RuntimeError(
                    f"Adj-factor response for {trade_date} is missing columns: "
                    f"{sorted(missing_columns)}"
                )
            frame = frame[["ts_code", "trade_date", "adj_factor"]].copy()
            frame["trade_date"] = frame["trade_date"].astype(str)
            if set(frame["trade_date"]) != {trade_date}:
                raise RuntimeError(f"Adj-factor response date mismatch for {trade_date}")
            if frame.duplicated(["trade_date", "ts_code"]).any():
                raise RuntimeError(f"Duplicate adj factors returned for {trade_date}")
            frame["adj_factor"] = pd.to_numeric(frame["adj_factor"], errors="coerce")
            if frame["adj_factor"].isna().any():
                raise RuntimeError(f"Invalid adj factors returned for {trade_date}")
            return frame
        except Exception as exc:
            if attempt == MAX_RETRIES:
                raise RuntimeError(
                    f"Failed to download adj factors for {trade_date} "
                    f"after {MAX_RETRIES} attempts"
                ) from exc
            time.sleep(attempt * 2)
    raise AssertionError("unreachable")


def fetch_daily_basic(pro, trade_date: str) -> pd.DataFrame:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            frame = pro.daily_basic(trade_date=trade_date, fields=DAILY_BASIC_FIELDS)
            if frame is None or frame.empty:
                raise RuntimeError(f"Tushare returned no daily indicators for {trade_date}")
            if len(frame) >= DAILY_ROW_LIMIT:
                raise RuntimeError(
                    f"Tushare returned {len(frame)} daily indicators for {trade_date}; "
                    "the response may have reached the API limit"
                )
            missing_columns = {"ts_code", "trade_date", *DAILY_BASIC_COLUMNS} - set(
                frame.columns
            )
            if missing_columns:
                raise RuntimeError(
                    f"Daily-indicator response for {trade_date} is missing columns: "
                    f"{sorted(missing_columns)}"
                )
            frame = frame[["ts_code", "trade_date", *DAILY_BASIC_COLUMNS]].copy()
            frame["trade_date"] = frame["trade_date"].astype(str)
            if set(frame["trade_date"]) != {trade_date}:
                raise RuntimeError(f"Daily-indicator response date mismatch for {trade_date}")
            if frame.duplicated(["trade_date", "ts_code"]).any():
                raise RuntimeError(f"Duplicate daily indicators returned for {trade_date}")
            for column in DAILY_BASIC_COLUMNS:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
            return frame
        except Exception as exc:
            if attempt == MAX_RETRIES:
                raise RuntimeError(
                    f"Failed to download daily indicators for {trade_date} "
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
        create_market_table(connection)
        create_calendar_table(connection)
        start_date, end_date = market_date_range(connection)
        pro = ts.pro_api(TUSHARE_API_KEY.strip())
        stocks = fetch_all_a_universe(pro, start_date, end_date)
        universe = set(stocks["ts_code"])
        existing_dates = {
            row[0] for row in connection.execute("SELECT DISTINCT trade_date FROM market").fetchall()
        }
        existing_codes = {
            row[0] for row in connection.execute("SELECT DISTINCT ts_code FROM market").fetchall()
        }
        missing_codes = universe - existing_codes
        expanded_codes = sorted(existing_codes - zz500_universe(connection))
        if expanded_codes:
            placeholders = ", ".join("?" for _ in expanded_codes)
            expansion_frontier = connection.execute(
                f"SELECT MAX(trade_date) FROM market WHERE ts_code IN ({placeholders})",
                expanded_codes,
            ).fetchone()[0]
        else:
            expansion_frontier = None

        calendar = fetch_calendar(pro, start_date, end_date)
        write_calendar(connection, calendar, start_date, end_date)
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
        pending_dates = [trade_date for trade_date in trading_dates if trade_date not in existing_dates]
        if missing_codes and expansion_frontier != end_date:
            remaining_dates = [
                trade_date
                for trade_date in trading_dates
                if expansion_frontier is None or trade_date > expansion_frontier
            ]
            backfill_start = remaining_dates[0] if remaining_dates else None
            download_dates = sorted(
                set(pending_dates)
                | set(remaining_dates)
            )
        else:
            backfill_start = None
            download_dates = pending_dates
        factor_dates = [
            row[0]
            for row in connection.execute(
                """
                SELECT trade_date
                FROM market
                GROUP BY trade_date
                HAVING COUNT(*) FILTER (WHERE adj_factor IS NULL) = COUNT(*)
                ORDER BY trade_date
                """
            ).fetchall()
        ]
        indicator_missing_condition = " AND ".join(
            f"{column} IS NULL" for column in DAILY_BASIC_COLUMNS
        )
        indicator_dates = [
            row[0]
            for row in connection.execute(
                f"""
                SELECT trade_date
                FROM market
                GROUP BY trade_date
                HAVING COUNT(*) FILTER (WHERE {indicator_missing_condition}) = COUNT(*)
                ORDER BY trade_date
                """
            ).fetchall()
        ]
        print(
            f"Saved {len(calendar):,} SSE calendar rows. "
            f"Universe: {len(universe):,} stocks; range: {start_date}-{end_date}; "
            f"existing stocks: {len(existing_codes & universe):,}; "
            f"missing stocks: {len(missing_codes):,}; "
            f"expansion frontier: {expansion_frontier or 'none'}; "
            f"backfill start: {backfill_start or 'none'}; "
            f"open dates: {len(trading_dates):,}; download dates: {len(download_dates):,}; "
            f"pending factor dates: {len(factor_dates):,}; "
            f"pending indicator dates: {len(indicator_dates):,}"
        )

        updated_factor_rows = 0
        for index, trade_date in enumerate(factor_dates, start=1):
            factors = fetch_adj_factor(pro, trade_date)
            factors = factors[factors["ts_code"].isin(universe)].copy()
            connection.register("adj_factor_batch", factors)
            try:
                connection.execute("BEGIN TRANSACTION")
                missing_rows = connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM market AS m
                    LEFT JOIN adj_factor_batch AS a
                        ON m.trade_date = a.trade_date AND m.ts_code = a.ts_code
                    WHERE m.trade_date = ? AND m.adj_factor IS NULL AND a.adj_factor IS NULL
                    """,
                    [trade_date],
                ).fetchone()[0]
                if missing_rows:
                    print(
                        f"Warning: adj-factor response for {trade_date} is missing "
                        f"{missing_rows:,} market rows; leaving them NULL"
                    )
                rows_to_update = connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM market AS m
                    JOIN adj_factor_batch AS a USING (trade_date, ts_code)
                    WHERE m.trade_date = ? AND m.adj_factor IS NULL
                    """,
                    [trade_date],
                ).fetchone()[0]
                connection.execute(
                    """
                    UPDATE market AS m
                    SET adj_factor = a.adj_factor
                    FROM adj_factor_batch AS a
                    WHERE m.trade_date = a.trade_date
                      AND m.ts_code = a.ts_code
                      AND m.trade_date = ?
                      AND m.adj_factor IS NULL
                    """,
                    [trade_date],
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
            finally:
                connection.unregister("adj_factor_batch")

            updated_factor_rows += rows_to_update
            print(
                f"[factor {index:,}/{len(factor_dates):,}] {trade_date}: "
                f"updated {rows_to_update:,} rows"
            )
            if index % 100 == 0:
                connection.execute("CHECKPOINT")
            time.sleep(REQUEST_INTERVAL_SECONDS)

        updated_indicator_rows = 0
        indicator_assignments = ", ".join(
            f"{column} = d.{column}" for column in DAILY_BASIC_COLUMNS
        )
        aliased_missing_condition = " AND ".join(
            f"m.{column} IS NULL" for column in DAILY_BASIC_COLUMNS
        )
        for index, trade_date in enumerate(indicator_dates, start=1):
            indicators = fetch_daily_basic(pro, trade_date)
            indicators = indicators[indicators["ts_code"].isin(universe)].copy()
            connection.register("daily_basic_batch", indicators)
            try:
                connection.execute("BEGIN TRANSACTION")
                missing_rows = connection.execute(
                    f"""
                    SELECT COUNT(*)
                    FROM market AS m
                    LEFT JOIN daily_basic_batch AS d
                        ON m.trade_date = d.trade_date AND m.ts_code = d.ts_code
                    WHERE m.trade_date = ?
                      AND {aliased_missing_condition}
                      AND d.ts_code IS NULL
                    """,
                    [trade_date],
                ).fetchone()[0]
                if missing_rows:
                    print(
                        f"Warning: daily indicators for {trade_date} are missing "
                        f"{missing_rows:,} market rows; leaving them NULL"
                    )
                rows_to_update = connection.execute(
                    f"""
                    SELECT COUNT(*)
                    FROM market AS m
                    JOIN daily_basic_batch AS d
                        ON m.trade_date = d.trade_date AND m.ts_code = d.ts_code
                    WHERE m.trade_date = ? AND {aliased_missing_condition}
                    """,
                    [trade_date],
                ).fetchone()[0]
                connection.execute(
                    f"""
                    UPDATE market AS m
                    SET {indicator_assignments}
                    FROM daily_basic_batch AS d
                    WHERE m.trade_date = d.trade_date
                      AND m.ts_code = d.ts_code
                      AND m.trade_date = ?
                      AND {aliased_missing_condition}
                    """,
                    [trade_date],
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
            finally:
                connection.unregister("daily_basic_batch")

            updated_indicator_rows += rows_to_update
            print(
                f"[indicator {index:,}/{len(indicator_dates):,}] {trade_date}: "
                f"updated {rows_to_update:,} rows"
            )
            if index % 100 == 0:
                connection.execute("CHECKPOINT")
            time.sleep(REQUEST_INTERVAL_SECONDS)

        initial_total_rows = connection.execute("SELECT COUNT(*) FROM market").fetchone()[0]
        for index, trade_date in enumerate(download_dates, start=1):
            frame = fetch_daily(pro, trade_date)
            time.sleep(REQUEST_INTERVAL_SECONDS)
            factors = fetch_adj_factor(pro, trade_date)
            time.sleep(REQUEST_INTERVAL_SECONDS)
            indicators = fetch_daily_basic(pro, trade_date)

            missing_columns = set(DAILY_COLUMNS[:-2]) - set(frame.columns)
            if missing_columns:
                raise RuntimeError(
                    f"Daily response for {trade_date} is missing columns: {sorted(missing_columns)}"
                )
            for column in ("ah_vol", "ah_amount"):
                if column not in frame.columns:
                    frame[column] = pd.NA

            frame["trade_date"] = frame["trade_date"].astype(str)
            response_dates = set(frame["trade_date"])
            if response_dates != {trade_date}:
                raise RuntimeError(
                    f"Daily response date mismatch for {trade_date}: {sorted(response_dates)}"
                )

            frame = frame[frame["ts_code"].isin(universe)][DAILY_COLUMNS].copy()
            if frame.duplicated(["trade_date", "ts_code"]).any():
                raise RuntimeError(f"Duplicate daily records returned for {trade_date}")
            for column in DAILY_COLUMNS[2:]:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
            factors = factors[factors["ts_code"].isin(universe)].copy()
            frame = frame.merge(
                factors,
                on=["trade_date", "ts_code"],
                how="left",
                validate="one_to_one",
            )[[*DAILY_COLUMNS, "adj_factor"]]
            if frame["adj_factor"].isna().any():
                missing_rows = int(frame["adj_factor"].isna().sum())
                print(
                    f"Warning: adj-factor response for {trade_date} is missing "
                    f"{missing_rows:,} market rows; leaving them NULL"
                )
            indicators = indicators[indicators["ts_code"].isin(universe)].copy()
            indicators["_daily_basic_present"] = True
            frame = frame.merge(
                indicators,
                on=["trade_date", "ts_code"],
                how="left",
                validate="one_to_one",
            )
            missing_rows = int(frame["_daily_basic_present"].isna().sum())
            if missing_rows:
                print(
                    f"Warning: daily indicators for {trade_date} are missing "
                    f"{missing_rows:,} market rows; leaving them NULL"
                )
            frame = frame[MARKET_COLUMNS]

            connection.register("daily_batch", frame)
            try:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO market
                    SELECT * FROM daily_batch
                    """
                )
            finally:
                connection.unregister("daily_batch")

            print(
                f"[{index:,}/{len(download_dates):,}] {trade_date}: "
                f"downloaded {len(frame):,} rows"
            )
            if index % 100 == 0:
                connection.execute("CHECKPOINT")
            time.sleep(REQUEST_INTERVAL_SECONDS)

        connection.execute("CHECKPOINT")
        total_rows, first_date, last_date = connection.execute(
            "SELECT COUNT(*), MIN(trade_date), MAX(trade_date) FROM market"
        ).fetchone()
        inserted_rows = total_rows - initial_total_rows
        print(
            f"Done. Updated {updated_factor_rows:,} adj factors and "
            f"{updated_indicator_rows:,} daily indicators; inserted {inserted_rows:,} rows; "
            f"market now contains {total_rows:,} rows "
            f"from {first_date} to {last_date}."
        )
    finally:
        connection.close()


if __name__ == "__main__":
    main()
