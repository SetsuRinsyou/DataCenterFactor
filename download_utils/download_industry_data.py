#!/usr/bin/env python3
"""Expand SW2021 membership intervals onto market's stock/trading-day keys."""

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


MEMBER_COLUMNS = [
    "l1_code",
    "l1_name",
    "ts_code",
    "in_date",
    "out_date",
    "is_new",
]
MEMBER_ROW_LIMIT = 2000


def historical_universe(connection: duckdb.DuckDBPyConnection) -> set[str]:
    universe = {
        row[0]
        for row in connection.execute("SELECT DISTINCT ts_code FROM market").fetchall()
    }
    if not universe:
        raise RuntimeError("market is empty")
    return universe


def fetch_l1_industries(pro) -> pd.DataFrame:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            frame = pro.index_classify(level="L1", src="SW2021")
            if frame is None or frame.empty:
                raise RuntimeError("Tushare returned no SW2021 level-1 industries")
            required = {"index_code", "industry_name", "level"}
            missing = required - set(frame.columns)
            if missing:
                raise RuntimeError(f"Industry response is missing columns: {sorted(missing)}")
            frame = frame[list(required)].copy()
            if set(frame["level"]) != {"L1"}:
                raise RuntimeError("Industry response contains non-L1 classifications")
            if frame["index_code"].duplicated().any():
                raise RuntimeError("Industry response contains duplicate L1 codes")
            return frame.sort_values("index_code", ignore_index=True)
        except Exception as exc:
            if attempt == MAX_RETRIES:
                raise RuntimeError("Failed to download SW2021 level-1 industries") from exc
            time.sleep(attempt * 2)
    raise AssertionError("unreachable")


def fetch_members(pro, l1_code: str, is_new: str) -> pd.DataFrame:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            frame = pro.index_member_all(l1_code=l1_code, is_new=is_new)
            if frame is None or frame.empty:
                return pd.DataFrame(columns=MEMBER_COLUMNS)
            if len(frame) >= MEMBER_ROW_LIMIT:
                raise RuntimeError(
                    f"Tushare returned {len(frame):,} rows for {l1_code}/{is_new}; "
                    "the response may have reached the API limit"
                )
            missing = set(MEMBER_COLUMNS) - set(frame.columns)
            if missing:
                raise RuntimeError(
                    f"Member response for {l1_code}/{is_new} is missing columns: "
                    f"{sorted(missing)}"
                )
            frame = frame[MEMBER_COLUMNS].copy()
            if set(frame["l1_code"].dropna()) != {l1_code}:
                raise RuntimeError(f"Member response L1 mismatch for {l1_code}/{is_new}")
            if set(frame["is_new"].dropna()) != {is_new}:
                raise RuntimeError(f"Member response status mismatch for {l1_code}/{is_new}")
            return frame
        except Exception as exc:
            if attempt == MAX_RETRIES:
                raise RuntimeError(
                    f"Failed to download members for {l1_code}/{is_new}"
                ) from exc
            time.sleep(attempt * 2)
    raise AssertionError("unreachable")


def industry_intervals(members: pd.DataFrame, universe: set[str]) -> pd.DataFrame:
    """Use inclusive membership dates; unknown history is never backfilled.

    The API returns adjacent records ending 20260630 and starting 20260701,
    for example for 000876.SZ. Excluding out_date would invent a one-day gap.
    """
    members = members[members["ts_code"].isin(universe)].copy()
    members = members.drop_duplicates(MEMBER_COLUMNS)
    if members.empty:
        raise RuntimeError("No industry intervals match the market universe")
    for field in ["l1_code", "l1_name", "in_date"]:
        if members[field].isna().any() or members[field].astype(str).str.strip().eq("").any():
            raise RuntimeError(f"Industry intervals contain missing {field}")
    for field in ["in_date", "out_date"]:
        members[field] = members[field].astype("string").str.strip().replace("", pd.NA)
        present = members[field].dropna()
        if not present.str.fullmatch(r"\d{8}").all():
            raise RuntimeError(f"Invalid {field} format")
        pd.to_datetime(present, format="%Y%m%d", errors="raise")
    if members.loc[members["is_new"] == "N", "out_date"].isna().any():
        raise RuntimeError("Historical industry intervals must have out_date")
    if (members["out_date"].notna() & (members["in_date"] > members["out_date"])).any():
        raise RuntimeError("Industry interval ends before it starts")
    if members.groupby("l1_code")["l1_name"].nunique().gt(1).any():
        raise RuntimeError("An L1 code maps to multiple names")
    return members[["ts_code", "l1_code", "l1_name", "in_date", "out_date"]].drop_duplicates()


def rebuild_industry(connection: duckdb.DuckDBPyConnection, intervals: pd.DataFrame) -> None:
    """Validate the daily expansion before atomically replacing industry."""
    connection.register("industry_intervals_batch", intervals)
    try:
        connection.execute("BEGIN TRANSACTION")
        connection.execute(
            """
            CREATE TEMP TABLE industry_daily_stage AS
            SELECT m.trade_date, m.ts_code, MIN(i.l1_name) AS sw_l1_industry,
                   COUNT(DISTINCT i.l1_code) AS industry_count
            FROM market m
            LEFT JOIN industry_intervals_batch i
              ON m.ts_code = i.ts_code
             AND m.trade_date >= i.in_date
             AND (i.out_date IS NULL OR m.trade_date <= i.out_date)
            GROUP BY m.trade_date, m.ts_code
            """
        )
        conflicts = connection.execute(
            "SELECT trade_date, ts_code FROM industry_daily_stage "
            "WHERE industry_count > 1 LIMIT 10"
        ).fetchall()
        if conflicts:
            raise RuntimeError(f"Overlapping L1 memberships on market dates: {conflicts}")
        total_rows, classified_rows = connection.execute(
            "SELECT COUNT(*), COUNT(sw_l1_industry) FROM industry_daily_stage"
        ).fetchone()
        if total_rows != connection.execute("SELECT COUNT(*) FROM market").fetchone()[0]:
            raise RuntimeError("Daily industry keys do not match market")
        if not classified_rows:
            raise RuntimeError("No market rows could be classified")
        connection.execute("DROP TABLE IF EXISTS industry")
        connection.execute(
            """
            CREATE TABLE industry (
                trade_date VARCHAR NOT NULL,
                ts_code VARCHAR NOT NULL,
                sw_l1_industry VARCHAR,
                PRIMARY KEY (trade_date, ts_code)
            )
            """
        )
        connection.execute(
            "INSERT INTO industry SELECT trade_date, ts_code, sw_l1_industry "
            "FROM industry_daily_stage ORDER BY trade_date, ts_code"
        )
        connection.execute("DROP TABLE industry_daily_stage")
        connection.execute("COMMIT")
    except Exception:
        connection.execute("ROLLBACK")
        raise
    finally:
        connection.unregister("industry_intervals_batch")


def main() -> None:
    if not TUSHARE_API_KEY.strip():
        raise SystemExit("Please set the TUSHARE_API_KEY environment variable")
    if not DATABASE_PATH.exists():
        raise SystemExit(f"Database does not exist: {DATABASE_PATH}")

    connection = duckdb.connect(str(DATABASE_PATH))
    try:
        universe = historical_universe(connection)
        pro = ts.pro_api(TUSHARE_API_KEY.strip())
        industries = fetch_l1_industries(pro)
        print(f"SW2021 level-1 industries: {len(industries):,}; universe: {len(universe):,}")

        frames = []
        for index, row in enumerate(industries.itertuples(index=False), start=1):
            current = fetch_members(pro, row.index_code, "Y")
            time.sleep(REQUEST_INTERVAL_SECONDS)
            historical = fetch_members(pro, row.index_code, "N")
            frames.extend([current, historical])
            print(
                f"[{index:,}/{len(industries):,}] {row.index_code} {row.industry_name}: "
                f"{len(current):,} current, {len(historical):,} historical"
            )
            time.sleep(REQUEST_INTERVAL_SECONDS)

        members = pd.concat(frames, ignore_index=True)
        intervals = industry_intervals(members, universe)
        print(f"Expanding {len(intervals):,} membership intervals onto market dates", flush=True)
        rebuild_industry(connection, intervals)

        connection.execute("CHECKPOINT")
        total_rows, classified_rows, industry_count = connection.execute(
            """
            SELECT COUNT(*), COUNT(sw_l1_industry), COUNT(DISTINCT sw_l1_industry)
            FROM industry
            """
        ).fetchone()
        print(
            f"Done. industry contains {total_rows:,} stock/trading-day rows; "
            f"{classified_rows:,} classified into {industry_count:,} SW2021 L1 industries; "
            f"{total_rows - classified_rows:,} unclassified."
        )
    finally:
        connection.close()


if __name__ == "__main__":
    main()
