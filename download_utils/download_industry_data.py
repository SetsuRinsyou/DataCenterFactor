#!/usr/bin/env python3
"""Download SW2021 level-1 industries for all A-shares in market."""

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


def latest_industries(members: pd.DataFrame, universe: set[str]) -> pd.DataFrame:
    members = members[members["ts_code"].isin(universe)].copy()
    members = members.drop_duplicates(MEMBER_COLUMNS)
    members["in_date"] = members["in_date"].fillna("").astype(str)
    members["out_date"] = members["out_date"].fillna("").astype(str)
    members["current_priority"] = (members["is_new"] == "Y").astype("int8")
    members["last_date"] = members["out_date"].where(
        members["out_date"] != "", members["in_date"]
    )
    members = members.sort_values(
        ["ts_code", "current_priority", "last_date", "in_date"],
        ascending=[True, False, False, False],
    )

    best = members.groupby("ts_code", sort=False).head(1)
    top_rank = members.merge(
        best[["ts_code", "current_priority", "last_date", "in_date"]],
        on=["ts_code", "current_priority", "last_date", "in_date"],
        how="inner",
    )
    ambiguous = top_rank.groupby("ts_code")["l1_name"].nunique()
    ambiguous = ambiguous[ambiguous > 1]
    if not ambiguous.empty:
        raise RuntimeError(
            f"Ambiguous latest L1 industry for {len(ambiguous):,} stocks: "
            f"{', '.join(ambiguous.index[:10])}"
        )

    result = pd.DataFrame({"ts_code": sorted(universe)})
    result = result.merge(
        best[["ts_code", "l1_name"]].rename(columns={"l1_name": "sw_l1_industry"}),
        on="ts_code",
        how="left",
        validate="one_to_one",
    )
    return result


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
        result = latest_industries(members, universe)
        connection.register("industry_batch", result)
        try:
            connection.execute("BEGIN TRANSACTION")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS industry (
                    ts_code VARCHAR PRIMARY KEY,
                    sw_l1_industry VARCHAR
                )
                """
            )
            connection.execute("DELETE FROM industry")
            connection.execute("INSERT INTO industry SELECT * FROM industry_batch")
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.unregister("industry_batch")

        connection.execute("CHECKPOINT")
        total_rows, classified_rows, industry_count = connection.execute(
            """
            SELECT COUNT(*), COUNT(sw_l1_industry), COUNT(DISTINCT sw_l1_industry)
            FROM industry
            """
        ).fetchone()
        print(
            f"Done. industry contains {total_rows:,} stocks; "
            f"{classified_rows:,} classified into {industry_count:,} SW2021 L1 industries; "
            f"{total_rows - classified_rows:,} unclassified."
        )
    finally:
        connection.close()


if __name__ == "__main__":
    main()
