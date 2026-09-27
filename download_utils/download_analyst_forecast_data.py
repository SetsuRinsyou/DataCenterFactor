#!/usr/bin/env python3
"""Download raw Tushare sell-side earnings forecasts into DuckDB."""

import argparse
import hashlib
import time
from calendar import monthrange
from datetime import date, datetime, timedelta
from pathlib import Path

import duckdb
import pandas as pd
import tushare as ts

from download_utils.download_market_data import (
    DATABASE_PATH,
    MAX_RETRIES,
    REQUEST_INTERVAL_SECONDS,
    TUSHARE_API_KEY,
)


TABLE_NAME = "analyst_forecast"
FIRST_DATE = "20100101"
PAGE_LIMIT = 3000
IDENTITY_FIELDS = [
    "ts_code",
    "report_date",
    "report_title",
    "report_type",
    "classify",
    "org_name",
    "author_name",
    "quarter",
    "create_time",
]
TEXT_FIELDS = [
    "ts_code",
    "name",
    "report_date",
    "report_title",
    "report_type",
    "classify",
    "org_name",
    "author_name",
    "quarter",
    "rating",
    "imp_dg",
]
NUMERIC_FIELDS = [
    "op_rt",
    "op_pr",
    "tp",
    "np",
    "eps",
    "pe",
    "rd",
    "roe",
    "ev_ebitda",
    "max_price",
    "min_price",
]
PERIOD_FORECAST_FIELDS = NUMERIC_FIELDS[:9]
SOURCE_FIELDS = [
    "ts_code",
    "name",
    "report_date",
    "report_title",
    "report_type",
    "classify",
    "org_name",
    "author_name",
    "quarter",
    *NUMERIC_FIELDS[:9],
    "rating",
    *NUMERIC_FIELDS[9:],
    "imp_dg",
    "create_time",
]
TABLE_COLUMNS = ["source_row_hash", *SOURCE_FIELDS]


def parse_yyyymmdd(value: str) -> date:
    return datetime.strptime(value, "%Y%m%d").date()


def month_ranges(start_date: str, end_date: str) -> list[tuple[str, str]]:
    start = parse_yyyymmdd(start_date)
    end = parse_yyyymmdd(end_date)
    if start > end:
        raise ValueError(f"start_date {start_date} is after end_date {end_date}")

    ranges = []
    cursor = start
    while cursor <= end:
        last_day = monthrange(cursor.year, cursor.month)[1]
        range_end = min(end, date(cursor.year, cursor.month, last_day))
        ranges.append((cursor.strftime("%Y%m%d"), range_end.strftime("%Y%m%d")))
        cursor = range_end + timedelta(days=1)
    return ranges


def create_table(connection: duckdb.DuckDBPyConnection, table_name: str) -> None:
    connection.execute(f'DROP TABLE IF EXISTS "{table_name}"')
    connection.execute(
        f"""
        CREATE TABLE "{table_name}" (
            source_row_hash VARCHAR PRIMARY KEY,
            ts_code VARCHAR NOT NULL,
            name VARCHAR,
            report_date VARCHAR NOT NULL,
            report_title VARCHAR,
            report_type VARCHAR,
            classify VARCHAR,
            org_name VARCHAR,
            author_name VARCHAR,
            quarter VARCHAR,
            op_rt DOUBLE,
            op_pr DOUBLE,
            tp DOUBLE,
            np DOUBLE,
            eps DOUBLE,
            pe DOUBLE,
            rd DOUBLE,
            roe DOUBLE,
            ev_ebitda DOUBLE,
            rating VARCHAR,
            max_price DOUBLE,
            min_price DOUBLE,
            imp_dg VARCHAR,
            create_time TIMESTAMP
        )
        """
    )


def table_exists(connection: duckdb.DuckDBPyConnection, table_name: str) -> bool:
    return table_name in {
        row[0] for row in connection.execute("SHOW TABLES").fetchall()
    }


def validate_schema(connection: duckdb.DuckDBPyConnection, table_name: str) -> None:
    info = connection.execute(f"PRAGMA table_info('{table_name}')").fetchall()
    columns = [row[1] for row in info]
    primary_key = [row[1] for row in info if row[5]]
    if columns != TABLE_COLUMNS or primary_key != ["source_row_hash"]:
        raise RuntimeError(
            f"{table_name} does not have the expected raw forecast schema"
        )


def canonical_identity_value(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


def row_hash(row: pd.Series) -> str:
    payload = "\x1f".join(
        canonical_identity_value(row[field]) for field in IDENTITY_FIELDS
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalize_forecasts(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=TABLE_COLUMNS)
    missing = set(SOURCE_FIELDS) - set(frame.columns)
    if missing:
        raise RuntimeError(
            f"report_rc response is missing fields: {sorted(missing)}"
        )

    data = frame[SOURCE_FIELDS].copy()
    for field in TEXT_FIELDS:
        data[field] = data[field].astype("string")
        data[field] = data[field].replace(r"^\s*$", pd.NA, regex=True)

    required_missing = data["ts_code"].isna() | data["report_date"].isna()
    if required_missing.any():
        raise RuntimeError("report_rc returned rows without ts_code or report_date")
    if not data["report_date"].str.fullmatch(r"\d{8}", na=False).all():
        raise RuntimeError("report_rc returned an invalid report_date")
    pd.to_datetime(data["report_date"], format="%Y%m%d", errors="raise")

    for field in NUMERIC_FIELDS:
        data[field] = pd.to_numeric(data[field], errors="coerce")

    source_create_time = data["create_time"].copy()
    data["create_time"] = pd.to_datetime(data["create_time"], errors="coerce")
    if source_create_time.notna().sum() != data["create_time"].notna().sum():
        raise RuntimeError("report_rc returned an invalid create_time")

    data.insert(0, "source_row_hash", data.apply(row_hash, axis=1))
    exact_columns = [column for column in TABLE_COLUMNS if column != "source_row_hash"]
    data = data.drop_duplicates(exact_columns)
    duplicate_hashes = data["source_row_hash"].duplicated(keep=False)
    if duplicate_hashes.any():
        examples = data.loc[duplicate_hashes, "source_row_hash"].unique()[:5]
        raise RuntimeError(
            "report_rc returned conflicting rows with the same source identity: "
            f"{examples.tolist()}"
        )
    return data[TABLE_COLUMNS].reset_index(drop=True)


def call_report_rc(pro, start_date: str, end_date: str, offset: int) -> pd.DataFrame:
    for retry in range(MAX_RETRIES + 1):
        try:
            frame = pro.report_rc(
                start_date=start_date,
                end_date=end_date,
                fields=",".join(SOURCE_FIELDS),
                limit=PAGE_LIMIT,
                offset=offset,
            )
            time.sleep(REQUEST_INTERVAL_SECONDS)
            return frame if frame is not None else pd.DataFrame(columns=SOURCE_FIELDS)
        except Exception as exc:
            if retry == MAX_RETRIES:
                raise RuntimeError(
                    "report_rc failed after retries for "
                    f"{start_date}-{end_date}, offset={offset}"
                ) from exc
            time.sleep(2 ** (retry + 1))
    raise AssertionError("unreachable")


def insert_batch(
    connection: duckdb.DuckDBPyConnection,
    table_name: str,
    frame: pd.DataFrame,
) -> None:
    if frame.empty:
        return
    connection.register("analyst_forecast_batch", frame)
    try:
        assignments = ", ".join(
            f'"{column}" = excluded."{column}"'
            for column in TABLE_COLUMNS
            if column != "source_row_hash"
        )
        connection.execute(
            f"""
            INSERT INTO "{table_name}"
            SELECT {', '.join(f'"{column}"' for column in TABLE_COLUMNS)}
            FROM analyst_forecast_batch
            ON CONFLICT (source_row_hash) DO UPDATE SET {assignments}
            """
        )
    finally:
        connection.unregister("analyst_forecast_batch")


def download_range(
    connection: duckdb.DuckDBPyConnection,
    pro,
    table_name: str,
    start_date: str,
    end_date: str,
) -> tuple[int, int]:
    api_rows = 0
    calls = 0
    for index, (range_start, range_end) in enumerate(
        month_ranges(start_date, end_date), start=1
    ):
        offset = 0
        range_hashes: set[str] = set()
        while True:
            raw = call_report_rc(pro, range_start, range_end, offset)
            calls += 1
            api_rows += len(raw)
            normalized = normalize_forecasts(raw)
            page_hashes = set(normalized["source_row_hash"])
            if len(raw) == PAGE_LIMIT and page_hashes and page_hashes <= range_hashes:
                raise RuntimeError(
                    f"report_rc pagination repeated a full page for "
                    f"{range_start}-{range_end}, offset={offset}"
                )
            range_hashes.update(page_hashes)
            insert_batch(connection, table_name, normalized)
            if len(raw) < PAGE_LIMIT:
                break
            offset += PAGE_LIMIT
        print(
            f"[{index}/{len(month_ranges(start_date, end_date))}] "
            f"{range_start}-{range_end}: {len(range_hashes):,} unique rows",
            flush=True,
        )
    return api_rows, calls


def validate_table(
    connection: duckdb.DuckDBPyConnection,
    table_name: str,
    expected_start: str | None = None,
    expected_end: str | None = None,
) -> tuple[int, str, str]:
    validate_schema(connection, table_name)
    total, first_date, last_date, distinct_hashes = connection.execute(
        f"""
        SELECT COUNT(*), MIN(report_date), MAX(report_date),
               COUNT(DISTINCT source_row_hash)
        FROM "{table_name}"
        """
    ).fetchone()
    if total != distinct_hashes:
        raise RuntimeError(f"{table_name} contains duplicate source hashes")
    invalid_dates = connection.execute(
        f"""
        SELECT COUNT(*) FROM "{table_name}"
        WHERE NOT REGEXP_FULL_MATCH(report_date, '[0-9]{{8}}')
        """
    ).fetchone()[0]
    if invalid_dates:
        raise RuntimeError(f"{table_name} contains invalid report dates")
    if total and expected_start and first_date < expected_start:
        raise RuntimeError(f"{table_name} contains rows before the requested range")
    if total and expected_end and last_date > expected_end:
        raise RuntimeError(f"{table_name} contains rows after the requested range")
    print(
        f"Validated {table_name}: {total:,} unique raw forecasts, "
        f"range {first_date or 'NULL'}-{last_date or 'NULL'}"
    )
    return total, first_date, last_date


def report_coverage(connection: duckdb.DuckDBPyConnection) -> None:
    total = connection.execute(
        f"SELECT COUNT(*) FROM {TABLE_NAME}"
    ).fetchone()[0]
    print("Column coverage:")
    for field in SOURCE_FIELDS:
        count = connection.execute(
            f'SELECT COUNT("{field}") FROM {TABLE_NAME}'
        ).fetchone()[0]
        missing_rate = (total - count) / total if total else 0.0
        print(f"  {field}: non_null={count:,}, missing={missing_rate:.2%}")
    stocks, organizations = connection.execute(
        f"SELECT COUNT(DISTINCT ts_code), COUNT(DISTINCT org_name) FROM {TABLE_NAME}"
    ).fetchone()
    nonstandard_quarters = connection.execute(
        f"""
        SELECT COUNT(*) FROM {TABLE_NAME}
        WHERE quarter IS NOT NULL
          AND NOT REGEXP_FULL_MATCH(quarter, '[0-9]{{4}}Q[1-4]')
        """
    ).fetchone()[0]
    forecast_conditions = " OR ".join(
        f'"{field}" IS NOT NULL' for field in PERIOD_FORECAST_FIELDS
    )
    nonstandard_forecasts = connection.execute(
        f"""
        SELECT COUNT(*) FROM {TABLE_NAME}
        WHERE quarter IS NOT NULL
          AND NOT REGEXP_FULL_MATCH(quarter, '[0-9]{{4}}Q[1-4]')
          AND ({forecast_conditions})
        """
    ).fetchone()[0]
    print(
        f"Stocks: {stocks:,}; organizations: {organizations:,}; "
        f"nonstandard quarter rows: {nonstandard_quarters:,} "
        f"({nonstandard_forecasts:,} with period forecasts)"
    )
    rd_rows, rd_stocks, negative_rd, extreme_rd = connection.execute(
        f"""
        SELECT
            COUNT(rd),
            COUNT(DISTINCT ts_code) FILTER (WHERE rd IS NOT NULL),
            COUNT(*) FILTER (WHERE rd < 0),
            COUNT(*) FILTER (WHERE rd > 100)
        FROM {TABLE_NAME}
        """
    ).fetchone()
    report_identity = (
        "ts_code, report_date, report_title, report_type, classify, "
        "org_name, author_name, create_time"
    )
    three_horizon_reports, positive_cagr_reports = connection.execute(
        f"""
        WITH annual_horizons AS (
            SELECT
                {report_identity},
                quarter,
                MEDIAN(eps) AS eps
            FROM {TABLE_NAME}
            WHERE REGEXP_FULL_MATCH(quarter, '[0-9]{{4}}Q4')
              AND CAST(SUBSTR(quarter, 1, 4) AS INTEGER)
                  >= CAST(SUBSTR(report_date, 1, 4) AS INTEGER)
            GROUP BY {report_identity}, quarter
        ),
        ranked AS (
            SELECT
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY {report_identity}
                    ORDER BY quarter
                ) AS horizon_number,
                COUNT(*) OVER (
                    PARTITION BY {report_identity}
                ) AS horizon_count
            FROM annual_horizons
        ),
        reports AS (
            SELECT
                {report_identity},
                MAX(horizon_count) AS horizon_count,
                MAX(eps) FILTER (WHERE horizon_number = 1) AS fy1_eps,
                MAX(eps) FILTER (WHERE horizon_number = 3) AS fy3_eps
            FROM ranked
            GROUP BY {report_identity}
        )
        SELECT
            COUNT(*) FILTER (WHERE horizon_count >= 3),
            COUNT(*) FILTER (
                WHERE horizon_count >= 3 AND fy1_eps > 0 AND fy3_eps > 0
            )
        FROM reports
        """
    ).fetchone()
    rd_stock_rate = rd_stocks / stocks if stocks else 0.0
    print(
        f"Forecast dividend yield rd: non_null={rd_rows:,}; "
        f"stocks={rd_stocks:,}/{stocks:,} ({rd_stock_rate:.2%}); "
        f"negative={negative_rd:,}; extreme_gt_100={extreme_rd:,}"
    )
    print(
        "Annual forecast report groups: "
        f"at_least_three_horizons={three_horizon_reports:,}; "
        f"positive_FY1_FY3_CAGR_inputs={positive_cagr_reports:,}"
    )


def run_self_tests() -> None:
    assert month_ranges("20240115", "20240302") == [
        ("20240115", "20240131"),
        ("20240201", "20240229"),
        ("20240301", "20240302"),
    ]
    sample = pd.DataFrame(
        [{
            "ts_code": "000001.SZ",
            "name": "平安银行",
            "report_date": "20240115",
            "report_title": "测试报告",
            "report_type": "公司报告",
            "classify": "一般报告",
            "org_name": "测试证券",
            "author_name": "测试分析师",
            "quarter": "2024Q4",
            "op_rt": "100",
            "op_pr": "20",
            "tp": "18",
            "np": "15",
            "eps": "1.5",
            "pe": "10",
            "rd": "2",
            "roe": "12",
            "ev_ebitda": "8",
            "rating": "买入",
            "max_price": "20",
            "min_price": "18",
            "imp_dg": "高",
            "create_time": "2024-01-15 20:00:00",
        }]
    )
    normalized = normalize_forecasts(pd.concat([sample, sample], ignore_index=True))
    assert len(normalized) == 1
    assert normalized.iloc[0]["eps"] == 1.5
    assert len(normalized.iloc[0]["source_row_hash"]) == 64
    nonstandard = sample.copy()
    nonstandard["quarter"] = "2024Q5"
    nonstandard["report_title"] = "原始异常报告期"
    normalized_nonstandard = normalize_forecasts(nonstandard)
    assert normalized_nonstandard.iloc[0]["quarter"] == "2024Q5"

    connection = duckdb.connect(":memory:")
    try:
        create_table(connection, TABLE_NAME)
        insert_batch(connection, TABLE_NAME, normalized)
        insert_batch(connection, TABLE_NAME, normalized)
        assert validate_table(connection, TABLE_NAME) == (
            1, "20240115", "20240115"
        )
    finally:
        connection.close()
    print("Self-tests passed: monthly ranges, normalization, hash, schema, upsert")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DATABASE_PATH)
    parser.add_argument("--start-date", help="inclusive YYYYMMDD report date")
    parser.add_argument("--end-date", help="inclusive YYYYMMDD report date")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        run_self_tests()
        return
    if not TUSHARE_API_KEY.strip():
        raise SystemExit("Please set the TUSHARE_API_KEY environment variable")
    if not args.database.exists():
        raise SystemExit(f"Database does not exist: {args.database}")

    requested_end = args.end_date or date.today().strftime("%Y%m%d")
    parse_yyyymmdd(requested_end)

    connection = duckdb.connect(str(args.database))
    staging_table = None
    preserve_staging_on_failure = False
    try:
        run_self_tests()
        existing = table_exists(connection, TABLE_NAME)
        if existing:
            validate_schema(connection, TABLE_NAME)
            latest = connection.execute(
                f"SELECT MAX(report_date) FROM {TABLE_NAME}"
            ).fetchone()[0]
            default_start = (
                parse_yyyymmdd(latest) - timedelta(days=7)
            ).strftime("%Y%m%d") if latest else FIRST_DATE
            requested_start = args.start_date or default_start
            staging_table = f"{TABLE_NAME}__stage"
        else:
            requested_start = args.start_date or FIRST_DATE
            staging_table = f"{TABLE_NAME}__build"
            preserve_staging_on_failure = True
        validation_start = requested_start
        parse_yyyymmdd(requested_start)
        if requested_start > requested_end:
            raise RuntimeError(
                f"start date {requested_start} is after end date {requested_end}"
            )

        if table_exists(connection, staging_table) and preserve_staging_on_failure:
            validate_schema(connection, staging_table)
            latest_staged = connection.execute(
                f"SELECT MAX(report_date) FROM {staging_table}"
            ).fetchone()[0]
            if latest_staged:
                staged_date = parse_yyyymmdd(latest_staged)
                resume_start = date(staged_date.year, staged_date.month, 1)
                requested_start = max(
                    parse_yyyymmdd(requested_start), resume_start
                ).strftime("%Y%m%d")
                print(
                    f"Resuming {staging_table} from {requested_start}; "
                    f"latest staged report date is {latest_staged}",
                    flush=True,
                )
        else:
            create_table(connection, staging_table)
        if requested_start > requested_end:
            raise RuntimeError(
                f"resume date {requested_start} is after end date {requested_end}"
            )
        pro = ts.pro_api(TUSHARE_API_KEY.strip())
        print(
            f"Downloading raw analyst forecasts for "
            f"{requested_start}-{requested_end} into {staging_table}",
            flush=True,
        )
        api_rows, calls = download_range(
            connection,
            pro,
            staging_table,
            requested_start,
            requested_end,
        )
        validate_table(
            connection,
            staging_table,
            expected_start=validation_start,
            expected_end=requested_end,
        )

        connection.execute("BEGIN TRANSACTION")
        try:
            if existing:
                connection.execute(
                    f"""
                    DELETE FROM {TABLE_NAME}
                    WHERE source_row_hash IN (
                        SELECT source_row_hash FROM {staging_table}
                    )
                    """
                )
                connection.execute(
                    f"INSERT INTO {TABLE_NAME} SELECT * FROM {staging_table}"
                )
                connection.execute(f"DROP TABLE {staging_table}")
            else:
                connection.execute(
                    f"ALTER TABLE {staging_table} RENAME TO {TABLE_NAME}"
                )
            connection.execute("COMMIT")
            staging_table = None
        except Exception:
            connection.execute("ROLLBACK")
            raise

        validate_table(connection, TABLE_NAME)
        connection.execute("CHECKPOINT")
        report_coverage(connection)
        print(
            f"Done. API rows={api_rows:,}; calls={calls:,}. "
            "Only raw sell-side forecast records were stored."
        )
    finally:
        if (
            staging_table
            and not preserve_staging_on_failure
            and table_exists(connection, staging_table)
        ):
            connection.execute(f'DROP TABLE "{staging_table}"')
        connection.close()


if __name__ == "__main__":
    main()
