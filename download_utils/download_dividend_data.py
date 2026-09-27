#!/usr/bin/env python3
"""Download raw Tushare dividend events into the source DuckDB database."""

import argparse
import hashlib
import time
from collections.abc import Iterable
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


TABLE_NAME = "dividend"
DATE_FIELDS = [
    "end_date",
    "ann_date",
    "record_date",
    "ex_date",
    "pay_date",
    "div_listdate",
    "imp_ann_date",
    "base_date",
]
TEXT_FIELDS = ["ts_code", *DATE_FIELDS, "div_proc"]
NUMERIC_FIELDS = [
    "stk_div",
    "stk_bo_rate",
    "stk_co_rate",
    "cash_div",
    "cash_div_tax",
    "base_share",
]
SOURCE_FIELDS = [
    "ts_code",
    "end_date",
    "ann_date",
    "div_proc",
    "stk_div",
    "stk_bo_rate",
    "stk_co_rate",
    "cash_div",
    "cash_div_tax",
    "record_date",
    "ex_date",
    "pay_date",
    "div_listdate",
    "imp_ann_date",
    "base_date",
    "base_share",
]
TABLE_COLUMNS = [
    "event_key",
    "ts_code",
    "end_date",
    "ann_date",
    "effective_date",
    "div_proc",
    "stk_div",
    "stk_bo_rate",
    "stk_co_rate",
    "cash_div",
    "cash_div_tax",
    "record_date",
    "ex_date",
    "pay_date",
    "div_listdate",
    "imp_ann_date",
    "base_date",
    "base_share",
]


def create_table(
    connection: duckdb.DuckDBPyConnection,
    table_name: str = TABLE_NAME,
) -> None:
    connection.execute(
        f"""
        CREATE TABLE "{table_name}" (
            event_key VARCHAR PRIMARY KEY,
            ts_code VARCHAR NOT NULL,
            end_date VARCHAR,
            ann_date VARCHAR,
            effective_date VARCHAR,
            div_proc VARCHAR,
            stk_div DOUBLE,
            stk_bo_rate DOUBLE,
            stk_co_rate DOUBLE,
            cash_div DOUBLE,
            cash_div_tax DOUBLE,
            record_date VARCHAR,
            ex_date VARCHAR,
            pay_date VARCHAR,
            div_listdate VARCHAR,
            imp_ann_date VARCHAR,
            base_date VARCHAR,
            base_share DOUBLE
        )
        """
    )


def table_exists(
    connection: duckdb.DuckDBPyConnection,
    table_name: str = TABLE_NAME,
) -> bool:
    return table_name in {
        row[0] for row in connection.execute("SHOW TABLES").fetchall()
    }


def validate_schema(
    connection: duckdb.DuckDBPyConnection,
    table_name: str = TABLE_NAME,
) -> None:
    info = connection.execute(f"PRAGMA table_info('{table_name}')").fetchall()
    columns = [row[1] for row in info]
    primary_key = [row[1] for row in info if row[5]]
    if columns != TABLE_COLUMNS or primary_key != ["event_key"]:
        raise RuntimeError(
            f"{table_name} does not have the expected dividend event schema"
        )


def canonical_key_value(value) -> str:
    return "" if pd.isna(value) else str(value)


def event_key(row: pd.Series) -> str:
    event_date = next(
        (
            row[field]
            for field in ("ex_date", "record_date", "imp_ann_date", "ann_date")
            if not pd.isna(row[field])
        ),
        None,
    )
    payload = "\x1f".join(
        canonical_key_value(value)
        for value in (
            row["ts_code"],
            row["end_date"],
            row["div_proc"],
            event_date,
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def numeric_precision_score(row: pd.Series) -> int:
    score = 0
    for field in NUMERIC_FIELDS:
        value = row[field]
        if pd.isna(value):
            continue
        text = format(float(value), ".15g")
        score += len(text.replace(".", "").replace("-", ""))
    return score


def rows_differ_only_by_numeric_precision(frame: pd.DataFrame) -> bool:
    text_columns = [
        column for column in TABLE_COLUMNS
        if column not in {"event_key", *NUMERIC_FIELDS}
    ]
    if len(frame[text_columns].drop_duplicates()) != 1:
        return False
    for field in NUMERIC_FIELDS:
        values = frame[field].dropna().astype(float)
        if len(values) < 2:
            continue
        reference = values.iloc[0]
        tolerance = max(1e-9, abs(reference) * 1e-6)
        if ((values - reference).abs() > tolerance).any():
            return False
    return True


def normalize_dividends(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=TABLE_COLUMNS)
    missing = set(SOURCE_FIELDS) - set(frame.columns)
    if missing:
        raise RuntimeError(
            f"dividend response is missing fields: {sorted(missing)}"
        )

    data = frame[SOURCE_FIELDS].copy()
    for field in TEXT_FIELDS:
        data[field] = data[field].astype("string")
        data[field] = data[field].replace(r"^\s*$", pd.NA, regex=True)
    if data["ts_code"].isna().any():
        raise RuntimeError("dividend returned rows without ts_code")

    for field in DATE_FIELDS:
        invalid = data[field].notna() & ~data[field].str.fullmatch(r"\d{8}")
        if invalid.any():
            values = data.loc[invalid, field].drop_duplicates().head(5).tolist()
            raise RuntimeError(f"dividend returned invalid {field}: {values}")
        pd.to_datetime(data[field], format="%Y%m%d", errors="raise")

    for field in NUMERIC_FIELDS:
        source_not_null = data[field].notna()
        converted = pd.to_numeric(data[field], errors="coerce")
        if (source_not_null & converted.isna()).any():
            raise RuntimeError(f"dividend returned invalid numeric values in {field}")
        data[field] = converted

    data.insert(
        data.columns.get_loc("div_proc"),
        "effective_date",
        data["imp_ann_date"].fillna(data["ann_date"]),
    )
    data.insert(0, "event_key", data.apply(event_key, axis=1))
    data = data.drop_duplicates(TABLE_COLUMNS)
    duplicate_keys = data["event_key"].duplicated(keep=False)
    if duplicate_keys.any():
        revisions = []
        for _, group in data.groupby("event_key", sort=False, dropna=False):
            if len(group) == 1:
                revisions.append(group.iloc[0])
                continue
            revision_order = group[
                [
                    "effective_date",
                    "ann_date",
                    "imp_ann_date",
                    "record_date",
                    "ex_date",
                    "pay_date",
                    "base_date",
                ]
            ].fillna("").agg("\x1f".join, axis=1)
            latest = group.loc[revision_order == revision_order.max()]
            latest = latest.drop_duplicates(TABLE_COLUMNS)
            if len(latest) > 1 and rows_differ_only_by_numeric_precision(latest):
                scores = latest.apply(numeric_precision_score, axis=1)
                latest = latest.loc[scores == scores.max()].drop_duplicates(
                    TABLE_COLUMNS
                )
            if len(latest) != 1:
                if not (latest["div_proc"] == "实施").any():
                    for _, revision in latest.iterrows():
                        revision = revision.copy()
                        payload = "\x1f".join(
                            canonical_key_value(revision[column])
                            for column in TABLE_COLUMNS
                            if column != "event_key"
                        )
                        revision["event_key"] = hashlib.sha256(
                            f"{revision['event_key']}\x1f{payload}".encode("utf-8")
                        ).hexdigest()
                        revisions.append(revision)
                    continue
                examples = latest[
                    ["ts_code", "end_date", "div_proc", "ann_date", "ex_date"]
                ]
                raise RuntimeError(
                    "dividend returned indistinguishable conflicting revisions:\n"
                    f"{examples.to_string(index=False)}"
                )
            revisions.append(latest.iloc[0])
        data = pd.DataFrame(revisions, columns=TABLE_COLUMNS)
    return data[TABLE_COLUMNS].reset_index(drop=True)


def call_dividend(pro, ts_code: str) -> pd.DataFrame:
    for retry in range(MAX_RETRIES + 1):
        try:
            frame = pro.dividend(
                ts_code=ts_code,
                fields=",".join(SOURCE_FIELDS),
            )
            time.sleep(REQUEST_INTERVAL_SECONDS)
            return frame if frame is not None else pd.DataFrame(columns=SOURCE_FIELDS)
        except Exception as exc:
            if retry == MAX_RETRIES:
                raise RuntimeError(
                    f"dividend failed after retries for {ts_code}"
                ) from exc
            time.sleep(2 ** (retry + 1))
    raise AssertionError("unreachable")


def upsert_batch(
    connection: duckdb.DuckDBPyConnection,
    frame: pd.DataFrame,
    table_name: str = TABLE_NAME,
) -> None:
    if frame.empty:
        return
    connection.register("dividend_batch", frame)
    try:
        assignments = ", ".join(
            f'"{column}" = excluded."{column}"'
            for column in TABLE_COLUMNS
            if column != "event_key"
        )
        connection.execute(
            f"""
            INSERT INTO "{table_name}"
            SELECT {', '.join(f'"{column}"' for column in TABLE_COLUMNS)}
            FROM dividend_batch
            ON CONFLICT (event_key) DO UPDATE SET {assignments}
            """
        )
    finally:
        connection.unregister("dividend_batch")


def validate_table(
    connection: duckdb.DuckDBPyConnection,
    table_name: str = TABLE_NAME,
) -> tuple[int, int]:
    validate_schema(connection, table_name)
    total, keys, invalid_effective_dates = connection.execute(
        f"""
        SELECT
            COUNT(*),
            COUNT(DISTINCT event_key),
            COUNT(*) FILTER (
                WHERE effective_date IS DISTINCT FROM
                      COALESCE(imp_ann_date, ann_date)
            )
        FROM "{table_name}"
        """
    ).fetchone()
    if total != keys:
        raise RuntimeError(f"{table_name} contains duplicate event keys")
    if invalid_effective_dates:
        raise RuntimeError(
            f"{table_name} contains {invalid_effective_dates:,} invalid effective dates"
        )
    for field in DATE_FIELDS:
        invalid = connection.execute(
            f"""
            SELECT COUNT(*) FROM "{table_name}"
            WHERE "{field}" IS NOT NULL
              AND NOT REGEXP_FULL_MATCH("{field}", '[0-9]{{8}}')
            """
        ).fetchone()[0]
        if invalid:
            raise RuntimeError(
                f"{table_name} contains {invalid:,} invalid {field} values"
            )
    return total, keys


def report_coverage(
    connection: duckdb.DuckDBPyConnection,
    table_name: str = TABLE_NAME,
) -> None:
    row = connection.execute(
        f"""
        SELECT
            COUNT(*),
            COUNT(DISTINCT ts_code),
            MIN(ann_date),
            MAX(ann_date),
            COUNT(cash_div_tax),
            COUNT(*) FILTER (WHERE div_proc = '实施'),
            COUNT(*) FILTER (
                WHERE div_proc = '实施'
                  AND cash_div_tax IS NOT NULL
                  AND ex_date IS NOT NULL
            )
        FROM "{table_name}"
        """
    ).fetchone()
    print(
        f"Dividend coverage: rows={row[0]:,}; stocks={row[1]:,}; "
        f"ann_date={row[2] or 'NULL'}-{row[3] or 'NULL'}; "
        f"cash_div_tax={row[4]:,}; implemented={row[5]:,}; "
        f"DTOP-ready={row[6]:,}"
    )


def download_stocks(
    connection: duckdb.DuckDBPyConnection,
    pro,
    stock_codes: Iterable[str],
    table_name: str = TABLE_NAME,
) -> tuple[int, int]:
    codes = list(stock_codes)
    api_rows = 0
    normalized_rows = 0
    for index, ts_code in enumerate(codes, start=1):
        raw = call_dividend(pro, ts_code)
        normalized = normalize_dividends(raw)
        upsert_batch(connection, normalized, table_name)
        api_rows += len(raw)
        normalized_rows += len(normalized)
        if index == 1 or index % 50 == 0 or index == len(codes):
            print(
                f"[{index}/{len(codes)}] {ts_code}: raw={len(raw):,}, "
                f"normalized={len(normalized):,}",
                flush=True,
            )
    return api_rows, normalized_rows


def run_self_tests() -> None:
    sample = pd.DataFrame(
        [{
            "ts_code": "000001.SZ",
            "end_date": "20231231",
            "ann_date": "20240315",
            "div_proc": "实施",
            "stk_div": 0.0,
            "stk_bo_rate": 0.0,
            "stk_co_rate": 0.0,
            "cash_div": 0.25,
            "cash_div_tax": 0.30,
            "record_date": "20240619",
            "ex_date": "20240620",
            "pay_date": "20240620",
            "div_listdate": None,
            "imp_ann_date": "20240613",
            "base_date": "20240619",
            "base_share": 100.0,
        }]
    )
    normalized = normalize_dividends(pd.concat([sample, sample], ignore_index=True))
    assert len(normalized) == 1
    assert normalized.iloc[0]["effective_date"] == "20240613"
    revised = sample.copy()
    revised["cash_div_tax"] = 0.31
    revised["ann_date"] = "20240316"
    revised["pay_date"] = "20240621"
    revised_normalized = normalize_dividends(revised)
    assert normalized.iloc[0]["event_key"] == revised_normalized.iloc[0]["event_key"]
    combined = normalize_dividends(pd.concat([sample, revised], ignore_index=True))
    assert len(combined) == 1
    assert combined.iloc[0]["cash_div_tax"] == 0.31
    proposals = pd.concat([sample, sample], ignore_index=True)
    proposals["div_proc"] = "股东大会通过"
    proposals[["record_date", "ex_date", "pay_date", "imp_ann_date"]] = None
    proposals.loc[1, "stk_div"] = 0.5
    proposals.loc[1, "base_share"] = 80.0
    distinct_proposals = normalize_dividends(proposals)
    assert len(distinct_proposals) == 2
    assert distinct_proposals["event_key"].nunique() == 2

    connection = duckdb.connect(":memory:")
    try:
        create_table(connection)
        upsert_batch(connection, normalized)
        upsert_batch(connection, revised_normalized)
        assert validate_table(connection) == (1, 1)
        value = connection.execute(
            "SELECT cash_div_tax, pay_date FROM dividend"
        ).fetchone()
        assert value == (0.31, "20240621")
    finally:
        connection.close()
    print("Self-tests passed: normalization, stable event key, schema, upsert")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DATABASE_PATH)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--sample-ts-code",
        help="fetch one stock and validate in memory without writing the source database",
    )
    parser.add_argument(
        "--start-code",
        help="resume the ordered market universe from this stock code (inclusive)",
    )
    args = parser.parse_args()

    run_self_tests()
    if args.self_test:
        return
    if not TUSHARE_API_KEY.strip():
        raise SystemExit("Please set the TUSHARE_API_KEY environment variable")
    pro = ts.pro_api(TUSHARE_API_KEY.strip())

    if args.sample_ts_code:
        connection = duckdb.connect(":memory:")
        try:
            create_table(connection)
            raw = call_dividend(pro, args.sample_ts_code)
            normalized = normalize_dividends(raw)
            upsert_batch(connection, normalized)
            upsert_batch(connection, normalized)
            validate_table(connection)
            report_coverage(connection)
            print(
                f"Validated {args.sample_ts_code}: raw={len(raw):,}, "
                f"unique_events={len(normalized):,}; source database unchanged."
            )
        finally:
            connection.close()
        return

    if not args.database.exists():
        raise SystemExit(f"Database does not exist: {args.database}")
    connection = duckdb.connect(str(args.database))
    try:
        if table_exists(connection):
            validate_schema(connection)
        else:
            create_table(connection)
        stock_codes = [
            row[0]
            for row in connection.execute(
                "SELECT DISTINCT ts_code FROM market ORDER BY ts_code"
            ).fetchall()
        ]
        if args.start_code:
            stock_codes = [
                ts_code for ts_code in stock_codes if ts_code >= args.start_code
            ]
        print(
            f"Downloading dividend history for {len(stock_codes):,} market stocks",
            flush=True,
        )
        api_rows, normalized_rows = download_stocks(
            connection,
            pro,
            stock_codes,
        )
        validate_table(connection)
        connection.execute("CHECKPOINT")
        report_coverage(connection)
        print(
            f"Done. API rows={api_rows:,}; normalized rows={normalized_rows:,}. "
            "Raw dividend events were upserted idempotently."
        )
    finally:
        connection.close()


if __name__ == "__main__":
    main()
