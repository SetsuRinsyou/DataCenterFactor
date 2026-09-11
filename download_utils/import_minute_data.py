#!/usr/bin/env python3
"""Import annual per-stock minute CSV files directly from ZIP archives."""

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path
from zipfile import ZipFile

import duckdb
import numpy as np
import pandas as pd


DATABASE_PATH = Path(__file__).resolve().parents[1] / "data" / "market_1min.duckdb"
COLUMNS = ["ts_code", "trade_time", "open", "high", "low", "close", "vol", "amount"]


def read_minutes(archive: ZipFile, member, code: str, year: int) -> pd.DataFrame:
    with archive.open(member) as source:
        frame = pd.read_csv(source, encoding="utf-8-sig")
    missing = set(COLUMNS[1:]) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if "ts_code" in frame and not frame["ts_code"].eq(code).all():
        raise ValueError("CSV stock code differs from member name")
    frame = frame[COLUMNS[1:]].copy()
    frame.insert(0, "ts_code", code)
    frame["trade_time"] = pd.to_datetime(
        frame["trade_time"], format="%Y-%m-%d %H:%M:%S", errors="raise"
    )
    for column in COLUMNS[2:]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if frame.isna().any().any() or not np.isfinite(frame[COLUMNS[2:]].to_numpy()).all():
        raise ValueError("Null or non-finite minute values")
    if not frame["trade_time"].dt.year.eq(year).all():
        raise ValueError("Minute timestamp is outside the archive year")
    if frame.duplicated(["ts_code", "trade_time"]).any():
        raise ValueError("Duplicate stock/timestamp keys")
    if (frame[["vol", "amount"]] < 0).any().any():
        raise ValueError("Negative volume or amount")
    if (frame["high"] < frame[["open", "close", "low"]].max(axis=1)).any() or (
        frame["low"] > frame[["open", "close", "high"]].min(axis=1)
    ).any():
        raise ValueError("Invalid OHLC relationship")
    return frame.sort_values("trade_time", ignore_index=True)


def import_archives(connection, paths: list[Path], codes: list[str] | None) -> None:
    connection.execute("""
        CREATE TABLE IF NOT EXISTS market_1min (
            ts_code VARCHAR NOT NULL, trade_time TIMESTAMP NOT NULL,
            open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE,
            vol DOUBLE, amount DOUBLE,
            PRIMARY KEY (ts_code, trade_time)
        )
    """)
    connection.execute("""
        CREATE TABLE IF NOT EXISTS market_1min_import_progress (
            zip_name VARCHAR NOT NULL, member_path VARCHAR NOT NULL,
            crc BIGINT NOT NULL, file_size BIGINT NOT NULL,
            row_count BIGINT NOT NULL, completed_at TIMESTAMP NOT NULL,
            PRIMARY KEY (zip_name, member_path)
        )
    """)
    requested = set(codes or [])
    seen = set()
    imported = skipped = total_rows = 0
    first_time = last_time = None
    try:
        for path in paths:
            location = str(path)
            try:
                if not re.fullmatch(r"\d{4}\.zip", path.name):
                    raise ValueError("Archive name must be YYYY.zip")
                year = int(path.stem)
                with ZipFile(path) as archive:
                    names = set()
                    for member in archive.infolist():
                        location = f"{path}!{member.filename}"
                        if member.is_dir():
                            continue
                        match = re.fullmatch(r"(\d{4})/(\d{6}\.(?:SH|SZ|BJ))\.csv", member.filename)
                        if match is None or int(match[1]) != year:
                            raise ValueError("Expected YYYY/000001.SZ.csv with matching year")
                        if member.filename in names:
                            raise ValueError("Duplicate ZIP member name")
                        names.add(member.filename)
                        code = match[2]
                        if requested and code not in requested:
                            continue
                        seen.add(code)
                        done = connection.execute(
                            "SELECT crc, file_size FROM market_1min_import_progress "
                            "WHERE zip_name = ? AND member_path = ?",
                            [path.name, member.filename],
                        ).fetchone()
                        if done is not None:
                            if done != (member.CRC, member.file_size):
                                raise ValueError("Source fingerprint changed; existing data was not overwritten")
                            skipped += 1
                            print(f"SKIP {location}", flush=True)
                            continue
                        frame = read_minutes(archive, member, code, year)
                        connection.register("minute_batch", frame)
                        try:
                            connection.execute("BEGIN TRANSACTION")
                            try:
                                connection.execute(
                                    "INSERT INTO market_1min (" + ", ".join(COLUMNS) + ") "
                                    "SELECT " + ", ".join(COLUMNS) + " FROM minute_batch"
                                )
                                connection.execute(
                                    "INSERT INTO market_1min_import_progress VALUES (?, ?, ?, ?, ?, ?)",
                                    [path.name, member.filename, member.CRC, member.file_size,
                                     len(frame), datetime.now()],
                                )
                                connection.execute("COMMIT")
                            except Exception:
                                connection.execute("ROLLBACK")
                                raise
                        finally:
                            connection.unregister("minute_batch")
                        imported += 1
                        total_rows += len(frame)
                        if not frame.empty:
                            start, end = frame["trade_time"].min(), frame["trade_time"].max()
                            first_time = start if first_time is None else min(first_time, start)
                            last_time = end if last_time is None else max(last_time, end)
                            date_range = f"{start}..{end}"
                        else:
                            date_range = "empty"
                        print(f"SAVED {location}: {len(frame)} rows, {date_range}", flush=True)
            except Exception as exc:
                raise RuntimeError(f"{location}: {exc}") from None
        if requested - seen:
            raise ValueError(f"Requested codes absent from archives: {sorted(requested - seen)}")
    finally:
        print(f"SUMMARY imported={imported} skipped={skipped} rows={total_rows} "
              f"range={first_time}..{last_time}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", nargs="+", required=True, type=Path, dest="archives")
    parser.add_argument("--codes", nargs="+", help="Omit to import all stocks in the supplied archives")
    parser.add_argument("--database", type=Path, default=DATABASE_PATH)
    args = parser.parse_args()
    if args.codes and any(not re.fullmatch(r"\d{6}\.(?:SH|SZ|BJ)", code) for code in args.codes):
        parser.error("Stock codes must look like 000001.SZ, 600000.SH or 920001.BJ")
    try:
        with duckdb.connect(str(args.database)) as connection:
            import_archives(connection, args.archives, args.codes)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
