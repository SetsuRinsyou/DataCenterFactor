"""将DuckDB中的因子按因子和股票导出为CSV文件。"""

import argparse
import csv
from datetime import datetime
from pathlib import Path

import duckdb


SCRIPT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按因子和股票导出指定日期区间的因子值")
    parser.add_argument("--factor-name", help="只导出factor表中的指定因子列")
    parser.add_argument("--start-date", required=True, help="开始日期，格式YYYYMMDD")
    parser.add_argument("--end-date", required=True, help="结束日期，格式YYYYMMDD")
    parser.add_argument("--output-dir", required=True, help="CSV输出目录")
    parser.add_argument(
        "--factor-db",
        default=str(SCRIPT_ROOT / "data" / "factor_results.duckdb"),
        help="因子数据库路径",
    )
    parser.add_argument(
        "--calendar-db",
        default=str(SCRIPT_ROOT / "data" / "src_data.duckdb"),
        help="包含calender表的数据库路径",
    )
    return parser.parse_args()


def validate_date(value: str, argument_name: str) -> None:
    try:
        datetime.strptime(value, "%Y%m%d")
    except ValueError as exc:
        raise ValueError(f"{argument_name} must use YYYYMMDD format") from exc


def quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def main() -> None:
    args = parse_args()
    validate_date(args.start_date, "start-date")
    validate_date(args.end_date, "end-date")
    if args.start_date > args.end_date:
        raise ValueError("start-date must not be later than end-date")

    factor_db = Path(args.factor_db).expanduser().resolve()
    calendar_db = Path(args.calendar_db).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not factor_db.is_file():
        raise FileNotFoundError(f"factor database does not exist: {factor_db}")
    if not calendar_db.is_file():
        raise FileNotFoundError(f"calendar database does not exist: {calendar_db}")

    factor_connection = duckdb.connect(str(factor_db), read_only=True)
    try:
        factor_columns = [
            row[1]
            for row in factor_connection.execute(
                "PRAGMA table_info('factor')"
            ).fetchall()
            if row[1] not in {"trade_date", "code"}
        ]
        if args.factor_name and args.factor_name not in factor_columns:
            raise ValueError(
                f"factor table does not contain column: {args.factor_name}"
            )
        factor_names = [args.factor_name] if args.factor_name else factor_columns

        output_dir.mkdir(parents=True, exist_ok=True)
        if any(output_dir.rglob("*.csv")):
            raise FileExistsError(
                f"output directory already contains CSV files: {output_dir}"
            )

        calendar_connection = duckdb.connect(str(calendar_db), read_only=True)
        try:
            open_dates = [
                row[0]
                for row in calendar_connection.execute(
                    """
                    SELECT cal_date
                    FROM calender
                    WHERE is_open = 1
                      AND cal_date BETWEEN ? AND ?
                    ORDER BY cal_date
                    """,
                    [args.start_date, args.end_date],
                ).fetchall()
            ]
        finally:
            calendar_connection.close()
        formatted_dates = {
            date: datetime.strptime(date, "%Y%m%d").strftime("%Y-%m-%d")
            for date in open_dates
        }
        available_date_by_signal = {
            date: formatted_dates[next_date]
            for date, next_date in zip(open_dates, open_dates[1:])
        }

        total_rows = 0
        total_values = 0
        for factor_name in factor_names:
            if Path(factor_name).name != factor_name:
                raise ValueError(f"invalid factor name for filename: {factor_name}")
            factor_column = quote_identifier(factor_name)
            factor_dir = output_dir / factor_name / "factors"
            factor_dir.mkdir(parents=True, exist_ok=True)
            result = factor_connection.execute(
                f"""
                SELECT trade_date, code, {factor_column}
                FROM factor
                WHERE trade_date BETWEEN ? AND ?
                ORDER BY code, trade_date
                """,
                [args.start_date, args.end_date],
            )
            current_code = None
            output_file = None
            writer = None
            stock_count = 0
            row_count = 0
            value_count = 0
            try:
                while rows := result.fetchmany(10_000):
                    for signal_date, code, factor_value in rows:
                        if signal_date not in formatted_dates:
                            raise ValueError(
                                f"calendar does not contain signal date: {signal_date}"
                            )
                        if code != current_code:
                            if output_file is not None:
                                output_file.close()
                            if Path(code).name != code:
                                raise ValueError(
                                    f"invalid stock code for filename: {code}"
                                )
                            output_file = (factor_dir / f"{code}.csv").open(
                                "w", newline="", encoding="utf-8"
                            )
                            writer = csv.writer(output_file, lineterminator="\n")
                            writer.writerow(
                                [
                                    "signal_date",
                                    "available_date",
                                    f"{factor_name}_raw",
                                    f"{factor_name}_neutralization",
                                ]
                            )
                            current_code = code
                            stock_count += 1
                        writer.writerow(
                            [
                                formatted_dates[signal_date],
                                available_date_by_signal.get(signal_date),
                                factor_value,
                                None,
                            ]
                        )
                        row_count += 1
                        if factor_value is not None:
                            value_count += 1
            finally:
                if output_file is not None:
                    output_file.close()
            total_rows += row_count
            total_values += value_count
            print(
                f"{factor_name}: {stock_count} 只股票, "
                f"{row_count} 行, {value_count} 个因子值"
            )

        print(f"导出因子数: {len(factor_names)}")
        print(f"信号区间: {args.start_date} - {args.end_date}")
        print(f"导出数据行: {total_rows}")
        print(f"导出非空因子值: {total_values}")
        print(f"输出目录: {output_dir}")
    finally:
        factor_connection.close()


if __name__ == "__main__":
    main()
