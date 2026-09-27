import re
from pathlib import Path
import duckdb
import pandas as pd
import numpy as np
from collections import defaultdict
from itertools import groupby
from bisect import bisect_left, bisect_right


ANALYST_FORECAST_FIELDS = {
    "forecast_np_12m",
    "forecast_eps_12m",
    "forecast_eps_12m_std",
    "forecast_dividend_yield_12m",
    "forecast_eps_fy1_fy3_cagr",
    "forecast_revision_up_count",
    "forecast_revision_down_count",
    "forecast_revision_total_count",
}
DIVIDEND_FIELDS = {
    "cash_div_tax_ttm",
    "previous_month_end_raw_close",
}


class DataManager(object):

    def __init__(self, db_path: str, pool_name: str, start_date: str, end_date: str,
                 minute_db_path: str | None = None, factor_db_path: str | None = None):
        self.connection = None
        self.minute_connection = None
        self.factor_connection = None
        self.market_columns = set()
        self.financial_columns = set()
        self.financial_report_columns = set()
        self.industry_columns = set()
        self.analyst_forecast_columns = set()
        self.dividend_columns = set()
        self.minute_columns = set()
        self.factor_columns = set()

        try:
            self.initialize(db_path, pool_name, start_date, end_date, minute_db_path,
                            factor_db_path)
        except BaseException:
            try:
                self.close()
            except Exception:
                pass  # 保留原始初始化异常。
            raise

    def initialize(self, db_path: str, pool_name: str, start_date: str, end_date: str,
                    minute_db_path: str | None = None, factor_db_path: str | None = None):
        """初始化数据库连接、字段、交易日历和股票池。"""
        if pool_name != "zz500":
            raise ValueError("pool_name must be 'zz500'")

        self.db_path = db_path
        self.pool_name = pool_name
        self.minute_db_path = minute_db_path
        self.connection = duckdb.connect(db_path, read_only=True)
        if minute_db_path is not None:
            self.minute_connection = duckdb.connect(minute_db_path, read_only=True)
        if factor_db_path is not None:
            if not Path(factor_db_path).is_file():
                raise FileNotFoundError(f"factor database does not exist: {factor_db_path}")
            self.factor_connection = duckdb.connect(factor_db_path, read_only=True)

        # 检验各数据源中所需的表和字段。
        self.market_columns = {row[1] for row in self.connection.execute("PRAGMA table_info('market')").fetchall()}
        self.financial_columns = {row[1] for row in self.connection.execute("PRAGMA table_info('financial')").fetchall()}
        tables = {row[0] for row in self.connection.execute("SHOW TABLES").fetchall()}
        if "financial_report_event" in tables:
            self.financial_report_columns = {
                row[1] for row in self.connection.execute(
                    "PRAGMA table_info('financial_report_event')"
                ).fetchall()
            } - {"ts_code", "report_end_date", "source_type", "effective_date", "comp_type"}
        if "industry" in tables:
            self.industry_columns = {
                row[1]
                for row in self.connection.execute(
                    "PRAGMA table_info('industry')"
                ).fetchall()
            } - {"trade_date", "ts_code"}
        if "analyst_forecast" in tables:
            self.analyst_forecast_columns = ANALYST_FORECAST_FIELDS.copy()
        if "dividend" in tables:
            self.dividend_columns = DIVIDEND_FIELDS.copy()
        if self.minute_connection is not None:
            self.minute_columns = {row[1] for row in self.minute_connection.execute("PRAGMA table_info('market_1min')").fetchall()}
        if self.factor_connection is not None:
            self.factor_columns = {
                row[1] for row in self.factor_connection.execute(
                    "PRAGMA table_info('factor')"
                ).fetchall()
            } - {"trade_date", "code"}

        # 检查当前数据库支持的最早和最晚时间
        self.min_date, self.max_date = self.connection.execute("SELECT MIN(cal_date), MAX(cal_date) FROM calender").fetchone()
        if self.min_date is None or self.max_date is None:
            raise RuntimeError("calender table is empty")

        # 整个数据库的日历，方便从回测日之前开始的历史数据查询
        self.database_calender = self.connection.execute("SELECT cal_date, is_open FROM calender ORDER BY cal_date").fetchdf()
        self.calender = self.get_calendar(start_date, end_date)

        # 交易日列表用于按“交易日数量”移动，而不是按自然日做日期加减。
        # 反向位置字典使后续 T+1、T+6 和 252D 起点都能 O(1) 定位。
        self.open_dates = self.database_calender.loc[self.database_calender["is_open"] == 1, "cal_date"].tolist()
        self.open_date_positions = {
            trade_date: position
            for position, trade_date in enumerate(self.open_dates)
        }

        self.snapshot_dates, self.snapshot_symbols = self.get_unique_snapshot()
        self.excluded_symbols = self.get_excluded_symbols(start_date, end_date)

    def close(self):
        """关闭所有数据源；重复关闭无副作用。"""
        try:
            if self.minute_connection is not None:
                self.minute_connection.close()
        finally:
            try:
                if self.factor_connection is not None:
                    self.factor_connection.close()
            finally:
                if self.connection is not None:
                    self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.close()
        except Exception:
            if exc_type is None:
                raise
        return False

    def get_minute_data(self, trade_date: str, data_fields: list[str],
                        symbols: list[str], window_days: int = 1) -> pd.DataFrame:
        """读取截至 trade_date、含当天共 window_days 个交易日的原始分钟长表。"""
        if (isinstance(window_days, bool)
                or not isinstance(window_days, (int, np.integer))
                or window_days < 1):
            raise ValueError("window_days must be a positive integer")
        if not isinstance(trade_date, str) or not re.fullmatch(r"\d{8}", trade_date):
            raise ValueError("Minute dates must be YYYYMMDD")
        if trade_date not in self.open_date_positions:
            raise ValueError(f"Not a database trading day: {trade_date}")
        position = self.open_date_positions[trade_date]
        if position + 1 < window_days:
            raise ValueError(f"Not enough calendar history for {window_days} trading days through {trade_date}")
        allowed = self.minute_columns - {"trade_time", "ts_code"}
        if set(data_fields) - allowed:
            raise ValueError(f"Unknown minute fields: {sorted(set(data_fields) - allowed)}")
        if len(set(data_fields)) != len(data_fields):
            raise ValueError("Duplicate minute fields")
        if not symbols:
            return pd.DataFrame({
                "trade_time": pd.Series(dtype="datetime64[ns]"),
                "ts_code": pd.Series(dtype=str),
                **{field: pd.Series(dtype=float) for field in data_fields},
            }).set_index(["trade_time", "ts_code"])
        start = pd.to_datetime(self.open_dates[position - int(window_days) + 1], format="%Y%m%d")
        fields_sql = ", ".join(f'"{field}"' for field in ["trade_time", "ts_code", *data_fields])
        placeholders = ",".join("?" for _ in symbols)
        return self.minute_connection.execute(
            f"SELECT {fields_sql} FROM market_1min "
            f"WHERE trade_time >= ? AND trade_time < ? AND ts_code IN ({placeholders}) "
            "ORDER BY trade_time, ts_code",
            [start, pd.to_datetime(trade_date, format="%Y%m%d") + pd.Timedelta(days=1), *symbols],
        ).fetchdf().set_index(["trade_time", "ts_code"])

    def get_excluded_symbols(self, start_date: str, end_date: str):
        # ST 和停牌状态按日变化，但只需加载用户请求区间。使用 set 保存当天应
        # 排除的股票，可以在生成每日截面时快速判断，口径与基础版 get_symbols
        # 中的 SQL 条件完全一致。
        excluded_symbols = defaultdict(set)
        for trade_date, ts_code in self.connection.execute(
            """
            SELECT trade_date, ts_code
            FROM anomaly
            WHERE trade_date BETWEEN ? AND ?
                AND (value LIKE 'ST%' OR value LIKE '%SUSPENDED%')
            """,
            [start_date, end_date],
        ).fetchall():
            excluded_symbols[trade_date].add(ts_code)
        return dict(excluded_symbols)

    def get_unique_snapshot(self):
        # 成分股表每一行是一份快照：第一列为生效日期，后续列为股票代码。
        # 快照数量远少于交易日数量，因此初始化时一次读入，避免每天查询
        # “trade_date <= 当日”的最近一条记录。
        constituent_data = self.connection.execute(
            "SELECT * FROM zz500_constituents ORDER BY trade_date"
        ).fetchdf()
        snapshot_symbols = {}
        previous_symbols = None
        for _, row in constituent_data.iterrows():
            snapshot_date = row.iloc[0]
            symbols = [symbol for symbol in row.iloc[1:] if pd.notna(symbol)]
            if symbols != previous_symbols:
                snapshot_symbols[snapshot_date] = symbols
                previous_symbols = symbols
        snapshot_dates = list(snapshot_symbols)
        return snapshot_dates, snapshot_symbols

    def get_calendar(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get the trading calendar between start_date and end_date.
        """
        if start_date > end_date:
            raise ValueError("start_date must not be later than end_date")
        if start_date < self.min_date or end_date > self.max_date:
            raise ValueError(
                f"Date range must be within database calendar range "
                f"[{self.min_date}, {self.max_date}]"
            )

        return self.connection.execute(
            """
            SELECT cal_date, is_open
            FROM calender
            WHERE cal_date BETWEEN ? AND ?
            ORDER BY cal_date
            """,
            [start_date, end_date],
        ).fetchdf()

    def get_snapshot_date(self, trade_date: str) -> str:
        """返回不晚于 ``trade_date`` 的最近一组成分股快照日期。

        ``bisect_right`` 找到插入点后向前移动一位，等价于基础版 SQL 中的
        ``WHERE trade_date <= ? ORDER BY trade_date DESC LIMIT 1``。
        """
        snapshot_position = bisect_right(self.snapshot_dates, trade_date) - 1
        if snapshot_position < 0:
            raise ValueError(f"No constituent data available on or before {trade_date}")
        return self.snapshot_dates[snapshot_position]

    def get_symbols(self, trade_date: str) -> list[str]:
        """返回当日可进入因子截面的中证500成分股。

        先取得当日有效的名义成分股快照，再删除当日被标记为 ST 或停牌的股票。
        股票顺序沿用数据库快照顺序，以保证最终 DataFrame 的列顺序与基础版一致。
        """
        snapshot_date = self.get_snapshot_date(trade_date)
        excluded = self.excluded_symbols.get(trade_date, set())
        return [
            symbol
            for symbol in self.snapshot_symbols[snapshot_date]
            if symbol not in excluded
        ]

    def get_constituent_periods(self):
        """按成分股快照把请求区间内的交易日划分成连续区间。

        Yields
        ------
        signal_dates
            当前快照实际覆盖的连续交易日列表。
        constituent_symbols
            当前快照的名义成分股。这里暂不删除每日 ST/停牌股票，因为这些状态
            会在生成每日信号时单独应用。

        同一快照区间只需读取一次行情，这是优化版减少数据库重复扫描的核心。
        """
        trading_dates = self.calender.loc[
            self.calender["is_open"] == 1, "cal_date"
        ].tolist()
        for snapshot_date, dates in groupby(trading_dates, self.get_snapshot_date):
            yield list(dates), self.snapshot_symbols[snapshot_date]

    def get_pre_window_start_date(self, trade_date: str, pre_window: str) -> str:
        match = re.fullmatch(r"([1-9]\d*)([DMY])", pre_window)
        if match is None:
            raise ValueError("pre_window must be a positive integer followed by D, M, or Y")

        calendar_row = self.calender.loc[
            self.calender["cal_date"] == trade_date, "is_open"
        ]
        if calendar_row.empty:
            raise ValueError(f"trade_date is outside the database calendar: {trade_date}")
        if calendar_row.iloc[0] != 1:
            raise ValueError(f"trade_date is not a trading day: {trade_date}")

        value = int(match.group(1))
        unit = match.group(2)
        if unit == "D":
            trading_dates = self.database_calender.loc[
                (self.database_calender["cal_date"] < trade_date)
                & (self.database_calender["is_open"] == 1),
                "cal_date",
            ]
            if len(trading_dates) < value:
                raise ValueError(
                    f"Not enough trading-day history for {pre_window} before {trade_date}"
                )
            return trading_dates.iloc[-value]

        trade_timestamp = pd.to_datetime(trade_date, format="%Y%m%d")
        offset = (
            pd.DateOffset(months=value)
            if unit == "M"
            else pd.DateOffset(years=value)
        )
        start_date = (trade_timestamp - offset).strftime("%Y%m%d")
        if start_date < self.min_date:
            raise ValueError(
                f"Not enough database history for {pre_window} before {trade_date}"
            )
        return start_date

    def get_market_data(
        self,
        trade_dates: list[str],
        pre_window: str,
        data_fields: list[str],
        symbols: list[str],
    ) -> pd.DataFrame:
        missing_fields = [
            field for field in data_fields if field not in self.market_columns
        ]
        if missing_fields:
            raise ValueError(f"market table does not contain fields: {missing_fields}")
        if not symbols:
            raise ValueError("symbols list cannot be empty")

        selected_columns = ["trade_date", "ts_code"]
        for field in data_fields:
            if field != "adj_factor" and field not in selected_columns:
                selected_columns.append(field)

        start_date = self.get_pre_window_start_date(trade_dates[0], pre_window)
        end_date = trade_dates[-1]
        price_columns = [
            column
            for column in ("open", "high", "low", "close")
            if column in selected_columns
        ]
        query_columns = selected_columns.copy()
        if price_columns:
            query_columns.append("adj_factor")
        column_sql = ", ".join(f'"{column}"' for column in query_columns)
        symbol_placeholders = ", ".join("?" for _ in symbols)
        data = self.connection.execute(
            f"""
            SELECT {column_sql}
            FROM market
            WHERE trade_date BETWEEN ? AND ?
              AND ts_code IN ({symbol_placeholders})
            ORDER BY trade_date, ts_code
            """,
            [start_date, end_date, *symbols],
        ).fetchdf()
        if price_columns:
            data[price_columns] = data[price_columns].mul(data["adj_factor"], axis=0)
            data = data.drop(columns="adj_factor")
        return data.set_index(["trade_date", "ts_code"])

    def get_factor_data(
        self,
        trade_dates: list[str],
        pre_window: str,
        factor_fields: list[str],
        symbols: list[str],
    ) -> pd.DataFrame:
        """读取原始因子值，补齐窗口内交易日和请求股票，缺失值保留 NaN。"""
        if self.factor_connection is None:
            raise ValueError("factor_db_path is required to read factor fields")
        missing_fields = [field for field in factor_fields if field not in self.factor_columns]
        if missing_fields:
            raise ValueError(f"factor table does not contain fields: {missing_fields}")
        if len(set(factor_fields)) != len(factor_fields):
            raise ValueError("Duplicate factor fields")
        if not symbols:
            raise ValueError("symbols list cannot be empty")

        columns = ["trade_date", "code", *factor_fields]
        column_sql = ", ".join('"' + field.replace('"', '""') + '"' for field in columns)
        symbol_placeholders = ", ".join("?" for _ in symbols)
        start_date = self.get_pre_window_start_date(trade_dates[0], pre_window)
        data = self.factor_connection.execute(
            f"""
            SELECT {column_sql} FROM factor
            WHERE trade_date BETWEEN ? AND ? AND code IN ({symbol_placeholders})
            ORDER BY trade_date, code
            """,
            [start_date, trade_dates[-1], *symbols],
        ).fetchdf()
        history_dates = [
            day for day in self.open_dates if start_date <= day <= trade_dates[-1]
        ]
        # 保留无记录的交易日，避免下游 rolling/tail 跳过缺失日期。
        result_index = pd.MultiIndex.from_product(
            [history_dates, symbols], names=["trade_date", "ts_code"]
        )
        return (
            data.rename(columns={"code": "ts_code"})
            .set_index(["trade_date", "ts_code"])
            .reindex(result_index)
        )

    def get_financial_data(
        self,
        trade_dates: list[str],
        pre_window: str,
        data_fields: list[str],
        symbols: list[str],
    ) -> pd.DataFrame:
        missing_fields = [
            field for field in data_fields if field not in self.financial_columns
        ]
        if missing_fields:
            raise ValueError(
                f"financial table does not contain fields: {missing_fields}"
            )
        if not symbols:
            raise ValueError("symbols list cannot be empty")

        selected_columns = ["trade_date", "ts_code"]
        for field in data_fields:
            if field not in selected_columns:
                selected_columns.append(field)

        start_date = self.get_pre_window_start_date(trade_dates[0], pre_window)
        end_date = trade_dates[-1]
        column_sql = ", ".join(f'"{column}"' for column in selected_columns)
        symbol_placeholders = ", ".join("?" for _ in symbols)
        data = self.connection.execute(
            f"""
            SELECT {column_sql}
            FROM financial
            WHERE trade_date BETWEEN ? AND ?
              AND ts_code IN ({symbol_placeholders})
            ORDER BY trade_date, ts_code
            """,
            [start_date, end_date, *symbols],
        ).fetchdf()
        return data.set_index(["trade_date", "ts_code"])

    def get_financial_report_data(
        self,
        trade_dates: list[str],
        pre_window: str,
        data_fields: list[str],
        symbols: list[str],
    ) -> pd.DataFrame:
        """Return latest disclosed event values by day, report period, and stock.

        Source revisions are selected independently. An event dated on the signal
        day is unavailable until the next trading day.
        """
        if not self.financial_report_columns:
            raise ValueError("financial_report_event table is unavailable")
        missing = set(data_fields) - self.financial_report_columns
        if missing:
            raise ValueError(f"financial_report_event missing fields: {sorted(missing)}")
        if not trade_dates or not symbols or not data_fields:
            raise ValueError("trade_dates, data_fields, and symbols must be nonempty")
        start_date = self.get_pre_window_start_date(trade_dates[0], pre_window)
        report_start = (
            pd.to_datetime(start_date, format="%Y%m%d") - pd.DateOffset(years=1)
        ).strftime("%Y%m%d")
        selected = ", ".join(
            f'MAX("{field}") AS "{field}"' for field in data_fields
        )
        symbol_placeholders = ", ".join("?" for _ in symbols)
        date_placeholders = ", ".join("?" for _ in trade_dates)
        data = self.connection.execute(
            f"""
            WITH latest AS (
                SELECT d.cal_date AS trade_date, e.*,
                       ROW_NUMBER() OVER (
                           PARTITION BY d.cal_date, e.ts_code,
                                        e.report_end_date, e.source_type
                           ORDER BY e.effective_date DESC
                       ) AS revision_rank
                FROM calender AS d
                JOIN financial_report_event AS e
                  ON e.effective_date < d.cal_date
                 AND e.report_end_date <= d.cal_date
                 AND e.report_end_date >= ?
                WHERE d.cal_date IN ({date_placeholders}) AND d.is_open = 1
                  AND e.ts_code IN ({symbol_placeholders})
            )
            SELECT trade_date, report_end_date, ts_code, {selected}
            FROM latest WHERE revision_rank = 1
            GROUP BY trade_date, report_end_date, ts_code
            ORDER BY trade_date, report_end_date, ts_code
            """,
            [report_start, *trade_dates, *symbols],
        ).fetchdf()
        return data.set_index(["trade_date", "report_end_date", "ts_code"])

    def get_industry_data(
        self,
        trade_dates: list[str],
        pre_window: str,
        data_fields: list[str],
        symbols: list[str],
    ) -> pd.DataFrame:
        """读取按交易日和股票对齐的行业分类字段。"""
        missing_fields = [
            field for field in data_fields
            if field not in self.industry_columns
        ]
        if missing_fields:
            raise ValueError(
                f"industry table does not contain fields: {missing_fields}"
            )
        if len(set(data_fields)) != len(data_fields):
            raise ValueError("Duplicate industry fields")
        if not symbols:
            raise ValueError("symbols list cannot be empty")

        start_date = self.get_pre_window_start_date(trade_dates[0], pre_window)
        end_date = trade_dates[-1]
        selected_columns = ["trade_date", "ts_code", *data_fields]
        column_sql = ", ".join(f'"{column}"' for column in selected_columns)
        placeholders = ", ".join("?" for _ in symbols)
        data = self.connection.execute(
            f"""
            SELECT {column_sql}
            FROM industry
            WHERE trade_date BETWEEN ? AND ?
              AND ts_code IN ({placeholders})
            ORDER BY trade_date, ts_code
            """,
            [start_date, end_date, *symbols],
        ).fetchdf()
        history_dates = [
            date for date in self.open_dates if start_date <= date <= end_date
        ]
        result_index = pd.MultiIndex.from_product(
            [history_dates, symbols], names=["trade_date", "ts_code"]
        )
        return data.set_index(["trade_date", "ts_code"]).reindex(result_index)

    def get_analyst_forecast_data(
        self,
        trade_dates: list[str],
        pre_window: str,
        data_fields: list[str],
        symbols: list[str],
    ) -> pd.DataFrame:
        """将一对多卖方预测整理为可通过 data_fields 使用的日频输入。"""
        missing_fields = [
            field for field in data_fields
            if field not in self.analyst_forecast_columns
        ]
        if missing_fields:
            raise ValueError(
                f"Unknown analyst forecast fields: {missing_fields}"
            )
        if len(set(data_fields)) != len(data_fields):
            raise ValueError("Duplicate analyst forecast fields")
        if not symbols:
            raise ValueError("symbols list cannot be empty")

        start_date = self.get_pre_window_start_date(trade_dates[0], pre_window)
        end_date = trade_dates[-1]
        history_dates = [
            date for date in self.open_dates if start_date <= date <= end_date
        ]
        result_index = pd.MultiIndex.from_product(
            [history_dates, symbols], names=["trade_date", "ts_code"]
        )
        result = pd.DataFrame(index=result_index, columns=data_fields, dtype=float)
        revision_fields = {
            "forecast_revision_up_count",
            "forecast_revision_down_count",
            "forecast_revision_total_count",
        }
        for field in revision_fields.intersection(data_fields):
            result[field] = 0.0
        if not data_fields:
            return result

        requested_fields = set(data_fields)
        needs_forecast_state = bool(
            requested_fields - {"forecast_eps_fy1_fy3_cagr"}
        )
        needs_cagr = "forecast_eps_fy1_fy3_cagr" in requested_fields
        placeholders = ", ".join("?" for _ in symbols)
        baseline = self.connection.execute(
            f"""
            WITH candidates AS (
                SELECT
                    ts_code,
                    org_name,
                    CAST(SUBSTR(quarter, 1, 4) AS INTEGER) AS target_year,
                    np,
                    eps,
                    rd,
                    ROW_NUMBER() OVER (
                        PARTITION BY ts_code, org_name, target_year
                        ORDER BY report_date DESC,
                                 create_time DESC NULLS LAST,
                                 report_title DESC NULLS LAST,
                                 source_row_hash DESC
                    ) AS row_number
                FROM analyst_forecast
                WHERE ?
                  AND report_date < ?
                  AND ts_code IN ({placeholders})
                  AND org_name IS NOT NULL
                  AND org_name <> ''
                  AND REGEXP_FULL_MATCH(quarter, '[0-9]{{4}}Q4')
            )
            SELECT ts_code, org_name, target_year, np, eps, rd
            FROM candidates
            WHERE row_number = 1
            """,
            [needs_forecast_state, start_date, *symbols],
        ).fetchdf()
        baseline_cagr = self.connection.execute(
            f"""
            WITH report_rows AS (
                SELECT
                    ts_code,
                    report_date,
                    org_name,
                    COALESCE(author_name, '') AS author_name,
                    COALESCE(report_title, '') AS report_title,
                    COALESCE(CAST(create_time AS VARCHAR), '') AS create_time,
                    CAST(SUBSTR(quarter, 1, 4) AS INTEGER) AS target_year,
                    TRY_CAST(eps AS DOUBLE) AS eps
                FROM analyst_forecast
                WHERE ?
                  AND report_date < ?
                  AND ts_code IN ({placeholders})
                  AND org_name IS NOT NULL
                  AND org_name <> ''
                  AND REGEXP_FULL_MATCH(quarter, '[0-9]{{4}}Q4')
                  AND CAST(SUBSTR(quarter, 1, 4) AS INTEGER)
                      >= CAST(SUBSTR(report_date, 1, 4) AS INTEGER)
            ),
            report_first_year AS (
                SELECT
                    ts_code,
                    report_date,
                    org_name,
                    author_name,
                    report_title,
                    create_time,
                    MIN(target_year) AS first_year
                FROM report_rows
                GROUP BY ALL
            ),
            report_growth AS (
                SELECT
                    rows.ts_code,
                    rows.report_date,
                    rows.org_name,
                    rows.report_title,
                    rows.create_time,
                    years.first_year,
                    MAX(CASE
                        WHEN rows.target_year = years.first_year
                        THEN rows.eps
                    END) AS first_eps,
                    MAX(CASE
                        WHEN rows.target_year = years.first_year + 2
                        THEN rows.eps
                    END) AS third_eps
                FROM report_rows AS rows
                JOIN report_first_year AS years
                  ON rows.ts_code = years.ts_code
                 AND rows.report_date = years.report_date
                 AND rows.org_name = years.org_name
                 AND rows.author_name = years.author_name
                 AND rows.report_title = years.report_title
                 AND rows.create_time = years.create_time
                GROUP BY rows.ts_code, rows.report_date, rows.org_name,
                         rows.report_title, rows.create_time, years.first_year
            ),
            valid_growth AS (
                SELECT
                    *,
                    ROW_NUMBER() OVER (
                        PARTITION BY ts_code, org_name
                        ORDER BY report_date DESC, create_time DESC,
                                 report_title DESC
                    ) AS row_number
                FROM report_growth
                WHERE first_eps > 0 AND third_eps > 0
            )
            SELECT
                ts_code,
                org_name,
                first_year,
                SQRT(third_eps / first_eps) - 1 AS cagr
            FROM valid_growth
            WHERE row_number = 1
            """,
            [needs_cagr, start_date, *symbols],
        ).fetchdf()
        raw = self.connection.execute(
            f"""
            SELECT
                ts_code,
                report_date,
                report_title,
                org_name,
                author_name,
                quarter,
                np,
                eps,
                rd,
                create_time
            FROM analyst_forecast
            WHERE report_date BETWEEN ? AND ?
              AND ts_code IN ({placeholders})
              AND org_name IS NOT NULL
              AND org_name <> ''
              AND REGEXP_FULL_MATCH(quarter, '[0-9]{{4}}Q4')
            ORDER BY ts_code, report_date, org_name, create_time,
                     report_title, quarter
            """,
            [start_date, end_date, *symbols],
        ).fetchdf()

        baseline_states = defaultdict(dict)
        for row in baseline.itertuples(index=False):
            baseline_states[row.ts_code][
                (row.org_name, int(row.target_year))
            ] = {"np": row.np, "eps": row.eps, "rd": row.rd}
        baseline_cagr_states = defaultdict(dict)
        for row in baseline_cagr.itertuples(index=False):
            baseline_cagr_states[row.ts_code][row.org_name] = {
                "first_year": int(row.first_year),
                "cagr": row.cagr,
            }

        report_group_columns = [
            "ts_code",
            "report_date",
            "org_name",
            "author_name",
            "report_title",
            "create_time",
        ]
        reports_by_stock = defaultdict(list)
        if not raw.empty:
            raw["target_year"] = raw["quarter"].str[:4].astype(int)
            for identity, group in raw.groupby(
                report_group_columns, sort=False, dropna=False
            ):
                group = (
                    group.sort_values("target_year")
                    .drop_duplicates("target_year", keep="last")
                )
                report_year = int(str(identity[1])[:4])
                forward = group[group["target_year"] >= report_year]
                forecasts = {
                    int(row.target_year): {
                        "np": row.np,
                        "eps": row.eps,
                        "rd": row.rd,
                    }
                    for row in forward.itertuples(index=False)
                }
                cagr = np.nan
                cagr_first_year = None
                if forecasts:
                    first_year = min(forecasts)
                    first_eps = pd.to_numeric(
                        forecasts[first_year]["eps"], errors="coerce"
                    )
                    third_eps = pd.to_numeric(
                        forecasts.get(first_year + 2, {}).get("eps"),
                        errors="coerce",
                    )
                    if first_eps > 0 and third_eps > 0:
                        cagr = (third_eps / first_eps) ** 0.5 - 1
                        cagr_first_year = first_year
                reports_by_stock[identity[0]].append(
                    {
                        "report_date": identity[1],
                        "org_name": identity[2],
                        "create_time": identity[5],
                        "report_title": identity[4],
                        "forecasts": forecasts,
                        "cagr": cagr,
                        "cagr_first_year": cagr_first_year,
                    }
                )

        output_rows = []
        for ts_code in symbols:
            reports = sorted(
                reports_by_stock.get(ts_code, []),
                key=lambda item: (
                    item["report_date"],
                    pd.Timestamp.min
                    if pd.isna(item["create_time"])
                    else item["create_time"],
                    "" if pd.isna(item["report_title"]) else item["report_title"],
                ),
            )
            forecast_state = baseline_states.get(ts_code, {}).copy()
            cagr_state = baseline_cagr_states.get(ts_code, {}).copy()
            report_position = 0
            for trade_date in history_dates:
                up_count = 0
                down_count = 0
                total_count = 0
                while (
                    report_position < len(reports)
                    and reports[report_position]["report_date"] <= trade_date
                ):
                    report = reports[report_position]
                    organization = report["org_name"]
                    report_year = int(report["report_date"][:4])
                    report_timestamp = pd.to_datetime(
                        report["report_date"], format="%Y%m%d"
                    )
                    report_year_end = pd.Timestamp(
                        year=report_year, month=12, day=31
                    )
                    report_weight = (
                        (report_year_end - report_timestamp).days + 1
                    ) / report_year_end.dayofyear
                    previous_current = forecast_state.get(
                        (organization, report_year)
                    )
                    previous_following = forecast_state.get(
                        (organization, report_year + 1)
                    )
                    previous_blended_eps = np.nan
                    if (
                        previous_current is not None
                        and previous_following is not None
                    ):
                        previous_current_eps = pd.to_numeric(
                            previous_current["eps"], errors="coerce"
                        )
                        previous_following_eps = pd.to_numeric(
                            previous_following["eps"], errors="coerce"
                        )
                        if (
                            pd.notna(previous_current_eps)
                            and pd.notna(previous_following_eps)
                        ):
                            previous_blended_eps = (
                                report_weight * previous_current_eps
                                + (1 - report_weight)
                                * previous_following_eps
                            )
                    for target_year, forecast in report["forecasts"].items():
                        state_key = (organization, target_year)
                        forecast_state[state_key] = forecast
                    current = forecast_state.get((organization, report_year))
                    following = forecast_state.get(
                        (organization, report_year + 1)
                    )
                    current_blended_eps = np.nan
                    if current is not None and following is not None:
                        current_eps = pd.to_numeric(
                            current["eps"], errors="coerce"
                        )
                        following_eps = pd.to_numeric(
                            following["eps"], errors="coerce"
                        )
                        if pd.notna(current_eps) and pd.notna(following_eps):
                            current_blended_eps = (
                                report_weight * current_eps
                                + (1 - report_weight) * following_eps
                            )
                    if (
                        report["report_date"] >= start_date
                        and pd.notna(current_blended_eps)
                    ):
                        total_count += 1
                        if pd.notna(previous_blended_eps):
                            if current_blended_eps > previous_blended_eps:
                                up_count += 1
                            elif current_blended_eps < previous_blended_eps:
                                down_count += 1
                    if pd.notna(report["cagr"]):
                        cagr_state[organization] = {
                            "first_year": report["cagr_first_year"],
                            "cagr": report["cagr"],
                        }
                    report_position += 1

                year = int(trade_date[:4])
                timestamp = pd.to_datetime(trade_date, format="%Y%m%d")
                year_end = pd.Timestamp(year=year, month=12, day=31)
                days_in_year = year_end.dayofyear
                current_weight = ((year_end - timestamp).days + 1) / days_in_year

                organization_names = {
                    organization
                    for organization, target_year in forecast_state
                    if target_year in {year, year + 1}
                }
                blended = {"np": [], "eps": [], "rd": []}
                for organization in organization_names:
                    current = forecast_state.get((organization, year))
                    following = forecast_state.get((organization, year + 1))
                    if current is None or following is None:
                        continue
                    for field in blended:
                        current_value = pd.to_numeric(
                            current[field], errors="coerce"
                        )
                        following_value = pd.to_numeric(
                            following[field], errors="coerce"
                        )
                        if pd.notna(current_value) and pd.notna(following_value):
                            blended[field].append(
                                current_weight * current_value
                                + (1 - current_weight) * following_value
                            )

                row = {"trade_date": trade_date, "ts_code": ts_code}
                if "forecast_np_12m" in data_fields:
                    row["forecast_np_12m"] = (
                        float(np.median(blended["np"]))
                        if blended["np"] else np.nan
                    )
                if "forecast_eps_12m" in data_fields:
                    row["forecast_eps_12m"] = (
                        float(np.median(blended["eps"]))
                        if blended["eps"] else np.nan
                    )
                if "forecast_eps_12m_std" in data_fields:
                    row["forecast_eps_12m_std"] = (
                        float(np.std(blended["eps"], ddof=1))
                        if len(blended["eps"]) >= 2 else np.nan
                    )
                if "forecast_dividend_yield_12m" in data_fields:
                    row["forecast_dividend_yield_12m"] = (
                        float(np.median(blended["rd"]))
                        if blended["rd"] else np.nan
                    )
                if "forecast_eps_fy1_fy3_cagr" in data_fields:
                    cagr_values = [
                        value["cagr"] for value in cagr_state.values()
                        if value["first_year"] in {year, year + 1}
                        and pd.notna(value["cagr"])
                    ]
                    row["forecast_eps_fy1_fy3_cagr"] = (
                        float(np.median(cagr_values))
                        if cagr_values else np.nan
                    )
                if "forecast_revision_up_count" in data_fields:
                    row["forecast_revision_up_count"] = float(up_count)
                if "forecast_revision_down_count" in data_fields:
                    row["forecast_revision_down_count"] = float(down_count)
                if "forecast_revision_total_count" in data_fields:
                    row["forecast_revision_total_count"] = float(total_count)
                output_rows.append(row)

        if output_rows:
            output = pd.DataFrame(output_rows).set_index(
                ["trade_date", "ts_code"]
            )
            result.loc[output.index, data_fields] = output[data_fields]
        return result

    def get_dividend_data(
        self,
        trade_dates: list[str],
        pre_window: str,
        data_fields: list[str],
        symbols: list[str],
    ) -> pd.DataFrame:
        """返回历史税前 DPS 和上月月末未复权收盘价。"""
        missing_fields = [
            field for field in data_fields
            if field not in self.dividend_columns
        ]
        if missing_fields:
            raise ValueError(f"Unknown dividend fields: {missing_fields}")
        if len(set(data_fields)) != len(data_fields):
            raise ValueError("Duplicate dividend fields")
        if not symbols:
            raise ValueError("symbols list cannot be empty")

        start_date = self.get_pre_window_start_date(trade_dates[0], pre_window)
        end_date = trade_dates[-1]
        history_dates = [
            date for date in self.open_dates if start_date <= date <= end_date
        ]
        result_index = pd.MultiIndex.from_product(
            [history_dates, symbols], names=["trade_date", "ts_code"]
        )
        result = pd.DataFrame(index=result_index, columns=data_fields, dtype=float)
        placeholders = ", ".join("?" for _ in symbols)

        if "cash_div_tax_ttm" in data_fields:
            lower_ex_date = (
                pd.to_datetime(start_date, format="%Y%m%d")
                - pd.DateOffset(months=12)
            ).strftime("%Y%m%d")
            events = self.connection.execute(
                f"""
                SELECT ts_code, effective_date, ex_date, cash_div_tax
                FROM dividend
                WHERE div_proc = '实施'
                  AND cash_div_tax IS NOT NULL
                  AND effective_date IS NOT NULL
                  AND ex_date IS NOT NULL
                  AND ex_date > ?
                  AND ex_date <= ?
                  AND effective_date <= ?
                  AND ts_code IN ({placeholders})
                ORDER BY ts_code, ex_date, effective_date
                """,
                [lower_ex_date, end_date, end_date, *symbols],
            ).fetchdf()
            date_array = np.asarray(history_dates)
            for ts_code in symbols:
                differences = np.zeros(len(history_dates) + 1, dtype=float)
                stock_events = events.loc[events["ts_code"] == ts_code]
                for event in stock_events.itertuples(index=False):
                    activation_date = max(event.effective_date, event.ex_date)
                    expiry_date = (
                        pd.to_datetime(event.ex_date, format="%Y%m%d")
                        + pd.DateOffset(months=12)
                    ).strftime("%Y%m%d")
                    first = int(np.searchsorted(
                        date_array, activation_date, side="left"
                    ))
                    last = int(np.searchsorted(
                        date_array, expiry_date, side="left"
                    ))
                    if first < last and first < len(history_dates):
                        differences[first] += float(event.cash_div_tax)
                        differences[min(last, len(history_dates))] -= float(
                            event.cash_div_tax
                        )
                result.loc[
                    pd.IndexSlice[:, ts_code], "cash_div_tax_ttm"
                ] = np.cumsum(differences[:-1])

        if "previous_month_end_raw_close" in data_fields:
            previous_month_end = {}
            for trade_date in history_dates:
                month_start = f"{trade_date[:6]}01"
                position = bisect_left(self.open_dates, month_start) - 1
                previous_month_end[trade_date] = (
                    self.open_dates[position] if position >= 0 else None
                )
            price_dates = sorted({
                date for date in previous_month_end.values() if date is not None
            })
            if price_dates:
                date_placeholders = ", ".join("?" for _ in price_dates)
                prices = self.connection.execute(
                    f"""
                    SELECT trade_date, ts_code, close
                    FROM market
                    WHERE trade_date IN ({date_placeholders})
                      AND ts_code IN ({placeholders})
                    ORDER BY trade_date, ts_code
                    """,
                    [*price_dates, *symbols],
                ).fetchdf().set_index(["trade_date", "ts_code"])["close"]
                for trade_date in history_dates:
                    price_date = previous_month_end[trade_date]
                    if price_date is None:
                        continue
                    values = prices.reindex(
                        pd.MultiIndex.from_product(
                            [[price_date], symbols],
                            names=["trade_date", "ts_code"],
                        )
                    ).to_numpy()
                    result.loc[
                        pd.IndexSlice[trade_date, :],
                        "previous_month_end_raw_close",
                    ] = values
        return result

    def get_index_data(self, index_code: str, trade_dates: list[str], pre_window: str) -> pd.DataFrame:
        if not index_code:
            raise ValueError("index_code cannot be empty")

        start_date = self.get_pre_window_start_date(trade_dates[0], pre_window)
        end_date = trade_dates[-1]
        data = self.connection.execute(
            """
            SELECT trade_date, ts_code, close
            FROM "index"
            WHERE trade_date BETWEEN ? AND ?
              AND ts_code = ?
            ORDER BY trade_date, ts_code
            """,
            [start_date, end_date, index_code],
        ).fetchdf()
        return data.set_index(["trade_date", "ts_code"])

    def get_ff3_factor_data(
        self,
        trade_dates: list[str],
        pre_window: str,
    ) -> pd.DataFrame:
        """返回与行情窗口对齐的中证500日频 FF3 因子收益。"""
        if not trade_dates:
            raise ValueError("trade_dates cannot be empty")

        start_date = self.get_pre_window_start_date(trade_dates[0], pre_window)
        end_date = trade_dates[-1]
        history_dates = [
            date for date in self.open_dates if start_date <= date <= end_date
        ]

        index_data = self.get_index_data("000905.SH", trade_dates, pre_window)
        index_close = (
            index_data["close"]
            .droplevel("ts_code")
            .reindex(history_dates)
        )
        index_return = index_close.pct_change(fill_method=None).replace(
            [np.inf, -np.inf], np.nan
        )

        risk_free_data = self.connection.execute(
            """
            SELECT trade_date, MAX(gc001_weight) AS gc001_weight
            FROM financial
            WHERE trade_date BETWEEN ? AND ?
            GROUP BY trade_date
            ORDER BY trade_date
            """,
            [start_date, end_date],
        ).fetchdf()
        risk_free_rate = (
            risk_free_data.set_index("trade_date")["gc001_weight"]
            .reindex(history_dates)
            / 100
            / 365
        )

        formation_dates = []
        start_year = int(history_dates[0][:4]) - 1
        end_year = int(history_dates[-1][:4])
        for year in range(start_year, end_year + 1):
            june_dates = [
                date
                for date in self.open_dates
                if date.startswith(f"{year}06")
            ]
            if june_dates and june_dates[-1] <= end_date:
                formation_dates.append(june_dates[-1])

        formation_groups = {}
        all_symbols = set()
        group_names = [
            "small_low",
            "small_middle",
            "small_high",
            "big_low",
            "big_middle",
            "big_high",
        ]
        for formation_date in formation_dates:
            snapshot_date = self.get_snapshot_date(formation_date)
            symbols = self.snapshot_symbols[snapshot_date]
            placeholders = ", ".join("?" for _ in symbols)
            formation_data = self.connection.execute(
                f"""
                SELECT m.ts_code, m.total_mv,
                       f.total_hldr_eqy_exc_min_int_mrq AS book_equity
                FROM market AS m
                LEFT JOIN financial AS f
                  ON m.trade_date = f.trade_date AND m.ts_code = f.ts_code
                WHERE m.trade_date = ?
                  AND m.ts_code IN ({placeholders})
                """,
                [formation_date, *symbols],
            ).fetchdf().set_index("ts_code")

            market_equity = formation_data["total_mv"] * 10_000
            book_to_market = formation_data["book_equity"] / market_equity
            valid = market_equity.gt(0) & book_to_market.notna()
            market_equity = market_equity[valid]
            book_to_market = book_to_market[valid]
            if market_equity.empty:
                continue

            size_break = market_equity.median()
            low_break = book_to_market.quantile(0.3)
            high_break = book_to_market.quantile(0.7)
            groups = {group_name: [] for group_name in group_names}
            for symbol in market_equity.index:
                size = "small" if market_equity[symbol] <= size_break else "big"
                if book_to_market[symbol] <= low_break:
                    value = "low"
                elif book_to_market[symbol] > high_break:
                    value = "high"
                else:
                    value = "middle"
                groups[f"{size}_{value}"].append(symbol)

            formation_groups[formation_date] = groups
            all_symbols.update(symbols)

        portfolio_returns = pd.DataFrame(
            index=history_dates,
            columns=group_names,
            dtype=float,
        )
        if all_symbols:
            ordered_symbols = sorted(all_symbols)
            placeholders = ", ".join("?" for _ in ordered_symbols)
            market_data = self.connection.execute(
                f"""
                SELECT trade_date, ts_code, close, adj_factor, total_mv
                FROM market
                WHERE trade_date BETWEEN ? AND ?
                  AND ts_code IN ({placeholders})
                ORDER BY trade_date, ts_code
                """,
                [start_date, end_date, *ordered_symbols],
            ).fetchdf()

            def pivot(field: str) -> pd.DataFrame:
                return market_data.pivot(
                    index="trade_date", columns="ts_code", values=field
                ).reindex(index=history_dates, columns=ordered_symbols)

            adjusted_close = (pivot("close") * pivot("adj_factor")).where(
                lambda values: values > 0
            )
            stock_returns = adjusted_close.pct_change(fill_method=None).replace(
                [np.inf, -np.inf], np.nan
            )
            previous_market_values = pivot("total_mv").where(
                lambda values: values > 0
            ).shift(1)

            for trade_date in history_dates[1:]:
                active_dates = [
                    date for date in formation_groups if date < trade_date
                ]
                if not active_dates:
                    continue
                groups = formation_groups[max(active_dates)]
                for group_name, group_symbols in groups.items():
                    available = [
                        symbol
                        for symbol in group_symbols
                        if symbol in stock_returns.columns
                    ]
                    if not available:
                        continue
                    returns = stock_returns.loc[trade_date, available]
                    weights = previous_market_values.loc[trade_date, available]
                    valid = returns.notna() & weights.gt(0)
                    if valid.any():
                        portfolio_returns.loc[trade_date, group_name] = np.average(
                            returns[valid], weights=weights[valid]
                        )

        small_return = portfolio_returns[
            ["small_low", "small_middle", "small_high"]
        ].sum(axis=1, min_count=3) / 3
        big_return = portfolio_returns[
            ["big_low", "big_middle", "big_high"]
        ].sum(axis=1, min_count=3) / 3
        high_return = portfolio_returns[
            ["small_high", "big_high"]
        ].sum(axis=1, min_count=2) / 2
        low_return = portfolio_returns[
            ["small_low", "big_low"]
        ].sum(axis=1, min_count=2) / 2

        return pd.DataFrame(
            {
                "mkt": index_return - risk_free_rate,
                "smb": small_return - big_return,
                "hml": high_return - low_return,
                "rf": risk_free_rate,
            },
            index=pd.Index(history_dates, name="trade_date"),
        )

    def get_eval_data(
        self,
        trade_dates: list[str],
        lookforward_windows: int,
        symbols: list[str],
    ) -> pd.DataFrame:
        if (
            isinstance(lookforward_windows, bool)
            or not isinstance(lookforward_windows, (int, np.integer))
            or lookforward_windows <= 0
        ):
            raise ValueError("lookforward_windows must be a positive integer")

        result_index = pd.MultiIndex.from_product(
            [trade_dates, symbols], names=["trade_date", "ts_code"]
        )
        if not trade_dates or not symbols:
            return pd.DataFrame(index=result_index, columns=["return"], dtype=float)

        calendar_status = self.calender.set_index("cal_date")["is_open"]
        start_dates = []
        end_dates = []
        for trade_date in trade_dates:
            if trade_date not in calendar_status.index:
                raise ValueError(
                    f"trade_date is outside the database calendar: {trade_date}"
                )
            if calendar_status.loc[trade_date] != 1:
                raise ValueError(f"trade_date is not a trading day: {trade_date}")

            trade_date_index = self.open_date_positions[trade_date]
            end_date_index = trade_date_index + 1 + int(lookforward_windows)
            if end_date_index >= len(self.open_dates):
                raise ValueError(
                    f"Not enough future trading days for {lookforward_windows}-day "
                    f"evaluation after {trade_date}"
                )
            start_dates.append(self.open_dates[trade_date_index + 1])
            end_dates.append(self.open_dates[end_date_index])

        required_dates = list(dict.fromkeys(start_dates + end_dates))
        date_placeholders = ", ".join("?" for _ in required_dates)
        symbol_placeholders = ", ".join("?" for _ in symbols)
        data = self.connection.execute(
            f"""
            SELECT trade_date, ts_code, close, adj_factor
            FROM market
            WHERE trade_date IN ({date_placeholders})
              AND ts_code IN ({symbol_placeholders})
            ORDER BY trade_date, ts_code
            """,
            [*required_dates, *symbols],
        ).fetchdf()
        data["close"] = data["close"] * data["adj_factor"]
        prices = data.pivot(index="trade_date", columns="ts_code", values="close")
        start_prices = prices.reindex(index=start_dates, columns=symbols)
        end_prices = prices.reindex(index=end_dates, columns=symbols)
        start_prices.index = trade_dates
        end_prices.index = trade_dates
        returns = (end_prices / start_prices - 1).replace(
            [np.inf, -np.inf], np.nan
        )
        return pd.DataFrame(
            {"return": returns.to_numpy().reshape(-1)}, index=result_index
        )


class AllStockDataManager(DataManager):
    """保持 ``DataManager`` 接口不变，按交易日返回全部股票。"""

    def __init__(
        self,
        db_path: str,
        pool_name: str,
        start_date: str,
        end_date: str,
        minute_db_path: str | None = None,
        factor_db_path: str | None = None,
    ):
        if pool_name != "all":
            raise ValueError("pool_name must be 'all'")

        # 父类仍加载中证500快照，供 get_ff3_factor_data 构造既有口径的
        # SMB/HML；因子评价所用股票池由本类下面两个重写方法决定。
        super().__init__(db_path, "zz500", start_date, end_date, minute_db_path, factor_db_path)
        self.pool_name = pool_name
        self.all_symbols_by_date = {
            trade_date: list(symbols)
            for trade_date, symbols in self.connection.execute(
                """
                SELECT trade_date, list(ts_code ORDER BY ts_code) AS symbols
                FROM market
                WHERE trade_date BETWEEN ? AND ?
                GROUP BY trade_date
                ORDER BY trade_date
                """,
                [start_date, end_date],
            ).fetchall()
        }

    def get_symbols(self, trade_date: str) -> list[str]:
        """返回当日有行情且未被标记为ST或停牌的全部股票。"""
        excluded = self.excluded_symbols.get(trade_date, set())
        return [
            symbol
            for symbol in self.all_symbols_by_date.get(trade_date, [])
            if symbol not in excluded
        ]

    def get_constituent_periods(self):
        """按季度分块返回交易日及该季度出现过的股票并集。"""
        trading_dates = self.calender.loc[
            self.calender["is_open"] == 1, "cal_date"
        ].tolist()
        for _, dates in groupby(
            trading_dates,
            key=lambda date: (date[:4], (int(date[4:6]) - 1) // 3),
        ):
            period_dates = list(dates)
            period_symbols = sorted(
                {
                    symbol
                    for trade_date in period_dates
                    for symbol in self.all_symbols_by_date.get(trade_date, [])
                }
            )
            yield period_dates, period_symbols
