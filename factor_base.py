"""因子计算的通用流程。"""

import re
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from data_manager import DataManager


class FactorBase(ABC):
    """固化分块取数、逐日计算和 RankIC 评估流程的因子基类。"""

    def __init__(
        self,
        window: str,
        data_fields: list[str],
        index_code: str | None = None,
        requires_ff3: bool = False,
        minute_fields: list[str] | None = None,
        minute_window_days: int = 1,
        factor_fields: list[str] | None = None,
        financial_report_fields: list[str] | None = None,
    ):
        window_match = re.fullmatch(r"([1-9]\d*)([DMY])", window)
        if window_match is None:
            raise ValueError(
                "window must be a positive window such as '252D', '6M', or '1Y'"
            )

        self.window = window
        self.data_fields = data_fields
        self.index_code = index_code
        self.requires_ff3 = requires_ff3
        self.minute_fields = list(minute_fields or [])
        self.factor_fields = list(factor_fields or [])
        self.financial_report_fields = list(financial_report_fields or [])
        if (isinstance(minute_window_days, bool)
                or not isinstance(minute_window_days, (int, np.integer))
                or minute_window_days < 1):
            raise ValueError("minute_window_days must be a positive integer")
        self.minute_window_days = int(minute_window_days)

    def compute_eval(
        self, data_manager: DataManager, forward_days: int
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """计算完整请求区间的因子矩阵、每日 RankIC 和五分层收益。"""
        signals = []
        ic_series = []
        group_returns = []

        # 查找缺失字段，避免在计算中途报错
        missing_fields = []
        missing_fields.extend(
            field
            for field in self.data_fields
            if field not in data_manager.market_columns
            and field not in data_manager.financial_columns
            and field not in data_manager.industry_columns
            and field not in data_manager.analyst_forecast_columns
            and field not in data_manager.dividend_columns
        )
        if self.minute_fields:
            missing_fields.extend(
                field
                for field in self.minute_fields
                if field not in data_manager.minute_columns
            )
        if self.factor_fields:
            missing_fields.extend(
                field
                for field in self.factor_fields
                if field not in data_manager.factor_columns
            )
        if self.financial_report_fields:
            missing_fields.extend(
                field for field in self.financial_report_fields
                if field not in data_manager.financial_report_columns
            )
        if missing_fields:
            raise ValueError(
                f"Missing required fields in data_manager: {missing_fields}"
            )

        market_field_names = [
            field
            for field in self.data_fields
            if field in data_manager.market_columns
        ]
        financial_field_names = [
            field
            for field in self.data_fields
            if field in data_manager.financial_columns
        ]
        industry_field_names = [
            field
            for field in self.data_fields
            if field in data_manager.industry_columns
        ]
        analyst_forecast_field_names = [
            field
            for field in self.data_fields
            if field in data_manager.analyst_forecast_columns
        ]
        dividend_field_names = [
            field
            for field in self.data_fields
            if field in data_manager.dividend_columns
        ]

        all_signal_dates = data_manager.calender.loc[
            data_manager.calender["is_open"] == 1, "cal_date"
        ].tolist()
        shared_data = {}
        if self.index_code is not None:
            index_data = data_manager.get_index_data(
                self.index_code,
                all_signal_dates,
                self.window,
            )
            shared_data["index_close"] = (
                index_data["close"].unstack("ts_code").sort_index()
            )
        if self.requires_ff3:
            shared_data["ff3"] = data_manager.get_ff3_factor_data(
                all_signal_dates,
                self.window,
            )

        # 同一成分股快照区间内，每类日频数据各自最多读取一次。
        for signal_dates, constituent_symbols in data_manager.get_constituent_periods():
            data_parts = []
            minute_cache = {}
            report_values = None
            report_dates = set()
            if self.financial_report_fields:
                report_values = data_manager.get_financial_report_data(
                    signal_dates, self.window,
                    self.financial_report_fields, constituent_symbols,
                )
                report_dates = set(report_values.index.get_level_values("trade_date"))

            if market_field_names:
                data_parts.append(
                    data_manager.get_market_data(
                        signal_dates,
                        self.window,
                        market_field_names,
                        constituent_symbols,
                    )
                )
            if financial_field_names:
                data_parts.append(
                    data_manager.get_financial_data(
                        signal_dates,
                        self.window,
                        financial_field_names,
                        constituent_symbols,
                    )
                )
            if industry_field_names:
                data_parts.append(
                    data_manager.get_industry_data(
                        signal_dates,
                        self.window,
                        industry_field_names,
                        constituent_symbols,
                    )
                )
            if analyst_forecast_field_names:
                analyst_window = (
                    self.window
                    if any(
                        field.startswith("forecast_revision_")
                        for field in analyst_forecast_field_names
                    )
                    else "1D"
                )
                data_parts.append(
                    data_manager.get_analyst_forecast_data(
                        signal_dates,
                        analyst_window,
                        analyst_forecast_field_names,
                        constituent_symbols,
                    )
                )
            if dividend_field_names:
                data_parts.append(
                    data_manager.get_dividend_data(
                        signal_dates,
                        self.window,
                        dividend_field_names,
                        constituent_symbols,
                    )
                )
            if self.factor_fields:
                factor_values = data_manager.get_factor_data(
                    signal_dates, self.window, self.factor_fields, constituent_symbols
                )

            return_ratios = data_manager.get_eval_data(
                signal_dates,
                forward_days,
                constituent_symbols,
            )

            # 每个字段只在区间级别转换一次，子类可直接按日期和股票切片。
            if data_parts:
                input_data = pd.concat(data_parts, axis=1).reindex(
                    columns=self.data_fields
                )
            else:
                input_data = pd.DataFrame()

            market_fields = {}
            for field in input_data.columns:
                field_data = input_data[field].unstack("ts_code").sort_index()
                if field in {"open", "high", "low", "close"}:
                    field_data = field_data.where(field_data > 0)
                market_fields[field] = field_data
            market_fields.update(shared_data)

            factor_fields = {}
            if self.factor_fields:
                for field in self.factor_fields:
                    factor_fields[field] = factor_values[field].unstack("ts_code").sort_index()

            for date in signal_dates:
                # 名义成分股在区间内固定，ST 和停牌过滤仍以当天状态为准。
                symbols = data_manager.get_symbols(date)
                history_start = data_manager.get_pre_window_start_date(
                    date,
                    self.window,
                )
                if self.minute_fields:
                    position = data_manager.open_date_positions[date]
                    read_days = self.minute_window_days
                    if minute_cache:
                        last_position = data_manager.open_date_positions[max(minute_cache)]
                        read_days = min(read_days, position - last_position)
                    # 首次完整窗口由数据管理器校验；后续只读取缓存之后的新交易日。
                    minute_data = data_manager.get_minute_data(
                        date, self.minute_fields, constituent_symbols, window_days=read_days
                    )
                    minute_dates = data_manager.open_dates[
                        position - self.minute_window_days + 1:position + 1
                    ]
                    minute_cache.update({
                        day: frame for day, frame in minute_data.groupby(
                            minute_data.index.get_level_values("trade_time").strftime("%Y%m%d")
                        )
                    })
                    # 只保留窗口内日期；没有记录的交易日用空表占位。
                    minute_cache = {
                        day: minute_cache.get(day, minute_data.iloc[:0].copy())
                        for day in minute_dates
                    }
                    minute_data = pd.concat(minute_cache.values()).sort_index()
                    # 空交易日也保留在元数据中，因子按该日历对齐后再做跨日统计。
                    minute_data.attrs["trade_dates"] = tuple(minute_dates)
                    present = minute_cache[date].index.get_level_values("ts_code").intersection(symbols).nunique()

                daily_fields = {
                    field: values.loc[:date] for field, values in market_fields.items()
                }
                daily_kwargs = {}
                if self.factor_fields:
                    daily_kwargs["factor_data"] = {
                        field: values.loc[:date].reindex(columns=symbols)
                        for field, values in factor_fields.items()
                    }
                if self.minute_fields:
                    daily_kwargs["minute_data"] = minute_data
                if report_values is not None:
                    daily_kwargs["financial_report_data"] = (
                        report_values.xs(date, level="trade_date")
                        if date in report_dates
                        else report_values.iloc[:0].droplevel("trade_date")
                    )

                signal = self.calculate_daily_factor(
                    date, symbols, daily_fields, history_start, **daily_kwargs
                ).reindex(symbols)
                daily_kwargs.clear()
                if self.minute_fields:
                    print(f"MINUTE {date}: window={minute_dates[0]}..{date} "
                          f"target={len(symbols)} present={present} valid={int(np.isfinite(signal).sum())}",
                          flush=True)
                    del minute_data
                signal.name = date
                signals.append(signal)

                future_returns = return_ratios["return"].loc[date].reindex(symbols)
                daily_group_returns = self.calculate_group_returns(
                    signal,
                    future_returns,
                )
                daily_group_returns.name = date
                group_returns.append(daily_group_returns)

                aligned = pd.concat(
                    [signal.rename("signal"), future_returns.rename("return")],
                    axis=1,
                ).dropna()
                rank_ic = (
                    aligned["signal"].corr(aligned["return"], method="spearman")
                    if len(aligned) >= 2
                    else np.nan
                )
                ic_series.append((date, rank_ic))

        signal_frame = pd.DataFrame(signals)
        signal_frame.index.name = "trade_date"
        rank_ic_series = pd.Series(
            [rank_ic for _, rank_ic in ic_series],
            index=pd.Index([date for date, _ in ic_series], name="trade_date"),
            name="rank_ic",
            dtype=float,
        )
        group_returns_frame = pd.DataFrame(group_returns)
        group_returns_frame.index.name = "trade_date"
        return signal_frame, rank_ic_series, group_returns_frame

    def calculate_group_net_values(
        self,
        signal_frame: pd.DataFrame,
        data_manager: DataManager,
        rebalance_days: int,
    ) -> pd.DataFrame:
        """次日收盘调仓、持有期固定复权份额，逐交易日计算五组净值。

        无费用；停牌仓位保留，不能买入的目标份额留作现金。
        非停牌报价缺失时拒绝继续估值，避免静默丢失持仓。
        """
        if (
            isinstance(rebalance_days, bool)
            or not isinstance(rebalance_days, (int, np.integer))
            or rebalance_days <= 0
        ):
            raise ValueError("rebalance_days must be a positive integer")
        dates = data_manager.calender.loc[
            data_manager.calender["is_open"] == 1, "cal_date"
        ].tolist()
        labels = [f"group_{group}" for group in range(1, 6)]
        result = pd.DataFrame(
            index=pd.Index(dates[1:], name="trade_date"), columns=labels, dtype=float
        )
        if result.empty:
            return result
        suspended = {}
        for date, symbol in data_manager.connection.execute(
            """SELECT trade_date, ts_code FROM anomaly
               WHERE trade_date BETWEEN ? AND ? AND value LIKE '%SUSPENDED%'""",
            [dates[1], dates[-1]],
        ).fetchall():
            suspended.setdefault(date, set()).add(symbol)
        holdings = {label: pd.Series(dtype=float) for label in labels}
        cash = dict.fromkeys(labels, 1.0)
        last_prices = pd.Series(dtype=float)

        for signal_position in range(0, len(dates) - 1, int(rebalance_days)):
            signal_date = dates[signal_position]
            period_dates = dates[
                signal_position + 1:signal_position + 1 + int(rebalance_days)
            ]
            signal = (
                signal_frame.loc[signal_date].replace([np.inf, -np.inf], np.nan).dropna()
                if signal_date in signal_frame.index
                else pd.Series(dtype=float)
            )
            groups = (
                pd.qcut(signal.rank(method="first"), q=5, labels=labels)
                if len(signal) >= 5 else None
            )
            symbols = sorted(set(signal.index).union(
                *(set(position.index) for position in holdings.values())
            ))
            prices = (
                data_manager.get_market_data(period_dates, "1D", ["close"], symbols)
                ["close"].unstack("ts_code").reindex(index=period_dates, columns=symbols)
                if symbols else pd.DataFrame(index=period_dates)
            )
            prices = prices.where(np.isfinite(prices) & (prices > 0))
            for date in period_dates:
                unavailable = suspended.get(date, set())
                marks = prices.loc[date].copy()
                for symbol in unavailable.intersection(marks.index):
                    if pd.isna(marks[symbol]) and symbol in last_prices.index:
                        marks[symbol] = last_prices[symbol]
                for label in labels:
                    position = holdings[label]
                    held_prices = marks.reindex(position.index)
                    if held_prices.isna().any():
                        missing = held_prices.index[held_prices.isna()].tolist()
                        raise ValueError(f"Missing held-stock price on {date}: {missing}")
                    wealth = cash[label] + float((position * held_prices).sum())
                    result.loc[date, label] = wealth
                    if date != period_dates[0] or groups is None:
                        continue
                    # 停牌旧仓无法卖出，其余资金在目标股票之间等额分配。
                    locked = position[position.index.isin(unavailable)]
                    budget = wealth - float((locked * marks.reindex(locked.index)).sum())
                    targets = groups.index[
                        (groups == label) & ~groups.index.isin(locked.index)
                    ]
                    new_position = locked.copy()
                    cash[label] = budget
                    if len(targets):
                        allocation = budget / len(targets)
                        for symbol in targets:
                            if symbol in unavailable:
                                continue
                            if pd.isna(marks[symbol]):
                                raise ValueError(f"Missing entry price on {date}: {symbol}")
                            if allocation > 0:
                                new_position.loc[symbol] = allocation / marks[symbol]
                                cash[label] -= allocation
                    holdings[label] = new_position
                last_prices = marks.combine_first(last_prices)
        return result

    @staticmethod
    def calculate_group_returns(
        signal: pd.Series,
        future_returns: pd.Series,
    ) -> pd.Series:
        """按因子值从低到高五等分，返回各组的等权未来收益。"""
        group_labels = [f"group_{group}" for group in range(1, 6)]
        result = pd.Series(index=group_labels, dtype=float)
        valid_signal = signal.dropna()
        if len(valid_signal) < 5:
            return result

        # 先按因子分组，再计算收益，避免未来收益缺失影响分组边界。
        groups = pd.qcut(
            valid_signal.rank(method="first"),
            q=5,
            labels=group_labels,
        )
        aligned_returns = future_returns.reindex(valid_signal.index)
        for group_label in group_labels:
            result[group_label] = aligned_returns[groups == group_label].mean()
        return result

    @abstractmethod
    def calculate_daily_factor(
        self,
        trade_date: str,
        symbols: list[str],
        market_data: dict[str, pd.DataFrame],
        history_start: str,
        *,
        minute_data: pd.DataFrame | None = None,
        factor_data: dict[str, pd.DataFrame] | None = None,
        financial_report_data: pd.DataFrame | None = None,
    ) -> pd.Series:
        """返回日频股票截面。

        日频子类沿用原签名；声明 minute_fields 的子类接收 minute_data 关键字。
        声明 factor_fields 的子类接收 factor_data：按交易日补齐、截止当天、
        列对齐 symbols 的基础因子矩阵；用 history_start:trade_date 截取窗口，
        恰好 N 日（含当天）用 tail(N)，缺失值保持 NaN。
        声明 financial_report_fields 的子类接收 financial_report_data：
        索引为 (report_end_date, ts_code)，只含信号日前已披露事件。
        分钟长表索引为 (trade_time, ts_code)，窗口含当天共 minute_window_days
        个交易日；attrs['trade_dates'] 包含缺数据的交易日。价格及单位保持原样。
        """
        raise NotImplementedError
