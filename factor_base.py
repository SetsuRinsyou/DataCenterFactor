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

    def compute_eval(
        self, data_manager: DataManager, forward_days: int
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """计算完整请求区间的因子矩阵、每日 RankIC 和五分层收益。"""
        signals = []
        ic_series = []
        group_returns = []

        missing_fields = [
            field
            for field in self.data_fields
            if field not in data_manager.market_columns
            and field not in data_manager.financial_columns
        ]
        if missing_fields:
            raise ValueError(
                f"market and financial tables do not contain fields: {missing_fields}"
            )
        market_field_names = [
            field
            for field in self.data_fields
            if field in data_manager.market_columns
        ]
        financial_field_names = [
            field
            for field in self.data_fields
            if field not in data_manager.market_columns
            and field in data_manager.financial_columns
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

        # 同一成分股快照区间内，行情和财务数据各自最多读取一次。
        for signal_dates, constituent_symbols in data_manager.get_constituent_periods():
            data_parts = []
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
            if data_parts:
                input_data = pd.concat(data_parts, axis=1).reindex(
                    columns=self.data_fields
                )
            else:
                input_data = data_manager.get_market_data(
                    signal_dates,
                    self.window,
                    [],
                    constituent_symbols,
                )
            return_ratios = data_manager.get_eval_data(
                signal_dates,
                forward_days,
                constituent_symbols,
            )

            # 每个字段只在区间级别转换一次，子类可直接按日期和股票切片。
            market_fields = {}
            for field in input_data.columns:
                field_data = input_data[field].unstack("ts_code").sort_index()
                if field in {"open", "high", "low", "close"}:
                    field_data = field_data.where(field_data > 0)
                market_fields[field] = field_data
            market_fields.update(shared_data)

            for date in signal_dates:
                # 名义成分股在区间内固定，ST 和停牌过滤仍以当天状态为准。
                symbols = data_manager.get_symbols(date)
                history_start = data_manager.get_pre_window_start_date(
                    date,
                    self.window,
                )
                signal = self.calculate_daily_factor(
                    date,
                    symbols,
                    market_fields,
                    history_start
                ).reindex(symbols)
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
        history_start: str
    ) -> pd.Series:
        """返回单个交易日的因子截面，索引必须为股票代码。"""
        raise NotImplementedError
