"""按成分股快照分块计算标准动量及 RankIC 的优化实现。

优化只针对数据读取方式：同一成分股快照区间的行情一次读入内存，避免每个
信号日重新查询高度重叠的 252 日窗口。每日因子求和、未来收益和 Spearman
RankIC 的计算顺序仍与基础版保持一致，从而保证数值和并列排名完全相同。
"""

import re

import numpy as np
import pandas as pd

from data_manager import DataManager
from factor_base import FactorBase


class StandardMomentum(FactorBase):
    """计算标准动量信号以及信号与未来收益之间的每日 RankIC。"""

    def __init__(
        self,
        window: str = "252D",
        remove_recent_days: int = 22,
        data_fields: list[str] = ["close"],
    ):
        """保存因子参数。

        ``window`` 表示历史交易日窗口；``remove_recent_days`` 表示从窗口末端
        排除的最近交易日数量；``data_fields`` 保留与基础版一致的构造接口，
        当前标准动量实际使用复权收盘价 ``close``。
        """
        window_match = re.fullmatch(r"([1-9]\d*)D", window)
        if window_match is None:
            raise ValueError("optimized StandardMomentum requires a trading-day window")

        super().__init__(window=window, data_fields=data_fields)
        self.remove_recent_days = remove_recent_days

    def calculate_daily_factor(
        self,
        trade_date: str,
        symbols: list[str],
        market_data: dict[str, pd.DataFrame],
        history_start: str,
    ) -> pd.Series:
        """计算单日标准动量截面。"""

        # 全为空的日期需与原逐窗口查询的行集合保持一致；基类已统一清洗非正价格。
        close_window = market_data["close"].loc[
            history_start:trade_date, symbols
        ].dropna(how="all")
        daily_log_returns = np.log(close_window / close_window.shift(1))
        if len(daily_log_returns) <= self.remove_recent_days:
            raise ValueError(
                f"Not enough data for {self.window} window before {trade_date} "
                f"to remove {self.remove_recent_days} recent days"
            )

        # 排除靠近信号日的短期收益，再按股票累计历史日对数收益。
        daily_log_returns = daily_log_returns.iloc[: -self.remove_recent_days]
        return daily_log_returns.sum(axis=0).reindex(symbols)
