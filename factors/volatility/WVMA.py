"""Alpha158 绝对日收益率乘成交量的窗口样本标准差除以均值。"""

import re

import numpy as np

from factor_base import FactorBase


class WVMA(FactorBase):
    """绝对日收益率乘成交量的窗口样本标准差除以均值。"""

    def __init__(self, window: str = "20D"):
        match = re.fullmatch(r"([1-9]\d*)D", window)
        if match is None:
            raise ValueError("window must be a positive trading-day window such as '20D'")
        self.window_days = int(match.group(1))
        super().__init__(window=window, data_fields=["close","vol"])

    def calculate_daily_factor(self, trade_date, symbols, market_data, history_start):
        close = market_data["close"].loc[history_start:trade_date].reindex(columns=symbols).tail(self.window_days + 1)
        vol = market_data["vol"].loc[history_start:trade_date].reindex(columns=symbols).tail(self.window_days + 1)
        weighted_change = (close / close.shift(1) - 1).abs() * vol
        weighted_change = weighted_change.iloc[1:].replace([np.inf, -np.inf], np.nan)
        result = weighted_change.std(ddof=1) / (weighted_change.mean() + 1e-12)
        return result.replace([np.inf, -np.inf], np.nan).reindex(symbols)
