"""Alpha158 窗口最高价除以当前收盘价。"""

import re

import numpy as np

from factor_base import FactorBase


class MAX(FactorBase):
    """窗口最高价除以当前收盘价。"""

    def __init__(self, window: str = "20D"):
        match = re.fullmatch(r"([1-9]\d*)D", window)
        if match is None:
            raise ValueError("window must be a positive trading-day window such as '20D'")
        self.window_days = int(match.group(1))
        super().__init__(window=window, data_fields=["high","close"])

    def calculate_daily_factor(self, trade_date, symbols, market_data, history_start):
        high = market_data["high"].loc[history_start:trade_date].reindex(columns=symbols).tail(self.window_days)
        close = market_data["close"].loc[history_start:trade_date].reindex(columns=symbols).tail(self.window_days)
        result = high.max() / close.iloc[-1]
        return result.replace([np.inf, -np.inf], np.nan).reindex(symbols)
