"""Alpha158 当前收盘价在窗口最高价和最低价区间内的位置。"""

import re

import numpy as np

from factor_base import FactorBase


class RSV(FactorBase):
    """当前收盘价在窗口最高价和最低价区间内的位置。"""

    def __init__(self, window: str = "20D"):
        match = re.fullmatch(r"([1-9]\d*)D", window)
        if match is None:
            raise ValueError("window must be a positive trading-day window such as '20D'")
        self.window_days = int(match.group(1))
        super().__init__(window=window, data_fields=["close","high","low"])

    def calculate_daily_factor(self, trade_date, symbols, market_data, history_start):
        close = market_data["close"].loc[history_start:trade_date].reindex(columns=symbols).tail(self.window_days)
        high = market_data["high"].loc[history_start:trade_date].reindex(columns=symbols).tail(self.window_days)
        low = market_data["low"].loc[history_start:trade_date].reindex(columns=symbols).tail(self.window_days)
        result = (close.iloc[-1] - low.min()) / (high.max() - low.min() + 1e-12)
        return result.replace([np.inf, -np.inf], np.nan).reindex(symbols)
