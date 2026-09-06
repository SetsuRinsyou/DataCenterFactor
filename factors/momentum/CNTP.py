"""Alpha158 有效相邻价格对中的上涨日占比，缺失对不计入分母。"""

import re

import numpy as np

from factor_base import FactorBase


class CNTP(FactorBase):
    """有效相邻价格对中的上涨日占比，缺失对不计入分母。"""

    def __init__(self, window: str = "20D"):
        match = re.fullmatch(r"([1-9]\d*)D", window)
        if match is None:
            raise ValueError("window must be a positive trading-day window such as '20D'")
        self.window_days = int(match.group(1))
        super().__init__(window=window, data_fields=["close"])

    def calculate_daily_factor(self, trade_date, symbols, market_data, history_start):
        close = market_data["close"].loc[history_start:trade_date].reindex(columns=symbols).tail(self.window_days + 1)
        change = close.diff().iloc[1:]
        up = (change > 0).astype(float).where(change.notna()).mean()
        down = (change < 0).astype(float).where(change.notna()).mean()
        result = up
        return result.replace([np.inf, -np.inf], np.nan).reindex(symbols)
