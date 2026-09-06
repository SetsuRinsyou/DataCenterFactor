"""Alpha158 窗口价格比与 log(成交量比+1) 的相关系数。"""

import re

import numpy as np

from factor_base import FactorBase


class CORD(FactorBase):
    """窗口价格比与 log(成交量比+1) 的相关系数。"""

    def __init__(self, window: str = "20D"):
        match = re.fullmatch(r"([1-9]\d*)D", window)
        if match is None:
            raise ValueError("window must be a positive trading-day window such as '20D'")
        self.window_days = int(match.group(1))
        super().__init__(window=window, data_fields=["close","vol"])

    def calculate_daily_factor(self, trade_date, symbols, market_data, history_start):
        close = market_data["close"].loc[history_start:trade_date].reindex(columns=symbols).tail(self.window_days + 1)
        vol = market_data["vol"].loc[history_start:trade_date].reindex(columns=symbols).tail(self.window_days + 1)
        price_ratio = (close / close.shift(1)).iloc[1:]
        volume_ratio = np.log(vol / vol.shift(1).where(vol.shift(1) > 0) + 1).iloc[1:]
        result = price_ratio.corrwith(volume_ratio)
        return result.replace([np.inf, -np.inf], np.nan).reindex(symbols)
