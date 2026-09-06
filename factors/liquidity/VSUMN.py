"""Alpha158 窗口累计减量占累计绝对成交量变化量的比例。"""

import re

import numpy as np

from factor_base import FactorBase


class VSUMN(FactorBase):
    """窗口累计减量占累计绝对成交量变化量的比例。"""

    def __init__(self, window: str = "20D"):
        match = re.fullmatch(r"([1-9]\d*)D", window)
        if match is None:
            raise ValueError("window must be a positive trading-day window such as '20D'")
        self.window_days = int(match.group(1))
        super().__init__(window=window, data_fields=["vol"])

    def calculate_daily_factor(self, trade_date, symbols, market_data, history_start):
        vol = market_data["vol"].loc[history_start:trade_date].reindex(columns=symbols).tail(self.window_days + 1)
        change = vol.diff().iloc[1:]
        up = change.clip(lower=0).sum(min_count=1)
        down = (-change).clip(lower=0).sum(min_count=1)
        denominator = change.abs().sum(min_count=1) + 1e-12
        result = down / denominator
        return result.replace([np.inf, -np.inf], np.nan).reindex(symbols)
