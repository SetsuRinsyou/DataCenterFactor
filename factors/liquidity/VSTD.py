"""Alpha158 窗口成交量样本标准差除以当前成交量。"""

import re

import numpy as np

from factor_base import FactorBase


class VSTD(FactorBase):
    """窗口成交量样本标准差除以当前成交量。"""

    def __init__(self, window: str = "20D"):
        match = re.fullmatch(r"([1-9]\d*)D", window)
        if match is None:
            raise ValueError("window must be a positive trading-day window such as '20D'")
        self.window_days = int(match.group(1))
        super().__init__(window=window, data_fields=["vol"])

    def calculate_daily_factor(self, trade_date, symbols, market_data, history_start):
        vol = market_data["vol"].loc[history_start:trade_date].reindex(columns=symbols).tail(self.window_days)
        result = vol.std(ddof=1) / (vol.iloc[-1] + 1e-12)
        return result.replace([np.inf, -np.inf], np.nan).reindex(symbols)
