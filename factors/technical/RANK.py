"""Alpha158 当前收盘价在历史窗口内的百分位排名，平均秩处理并列。"""

import re

import numpy as np

from factor_base import FactorBase


class RANK(FactorBase):
    """当前收盘价在历史窗口内的百分位排名，平均秩处理并列。"""

    def __init__(self, window: str = "20D"):
        match = re.fullmatch(r"([1-9]\d*)D", window)
        if match is None:
            raise ValueError("window must be a positive trading-day window such as '20D'")
        self.window_days = int(match.group(1))
        super().__init__(window=window, data_fields=["close"])

    def calculate_daily_factor(self, trade_date, symbols, market_data, history_start):
        close = market_data["close"].loc[history_start:trade_date].reindex(columns=symbols).tail(self.window_days)
        result = close.rank(pct=True).iloc[-1]
        return result.replace([np.inf, -np.inf], np.nan).reindex(symbols)
