"""Alpha158 最低价首次出现的位置除以窗口长度，位置从最早的 1 开始。"""

import re

import numpy as np
import pandas as pd

from factor_base import FactorBase


class IMIN(FactorBase):
    """最低价首次出现的位置除以窗口长度，位置从最早的 1 开始。"""

    def __init__(self, window: str = "20D"):
        match = re.fullmatch(r"([1-9]\d*)D", window)
        if match is None:
            raise ValueError("window must be a positive trading-day window such as '20D'")
        self.window_days = int(match.group(1))
        super().__init__(window=window, data_fields=["low"])

    def calculate_daily_factor(self, trade_date, symbols, market_data, history_start):
        low = market_data["low"].loc[history_start:trade_date].reindex(columns=symbols).tail(self.window_days)
        low_position = pd.Series(
            np.argmin(low.fillna(np.inf).to_numpy(), axis=0) + 1,
            index=symbols, dtype=float,
        ).where(low.notna().any())
        result = low_position / self.window_days
        return result.replace([np.inf, -np.inf], np.nan).reindex(symbols)
