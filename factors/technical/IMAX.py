"""Alpha158 最高价首次出现的位置除以窗口长度，位置从最早的 1 开始。"""

import re

import numpy as np
import pandas as pd

from factor_base import FactorBase


class IMAX(FactorBase):
    """最高价首次出现的位置除以窗口长度，位置从最早的 1 开始。"""

    def __init__(self, window: str = "20D"):
        match = re.fullmatch(r"([1-9]\d*)D", window)
        if match is None:
            raise ValueError("window must be a positive trading-day window such as '20D'")
        self.window_days = int(match.group(1))
        super().__init__(window=window, data_fields=["high"])

    def calculate_daily_factor(self, trade_date, symbols, market_data, history_start):
        high = market_data["high"].loc[history_start:trade_date].reindex(columns=symbols).tail(self.window_days)
        high_position = pd.Series(
            np.argmax(high.fillna(-np.inf).to_numpy(), axis=0) + 1,
            index=symbols, dtype=float,
        ).where(high.notna().any())
        result = high_position / self.window_days
        return result.replace([np.inf, -np.inf], np.nan).reindex(symbols)
