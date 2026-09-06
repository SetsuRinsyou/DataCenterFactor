"""Alpha158 当前价格对时间趋势回归的末点残差除以当前价格。"""

import re

import numpy as np
import pandas as pd

from factor_base import FactorBase


class RESI(FactorBase):
    """当前价格对时间趋势回归的末点残差除以当前价格。"""

    def __init__(self, window: str = "20D"):
        match = re.fullmatch(r"([1-9]\d*)D", window)
        if match is None:
            raise ValueError("window must be a positive trading-day window such as '20D'")
        self.window_days = int(match.group(1))
        super().__init__(window=window, data_fields=["close"])

    def calculate_daily_factor(self, trade_date, symbols, market_data, history_start):
        close = market_data["close"].loc[history_start:trade_date].reindex(columns=symbols).tail(self.window_days)
        # 对时间位置回归，不是市场收益率 Beta；缺失点保留其原始时间位置。
        y = close.to_numpy(dtype=float)
        valid = np.isfinite(y)
        x = np.arange(1, len(close) + 1, dtype=float)[:, None]
        count = valid.sum(axis=0)
        mean_x = np.sum(np.where(valid, x, 0), axis=0) / np.maximum(count, 1)
        mean_y = np.sum(np.where(valid, y, 0), axis=0) / np.maximum(count, 1)
        dx = np.where(valid, x - mean_x, 0)
        dy = np.where(valid, y - mean_y, 0)
        xx = np.sum(dx * dx, axis=0)
        xy = np.sum(dx * dy, axis=0)
        slope = np.divide(xy, xx, out=np.full_like(xy, np.nan), where=(xx > 0) & (count >= 2))
        fitted = mean_y + slope * (len(close) - mean_x)
        result = (close.iloc[-1] - pd.Series(fitted, index=symbols)) / close.iloc[-1]
        return result.replace([np.inf, -np.inf], np.nan).reindex(symbols)
