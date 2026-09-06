"""Alpha158 收盘价相对高低价中点的两倍偏移除以日内振幅。"""

import numpy as np

from factor_base import FactorBase


class KSFT2(FactorBase):
    """收盘价相对高低价中点的两倍偏移除以日内振幅。"""

    def __init__(self):
        super().__init__(window="1D", data_fields=["high","low","close"])

    def calculate_daily_factor(self, trade_date, symbols, market_data, history_start):
        high = market_data["high"].reindex(index=[trade_date], columns=symbols).iloc[0]
        low = market_data["low"].reindex(index=[trade_date], columns=symbols).iloc[0]
        close = market_data["close"].reindex(index=[trade_date], columns=symbols).iloc[0]
        result = (2 * close - high - low) / (high - low + 1e-12)
        return result.replace([np.inf, -np.inf], np.nan).reindex(symbols)
