"""Alpha158 下影线占日内振幅的比例。"""

import numpy as np

from factor_base import FactorBase


class KLOW2(FactorBase):
    """下影线占日内振幅的比例。"""

    def __init__(self):
        super().__init__(window="1D", data_fields=["open","high","low","close"])

    def calculate_daily_factor(self, trade_date, symbols, market_data, history_start):
        open = market_data["open"].reindex(index=[trade_date], columns=symbols).iloc[0]
        high = market_data["high"].reindex(index=[trade_date], columns=symbols).iloc[0]
        low = market_data["low"].reindex(index=[trade_date], columns=symbols).iloc[0]
        close = market_data["close"].reindex(index=[trade_date], columns=symbols).iloc[0]
        result = (np.minimum(open, close) - low) / (high - low + 1e-12)
        return result.replace([np.inf, -np.inf], np.nan).reindex(symbols)
