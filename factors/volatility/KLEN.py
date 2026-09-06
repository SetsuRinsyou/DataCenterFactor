"""Alpha158 日内高低价差除以开盘价。"""

import numpy as np

from factor_base import FactorBase


class KLEN(FactorBase):
    """日内高低价差除以开盘价。"""

    def __init__(self):
        super().__init__(window="1D", data_fields=["high","low","open"])

    def calculate_daily_factor(self, trade_date, symbols, market_data, history_start):
        high = market_data["high"].reindex(index=[trade_date], columns=symbols).iloc[0]
        low = market_data["low"].reindex(index=[trade_date], columns=symbols).iloc[0]
        open = market_data["open"].reindex(index=[trade_date], columns=symbols).iloc[0]
        result = (high - low) / open
        return result.replace([np.inf, -np.inf], np.nan).reindex(symbols)
