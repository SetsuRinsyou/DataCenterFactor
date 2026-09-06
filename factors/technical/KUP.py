"""Alpha158 上影线长度除以开盘价。"""

import numpy as np

from factor_base import FactorBase


class KUP(FactorBase):
    """上影线长度除以开盘价。"""

    def __init__(self):
        super().__init__(window="1D", data_fields=["open","high","close"])

    def calculate_daily_factor(self, trade_date, symbols, market_data, history_start):
        open = market_data["open"].reindex(index=[trade_date], columns=symbols).iloc[0]
        high = market_data["high"].reindex(index=[trade_date], columns=symbols).iloc[0]
        close = market_data["close"].reindex(index=[trade_date], columns=symbols).iloc[0]
        result = (high - np.maximum(open, close)) / open
        return result.replace([np.inf, -np.inf], np.nan).reindex(symbols)
