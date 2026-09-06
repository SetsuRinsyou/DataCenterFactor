"""Alpha158 最高价除以收盘价。"""

import numpy as np

from factor_base import FactorBase


class HIGH0(FactorBase):
    """最高价除以收盘价。"""

    def __init__(self):
        super().__init__(window="1D", data_fields=["high","close"])

    def calculate_daily_factor(self, trade_date, symbols, market_data, history_start):
        high = market_data["high"].reindex(index=[trade_date], columns=symbols).iloc[0]
        close = market_data["close"].reindex(index=[trade_date], columns=symbols).iloc[0]
        result = high / close
        return result.replace([np.inf, -np.inf], np.nan).reindex(symbols)
