"""Alpha158 开盘价除以收盘价。"""

import numpy as np

from factor_base import FactorBase


class OPEN0(FactorBase):
    """开盘价除以收盘价。"""

    def __init__(self):
        super().__init__(window="1D", data_fields=["open","close"])

    def calculate_daily_factor(self, trade_date, symbols, market_data, history_start):
        open = market_data["open"].reindex(index=[trade_date], columns=symbols).iloc[0]
        close = market_data["close"].reindex(index=[trade_date], columns=symbols).iloc[0]
        result = open / close
        return result.replace([np.inf, -np.inf], np.nan).reindex(symbols)
