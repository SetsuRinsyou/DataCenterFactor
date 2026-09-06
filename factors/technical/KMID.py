"""Alpha158 实体涨跌幅 (close-open)/open。"""

import numpy as np

from factor_base import FactorBase


class KMID(FactorBase):
    """实体涨跌幅 (close-open)/open。"""

    def __init__(self):
        super().__init__(window="1D", data_fields=["open","close"])

    def calculate_daily_factor(self, trade_date, symbols, market_data, history_start):
        open = market_data["open"].reindex(index=[trade_date], columns=symbols).iloc[0]
        close = market_data["close"].reindex(index=[trade_date], columns=symbols).iloc[0]
        result = (close - open) / open
        return result.replace([np.inf, -np.inf], np.nan).reindex(symbols)
