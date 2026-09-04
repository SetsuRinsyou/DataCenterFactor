"""论文 Sales-to-price 特征的日频实现。"""

import numpy as np

from factor_base import FactorBase


class SalesToPrice(FactorBase):
    """营业收入 TTM 除以信号日总市值。"""

    def __init__(self):
        super().__init__(window="1D", data_fields=["revenue_ttm", "total_mv"])

    def calculate_daily_factor(self, trade_date, symbols, market_data, history_start):
        sales = market_data["revenue_ttm"].reindex(
            index=[trade_date], columns=symbols
        ).iloc[0]
        market_equity = (
            market_data["total_mv"].reindex(
                index=[trade_date], columns=symbols
            ).iloc[0]
            * 10_000
        )
        return (sales / market_equity.where(market_equity != 0)).replace(
            [np.inf, -np.inf], np.nan
        ).reindex(symbols)
