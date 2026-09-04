"""论文 Price-to-cost-margin 特征的日频实现。"""

import numpy as np

from factor_base import FactorBase


class PriceToCostMargin(FactorBase):
    """毛利润 TTM 除以营业收入 TTM。"""

    def __init__(self):
        super().__init__(
            window="1D", data_fields=["revenue_ttm", "oper_cost_ttm"]
        )

    def calculate_daily_factor(self, trade_date, symbols, market_data, history_start):
        sales = market_data["revenue_ttm"].reindex(
            index=[trade_date], columns=symbols
        ).iloc[0]
        cost = market_data["oper_cost_ttm"].reindex(
            index=[trade_date], columns=symbols
        ).iloc[0]
        return ((sales - cost) / sales.where(sales != 0)).replace(
            [np.inf, -np.inf], np.nan
        ).reindex(symbols)
