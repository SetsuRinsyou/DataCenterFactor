"""毛利润 TTM 除以营业收入 TTM。"""

import numpy as np

from factor_base import FactorBase


class GrossMargin(FactorBase):
    """营业收入扣除营业成本后的毛利率。"""

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
