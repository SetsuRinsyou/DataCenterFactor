"""论文 Gross profitability 特征的日频实现。"""

import numpy as np

from factor_base import FactorBase


class GrossProfitability(FactorBase):
    """毛利润 TTM 除以归母股东权益 MRQ。"""

    def __init__(self):
        super().__init__(
            window="1D",
            data_fields=[
                "revenue_ttm",
                "oper_cost_ttm",
                "total_hldr_eqy_exc_min_int_mrq",
            ],
        )

    def calculate_daily_factor(self, trade_date, symbols, market_data, history_start):
        sales = market_data["revenue_ttm"].reindex(
            index=[trade_date], columns=symbols
        ).iloc[0]
        cost = market_data["oper_cost_ttm"].reindex(
            index=[trade_date], columns=symbols
        ).iloc[0]
        book_equity = market_data["total_hldr_eqy_exc_min_int_mrq"].reindex(
            index=[trade_date], columns=symbols
        ).iloc[0]
        return ((sales - cost) / book_equity.where(book_equity != 0)).replace(
            [np.inf, -np.inf], np.nan
        ).reindex(symbols)
