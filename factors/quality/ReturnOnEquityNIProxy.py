"""Return on equity 的归母净利润代理日频实现。"""

import numpy as np

from factor_base import FactorBase


class ReturnOnEquityNIProxy(FactorBase):
    """归母净利润 TTM 除以归母股东权益 MRQ。"""

    def __init__(self):
        super().__init__(
            window="1D",
            data_fields=[
                "n_income_attr_p_ttm",
                "total_hldr_eqy_exc_min_int_mrq",
            ],
        )

    def calculate_daily_factor(self, trade_date, symbols, market_data, history_start):
        income = market_data["n_income_attr_p_ttm"].reindex(
            index=[trade_date], columns=symbols
        ).iloc[0]
        book_equity = market_data["total_hldr_eqy_exc_min_int_mrq"].reindex(
            index=[trade_date], columns=symbols
        ).iloc[0]
        return (income / book_equity.where(book_equity != 0)).replace(
            [np.inf, -np.inf], np.nan
        ).reindex(symbols)
