"""论文 Book-to-market 特征的日频实现。"""

import numpy as np

from factor_base import FactorBase


class BookToMarket(FactorBase):
    """归母股东权益 MRQ 除以信号日总市值。"""

    def __init__(self):
        super().__init__(
            window="1D",
            data_fields=["total_hldr_eqy_exc_min_int_mrq", "total_mv"],
        )

    def calculate_daily_factor(self, trade_date, symbols, market_data, history_start):
        book_equity = market_data["total_hldr_eqy_exc_min_int_mrq"].reindex(
            index=[trade_date], columns=symbols
        ).iloc[0]
        market_equity = (
            market_data["total_mv"].reindex(
                index=[trade_date], columns=symbols
            ).iloc[0]
            * 10_000
        )
        return (book_equity / market_equity.where(market_equity != 0)).replace(
            [np.inf, -np.inf], np.nan
        ).reindex(symbols)
