"""当日可得的研发费用 TTM 除以总市值。"""

import numpy as np

from factor_base import FactorBase


class RDToMarket(FactorBase):
    """研发费用市值比；缺失研发费用保留为空，已披露零值保留为零。"""

    def __init__(self):
        super().__init__(window="1D", data_fields=["rd_exp_ttm", "total_mv"])

    def calculate_daily_factor(self, trade_date, symbols, market_data, history_start):
        rd_expense = market_data["rd_exp_ttm"].reindex(
            index=[trade_date], columns=symbols
        ).iloc[0]
        market_equity = (
            market_data["total_mv"].reindex(
                index=[trade_date], columns=symbols
            ).iloc[0]
            * 10_000
        )
        return (rd_expense / market_equity.where(market_equity > 0)).replace(
            [np.inf, -np.inf], np.nan
        ).reindex(symbols)
