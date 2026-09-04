"""Earnings-to-price 的归母净利润代理日频实现。"""

import numpy as np

from factor_base import FactorBase


class EarningsToPriceNIProxy(FactorBase):
    """归母净利润 TTM 除以信号日总市值。"""

    def __init__(self):
        super().__init__(
            window="1D", data_fields=["n_income_attr_p_ttm", "total_mv"]
        )

    def calculate_daily_factor(self, trade_date, symbols, market_data, history_start):
        income = market_data["n_income_attr_p_ttm"].reindex(
            index=[trade_date], columns=symbols
        ).iloc[0]
        market_equity = (
            market_data["total_mv"].reindex(
                index=[trade_date], columns=symbols
            ).iloc[0]
            * 10_000
        )
        return (income / market_equity.where(market_equity != 0)).replace(
            [np.inf, -np.inf], np.nan
        ).reindex(symbols)
