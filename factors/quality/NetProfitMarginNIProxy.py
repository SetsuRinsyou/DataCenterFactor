"""归母净利润 TTM 除以营业收入 TTM。"""

import numpy as np

from factor_base import FactorBase


class NetProfitMarginNIProxy(FactorBase):
    """归母净利润 TTM 除以营业收入 TTM。"""

    def __init__(self):
        super().__init__(
            window="1D", data_fields=["n_income_attr_p_ttm", "revenue_ttm"]
        )

    def calculate_daily_factor(self, trade_date, symbols, market_data, history_start):
        numerator = market_data["n_income_attr_p_ttm"].reindex(
            index=[trade_date], columns=symbols
        ).iloc[0]
        denominator = market_data["revenue_ttm"].reindex(
            index=[trade_date], columns=symbols
        ).iloc[0]
        return (numerator / denominator.where(denominator != 0)).replace(
            [np.inf, -np.inf], np.nan
        ).reindex(symbols)
