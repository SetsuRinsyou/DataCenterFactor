"""论文 SGA-to-sales 特征的日频实现。"""

import numpy as np
import pandas as pd

from factor_base import FactorBase


class SGAToSales(FactorBase):
    """销售及管理费用 TTM 除以营业收入 TTM。"""

    def __init__(self):
        super().__init__(
            window="1D",
            data_fields=["sell_exp_ttm", "admin_exp_ttm", "revenue_ttm"],
        )

    def calculate_daily_factor(self, trade_date, symbols, market_data, history_start):
        sga = pd.concat(
            [
                market_data[field].reindex(
                    index=[trade_date], columns=symbols
                ).iloc[0]
                for field in ["sell_exp_ttm", "admin_exp_ttm"]
            ],
            axis=1,
        ).sum(axis=1, min_count=1)
        sales = market_data["revenue_ttm"].reindex(
            index=[trade_date], columns=symbols
        ).iloc[0]
        return (sga / sales.where(sales != 0)).replace(
            [np.inf, -np.inf], np.nan
        ).reindex(symbols)
