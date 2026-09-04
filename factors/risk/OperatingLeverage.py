"""论文 Operating leverage 特征的日频实现。"""

import numpy as np
import pandas as pd

from factor_base import FactorBase


class OperatingLeverage(FactorBase):
    """营业成本及销售管理费用 TTM 除以总资产 MRQ。"""

    def __init__(self):
        super().__init__(
            window="1D",
            data_fields=[
                "oper_cost_ttm",
                "sell_exp_ttm",
                "admin_exp_ttm",
                "total_assets_mrq",
            ],
        )

    def calculate_daily_factor(self, trade_date, symbols, market_data, history_start):
        cost = market_data["oper_cost_ttm"].reindex(
            index=[trade_date], columns=symbols
        ).iloc[0]
        sga = pd.concat(
            [
                market_data[field].reindex(
                    index=[trade_date], columns=symbols
                ).iloc[0]
                for field in ["sell_exp_ttm", "admin_exp_ttm"]
            ],
            axis=1,
        ).sum(axis=1, min_count=1)
        assets = market_data["total_assets_mrq"].reindex(
            index=[trade_date], columns=symbols
        ).iloc[0]
        return ((cost + sga) / assets.where(assets != 0)).replace(
            [np.inf, -np.inf], np.nan
        ).reindex(symbols)
