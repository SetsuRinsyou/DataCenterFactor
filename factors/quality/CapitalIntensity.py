"""论文 Capital intensity 特征的日频实现。"""

import numpy as np

from factor_base import FactorBase


class CapitalIntensity(FactorBase):
    """折旧摊销 TTM 除以总资产 MRQ。"""

    def __init__(self):
        super().__init__(
            window="1D",
            data_fields=[
                "depreciation_ttm_pit",
                "total_assets_mrq",
            ],
        )

    def calculate_daily_factor(self, trade_date, symbols, market_data, history_start):
        depreciation = market_data["depreciation_ttm_pit"].reindex(
            index=[trade_date], columns=symbols
        ).iloc[0]
        assets = market_data["total_assets_mrq"].reindex(
            index=[trade_date], columns=symbols
        ).iloc[0]
        return (depreciation / assets.where(assets != 0)).replace(
            [np.inf, -np.inf], np.nan
        ).reindex(symbols)
