"""论文 Sales-to-assets 特征的日频实现。"""

import numpy as np

from factor_base import FactorBase


class SalesToAssets(FactorBase):
    """营业收入 TTM 除以总资产 MRQ。"""

    def __init__(self):
        super().__init__(
            window="1D", data_fields=["revenue_ttm", "total_assets_mrq"]
        )

    def calculate_daily_factor(self, trade_date, symbols, market_data, history_start):
        sales = market_data["revenue_ttm"].reindex(
            index=[trade_date], columns=symbols
        ).iloc[0]
        assets = market_data["total_assets_mrq"].reindex(
            index=[trade_date], columns=symbols
        ).iloc[0]
        return (sales / assets.where(assets != 0)).replace(
            [np.inf, -np.inf], np.nan
        ).reindex(symbols)
