"""中国财报固定资产 MRQ 除以总资产 MRQ。"""

import numpy as np

from factor_base import FactorBase


class FixedAssetsToAssetsCN(FactorBase):
    """中国财报固定资产 MRQ 除以总资产 MRQ。"""

    def __init__(self):
        super().__init__(
            window="1D", data_fields=["fix_assets_mrq", "total_assets_mrq"]
        )

    def calculate_daily_factor(self, trade_date, symbols, market_data, history_start):
        numerator = market_data["fix_assets_mrq"].reindex(
            index=[trade_date], columns=symbols
        ).iloc[0]
        denominator = market_data["total_assets_mrq"].reindex(
            index=[trade_date], columns=symbols
        ).iloc[0]
        return (numerator / denominator.where(denominator != 0)).replace(
            [np.inf, -np.inf], np.nan
        ).reindex(symbols)
