"""论文 Assets-to-market 特征的日频实现。"""

import numpy as np

from factor_base import FactorBase


class AssetsToMarket(FactorBase):
    """总资产 MRQ 除以信号日总市值。"""

    def __init__(self):
        super().__init__(
            window="1D", data_fields=["total_assets_mrq", "total_mv"]
        )

    def calculate_daily_factor(self, trade_date, symbols, market_data, history_start):
        assets = market_data["total_assets_mrq"].reindex(
            index=[trade_date], columns=symbols
        ).iloc[0]
        market_equity = (
            market_data["total_mv"].reindex(
                index=[trade_date], columns=symbols
            ).iloc[0]
            * 10_000
        )
        return (assets / market_equity.where(market_equity != 0)).replace(
            [np.inf, -np.inf], np.nan
        ).reindex(symbols)
