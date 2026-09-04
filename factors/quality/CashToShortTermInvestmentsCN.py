"""中国市场映射的 Cash-to-short-term-investments 日频实现。"""

import numpy as np

from factor_base import FactorBase


class CashToShortTermInvestmentsCN(FactorBase):
    """现金及交易性金融资产 MRQ 除以总资产 MRQ。"""

    def __init__(self):
        super().__init__(
            window="1D",
            data_fields=[
                "comp_type",
                "money_cap_mrq",
                "trad_asset_mrq",
                "total_assets_mrq",
            ],
        )

    def calculate_daily_factor(self, trade_date, symbols, market_data, history_start):
        cash = market_data["money_cap_mrq"].reindex(
            index=[trade_date], columns=symbols
        ).iloc[0]
        trading_assets = market_data["trad_asset_mrq"].reindex(
            index=[trade_date], columns=symbols
        ).iloc[0]
        assets = market_data["total_assets_mrq"].reindex(
            index=[trade_date], columns=symbols
        ).iloc[0]
        company_type = market_data["comp_type"].reindex(
            index=[trade_date], columns=symbols
        ).iloc[0]
        result = ((cash + trading_assets.fillna(0)) / assets.where(assets != 0)).replace(
            [np.inf, -np.inf], np.nan
        )
        return result.where(company_type.astype("string").eq("1")).reindex(symbols)
