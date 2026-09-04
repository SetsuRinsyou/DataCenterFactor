"""中国市场映射的 Tobin's Q 日频实现。"""

import numpy as np

from factor_base import FactorBase


class TobinsQCN(FactorBase):
    """总资产加市值、扣除现金短投和递延所得税负债后除以总资产。"""

    def __init__(self):
        super().__init__(
            window="1D",
            data_fields=[
                "comp_type",
                "total_assets_mrq",
                "total_mv",
                "money_cap_mrq",
                "trad_asset_mrq",
                "defer_tax_liab_mrq",
            ],
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
        cash = market_data["money_cap_mrq"].reindex(
            index=[trade_date], columns=symbols
        ).iloc[0]
        trading_assets = market_data["trad_asset_mrq"].reindex(
            index=[trade_date], columns=symbols
        ).iloc[0]
        cash_sti = cash + trading_assets.fillna(0)
        deferred_tax = market_data["defer_tax_liab_mrq"].reindex(
            index=[trade_date], columns=symbols
        ).iloc[0].fillna(0)
        result = (
            (assets + market_equity - cash_sti - deferred_tax)
            / assets.where(assets != 0)
        ).replace([np.inf, -np.inf], np.nan)
        company_type = market_data["comp_type"].reindex(
            index=[trade_date], columns=symbols
        )
        return result.where(
            company_type.iloc[0].astype("string").eq("1")
        ).reindex(symbols)
