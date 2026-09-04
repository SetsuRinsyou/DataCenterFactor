"""中国市场映射的 Net operating assets 日频实现。"""

import numpy as np
import pandas as pd

from factor_base import FactorBase


class NetOperatingAssetsCN(FactorBase):
    """净经营资产金额除以总资产 MRQ。"""

    def __init__(self):
        super().__init__(
            window="1D",
            data_fields=[
                "comp_type",
                "total_assets_mrq",
                "money_cap_mrq",
                "trad_asset_mrq",
                "total_liab_mrq",
                "st_borr_mrq",
                "st_bonds_payable_mrq",
                "non_cur_liab_due_1y_mrq",
                "lt_borr_mrq",
                "bond_payable_mrq",
                "lease_liab_mrq",
            ],
        )

    def calculate_daily_factor(self, trade_date, symbols, market_data, history_start):
        assets = market_data["total_assets_mrq"].reindex(
            index=[trade_date], columns=symbols
        ).iloc[0]
        cash = market_data["money_cap_mrq"].reindex(
            index=[trade_date], columns=symbols
        ).iloc[0]
        trading_assets = market_data["trad_asset_mrq"].reindex(
            index=[trade_date], columns=symbols
        ).iloc[0]
        cash_sti = cash + trading_assets.fillna(0)
        liabilities = market_data["total_liab_mrq"].reindex(
            index=[trade_date], columns=symbols
        ).iloc[0]
        current_debt_fields = [
            "st_borr_mrq",
            "st_bonds_payable_mrq",
            "non_cur_liab_due_1y_mrq",
        ]
        long_term_debt_fields = [
            "lt_borr_mrq",
            "bond_payable_mrq",
            "lease_liab_mrq",
        ]
        current_debt = pd.concat(
            [
                market_data[field].reindex(
                    index=[trade_date], columns=symbols
                ).iloc[0]
                for field in current_debt_fields
            ],
            axis=1,
        ).fillna(0).sum(axis=1)
        long_term_debt = pd.concat(
            [
                market_data[field].reindex(
                    index=[trade_date], columns=symbols
                ).iloc[0]
                for field in long_term_debt_fields
            ],
            axis=1,
        ).fillna(0).sum(axis=1)
        debt = current_debt + long_term_debt
        noa_amount = assets - cash_sti - liabilities + debt
        company_type = market_data["comp_type"].reindex(
            index=[trade_date], columns=symbols
        ).iloc[0]
        result = (noa_amount / assets.where(assets != 0)).replace(
            [np.inf, -np.inf], np.nan
        )
        return result.where(company_type.astype("string").eq("1")).reindex(symbols)
