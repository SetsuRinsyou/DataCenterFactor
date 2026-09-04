"""中国市场映射的 Leverage 日频实现。"""

import numpy as np
import pandas as pd

from factor_base import FactorBase


class LeverageCN(FactorBase):
    """带息债务 MRQ 除以带息债务与归母股东权益之和。"""

    def __init__(self):
        super().__init__(
            window="1D",
            data_fields=[
                "comp_type",
                "total_hldr_eqy_exc_min_int_mrq",
                "st_borr_mrq",
                "st_bonds_payable_mrq",
                "non_cur_liab_due_1y_mrq",
                "lt_borr_mrq",
                "bond_payable_mrq",
                "lease_liab_mrq",
            ],
        )

    def calculate_daily_factor(self, trade_date, symbols, market_data, history_start):
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
        book_equity = market_data["total_hldr_eqy_exc_min_int_mrq"].reindex(
            index=[trade_date], columns=symbols
        ).iloc[0]
        company_type = market_data["comp_type"].reindex(
            index=[trade_date], columns=symbols
        ).iloc[0]
        denominator = debt + book_equity
        result = (debt / denominator.where(denominator != 0)).replace(
            [np.inf, -np.inf], np.nan
        )
        return result.where(company_type.astype("string").eq("1")).reindex(symbols)
