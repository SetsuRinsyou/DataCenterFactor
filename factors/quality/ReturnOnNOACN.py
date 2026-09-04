"""中国市场映射的 Return on NOA 日频实现。"""

import numpy as np
import pandas as pd

from factor_base import FactorBase


class ReturnOnNOACN(FactorBase):
    """中国口径经营利润 TTM 除以去年同期净经营资产金额。"""

    DEBT_FIELDS = [
        "st_borr_mrq",
        "st_bonds_payable_mrq",
        "non_cur_liab_due_1y_mrq",
        "lt_borr_mrq",
        "bond_payable_mrq",
        "lease_liab_mrq",
    ]

    def __init__(self):
        super().__init__(
            window="504D",
            data_fields=[
                "report_end_date",
                "comp_type",
                "revenue_ttm",
                "oper_cost_ttm",
                "biz_tax_surchg_ttm",
                "sell_exp_ttm",
                "admin_exp_ttm",
                "rd_exp_ttm",
                "total_assets_mrq",
                "money_cap_mrq",
                "trad_asset_mrq",
                "total_liab_mrq",
                *self.DEBT_FIELDS,
            ],
        )

    @staticmethod
    def _current_value(market_data, field, trade_date, symbols):
        return market_data[field].reindex(
            index=[trade_date], columns=symbols
        ).iloc[0]

    @classmethod
    def _prior_year_value(
        cls, market_data, field, trade_date, symbols, history_start
    ):
        current_report = cls._current_value(
            market_data, "report_end_date", trade_date, symbols
        )
        target_report = (
            pd.to_datetime(current_report, format="%Y%m%d", errors="coerce")
            - pd.DateOffset(years=1)
        ).dt.strftime("%Y%m%d")
        report_history = market_data["report_end_date"].loc[
            history_start:trade_date
        ].reindex(columns=symbols)
        value_history = market_data[field].loc[
            history_start:trade_date
        ].reindex(columns=symbols)
        return value_history.where(
            report_history.eq(target_report, axis="columns")
        ).ffill().iloc[-1]

    def calculate_daily_factor(self, trade_date, symbols, market_data, history_start):
        sales = self._current_value(
            market_data, "revenue_ttm", trade_date, symbols
        )
        operating_cost = self._current_value(
            market_data, "oper_cost_ttm", trade_date, symbols
        )
        optional_costs = pd.concat(
            [
                self._current_value(market_data, field, trade_date, symbols)
                for field in [
                    "biz_tax_surchg_ttm",
                    "sell_exp_ttm",
                    "admin_exp_ttm",
                    "rd_exp_ttm",
                ]
            ],
            axis=1,
        ).fillna(0).sum(axis=1)
        operating_profit = sales - operating_cost - optional_costs
        prior_assets = self._prior_year_value(
            market_data, "total_assets_mrq", trade_date, symbols, history_start
        )
        prior_cash = self._prior_year_value(
            market_data, "money_cap_mrq", trade_date, symbols, history_start
        )
        prior_trading_assets = self._prior_year_value(
            market_data, "trad_asset_mrq", trade_date, symbols, history_start
        ).fillna(0)
        prior_liabilities = self._prior_year_value(
            market_data, "total_liab_mrq", trade_date, symbols, history_start
        )
        prior_debt = pd.concat(
            [
                self._prior_year_value(
                    market_data, field, trade_date, symbols, history_start
                )
                for field in self.DEBT_FIELDS
            ],
            axis=1,
        ).fillna(0).sum(axis=1)
        prior_noa = (
            prior_assets
            - prior_cash
            - prior_trading_assets
            - prior_liabilities
            + prior_debt
        )
        result = (operating_profit / prior_noa.where(prior_noa != 0)).replace(
            [np.inf, -np.inf], np.nan
        )
        company_type = self._current_value(
            market_data, "comp_type", trade_date, symbols
        )
        return result.where(company_type.astype("string").eq("1")).reindex(symbols)
