"""中国市场映射的 Operating accruals 日频实现。"""

import numpy as np
import pandas as pd

from factor_base import FactorBase


class OperatingAccrualsCN(FactorBase):
    """净营运资本同比变化扣除折旧摊销后除以总资产 MRQ。"""

    CURRENT_DEBT_FIELDS = [
        "st_borr_mrq",
        "st_bonds_payable_mrq",
        "non_cur_liab_due_1y_mrq",
    ]

    def __init__(self):
        super().__init__(
            window="504D",
            data_fields=[
                "report_end_date",
                "comp_type",
                "total_cur_assets_mrq",
                "money_cap_mrq",
                "trad_asset_mrq",
                "total_cur_liab_mrq",
                *self.CURRENT_DEBT_FIELDS,
                "depreciation_ttm_pit",
                "total_assets_mrq",
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
        cash = self._current_value(
            market_data, "money_cap_mrq", trade_date, symbols
        )
        trading_assets = self._current_value(
            market_data, "trad_asset_mrq", trade_date, symbols
        )
        current_debt = pd.concat(
            [
                self._current_value(market_data, field, trade_date, symbols)
                for field in self.CURRENT_DEBT_FIELDS
            ],
            axis=1,
        ).fillna(0).sum(axis=1)
        current_ncwc = (
            self._current_value(
                market_data, "total_cur_assets_mrq", trade_date, symbols
            )
            - cash
            - trading_assets.fillna(0)
            - self._current_value(
                market_data, "total_cur_liab_mrq", trade_date, symbols
            )
            + current_debt
        )
        prior_current_assets = self._prior_year_value(
            market_data, "total_cur_assets_mrq", trade_date, symbols, history_start
        )
        prior_cash = self._prior_year_value(
            market_data, "money_cap_mrq", trade_date, symbols, history_start
        )
        prior_trading_assets = self._prior_year_value(
            market_data, "trad_asset_mrq", trade_date, symbols, history_start
        ).fillna(0)
        prior_current_liabilities = self._prior_year_value(
            market_data, "total_cur_liab_mrq", trade_date, symbols, history_start
        )
        prior_current_debt = pd.concat(
            [
                self._prior_year_value(
                    market_data, field, trade_date, symbols, history_start
                )
                for field in self.CURRENT_DEBT_FIELDS
            ],
            axis=1,
        ).fillna(0).sum(axis=1)
        prior_ncwc = (
            prior_current_assets
            - prior_cash
            - prior_trading_assets
            - prior_current_liabilities
            + prior_current_debt
        )
        depreciation = self._current_value(
            market_data, "depreciation_ttm_pit", trade_date, symbols
        )
        assets = self._current_value(
            market_data, "total_assets_mrq", trade_date, symbols
        )
        result = (
            (current_ncwc - prior_ncwc - depreciation) / assets.where(assets != 0)
        ).replace([np.inf, -np.inf], np.nan)
        company_type = self._current_value(
            market_data, "comp_type", trade_date, symbols
        )
        return result.where(company_type.astype("string").eq("1")).reindex(symbols)
