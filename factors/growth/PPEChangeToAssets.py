"""论文 PPE-chg-to-assets 特征的日频实现。"""

import numpy as np
import pandas as pd

from factor_base import FactorBase


class PPEChangeToAssets(FactorBase):
    """固定资产合计 MRQ 同比变化除以去年同期总资产 MRQ。"""

    def __init__(self):
        super().__init__(
            window="504D",
            data_fields=[
                "report_end_date",
                "fix_assets_total_mrq",
                "total_assets_mrq",
            ],
        )

    @staticmethod
    def _prior_year_value(
        field, trade_date, symbols, market_data, history_start
    ):
        current_report = market_data["report_end_date"].reindex(
            index=[trade_date], columns=symbols
        ).iloc[0]
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
        fixed_assets = market_data["fix_assets_total_mrq"].reindex(
            index=[trade_date], columns=symbols
        ).iloc[0]
        prior_fixed_assets = self._prior_year_value(
            "fix_assets_total_mrq",
            trade_date,
            symbols,
            market_data,
            history_start,
        )
        prior_assets = self._prior_year_value(
            "total_assets_mrq",
            trade_date,
            symbols,
            market_data,
            history_start,
        )
        return (
            (fixed_assets - prior_fixed_assets) / prior_assets.where(prior_assets != 0)
        ).replace([np.inf, -np.inf], np.nan).reindex(symbols)
