"""论文 Capital turnover 特征的日频实现。"""

import numpy as np
import pandas as pd

from factor_base import FactorBase


class CapitalTurnover(FactorBase):
    """营业收入 TTM 除以当前报告期去年同期总资产 MRQ。"""

    def __init__(self):
        super().__init__(
            window="504D",
            data_fields=["report_end_date", "revenue_ttm", "total_assets_mrq"],
        )

    @staticmethod
    def _prior_year_assets(trade_date, symbols, market_data, history_start):
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
        asset_history = market_data["total_assets_mrq"].loc[
            history_start:trade_date
        ].reindex(columns=symbols)
        return asset_history.where(
            report_history.eq(target_report, axis="columns")
        ).ffill().iloc[-1]

    def calculate_daily_factor(self, trade_date, symbols, market_data, history_start):
        sales = market_data["revenue_ttm"].reindex(
            index=[trade_date], columns=symbols
        ).iloc[0]
        prior_assets = self._prior_year_assets(
            trade_date, symbols, market_data, history_start
        )
        return (sales / prior_assets.where(prior_assets != 0)).replace(
            [np.inf, -np.inf], np.nan
        ).reindex(symbols)
