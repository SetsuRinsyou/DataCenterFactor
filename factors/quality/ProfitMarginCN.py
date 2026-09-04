"""中国市场映射的 Profit margin 日频实现。"""

import numpy as np
import pandas as pd

from factor_base import FactorBase


class ProfitMarginCN(FactorBase):
    """中国口径折旧后经营利润 TTM 除以营业收入 TTM。"""

    def __init__(self):
        super().__init__(
            window="1D",
            data_fields=[
                "comp_type",
                "revenue_ttm",
                "oper_cost_ttm",
                "biz_tax_surchg_ttm",
                "sell_exp_ttm",
                "admin_exp_ttm",
                "rd_exp_ttm",
            ],
        )

    def calculate_daily_factor(self, trade_date, symbols, market_data, history_start):
        sales = market_data["revenue_ttm"].reindex(
            index=[trade_date], columns=symbols
        ).iloc[0]
        operating_cost = market_data["oper_cost_ttm"].reindex(
            index=[trade_date], columns=symbols
        ).iloc[0]
        optional_costs = pd.concat(
            [
                market_data[field].reindex(
                    index=[trade_date], columns=symbols
                ).iloc[0]
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
        result = (operating_profit / sales.where(sales != 0)).replace(
            [np.inf, -np.inf], np.nan
        )
        company_type = market_data["comp_type"].reindex(
            index=[trade_date], columns=symbols
        ).iloc[0]
        return result.where(company_type.astype("string").eq("1")).reindex(symbols)
