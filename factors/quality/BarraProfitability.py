"""CNE6 Profitability 二级因子的公开资料等权复现。"""

import pandas as pd

from factor_base import FactorBase
from factors.barra.BarraCNE6Descriptors import (
    calculate_ato,
    calculate_gp,
    calculate_gpm,
    calculate_roa,
    combine_descriptors_equal_weight,
    latest_annual_value,
)


class BarraProfitability(FactorBase):
    """ATO、GP、GPM 和 ROA 标准化后等权合成。"""

    def __init__(self):
        super().__init__(
            window="7Y",
            data_fields=[
                "total_mv",
                "report_end_date",
                "revenue_ttm",
                "oper_cost_ttm",
                "n_income_attr_p_ttm",
                "total_assets_mrq",
            ],
        )

    def calculate_daily_factor(
        self,
        trade_date: str,
        symbols: list[str],
        market_data: dict[str, pd.DataFrame],
        history_start: str,
    ) -> pd.Series:
        history = {
            field: market_data[field].loc[
                history_start:trade_date
            ].reindex(columns=symbols)
            for field in self.data_fields
        }
        report_end_dates = history["report_end_date"]
        annual_sales = latest_annual_value(
            history["revenue_ttm"], report_end_dates
        )
        annual_cost = latest_annual_value(
            history["oper_cost_ttm"], report_end_dates
        )
        annual_assets = latest_annual_value(
            history["total_assets_mrq"], report_end_dates
        )
        current_sales = history["revenue_ttm"].loc[trade_date]
        current_earnings = history["n_income_attr_p_ttm"].loc[trade_date]
        current_assets = history["total_assets_mrq"].loc[trade_date]
        market_cap = history["total_mv"].loc[trade_date] * 10_000

        descriptors = pd.DataFrame(
            {
                "ATO": calculate_ato(current_sales, current_assets),
                "GP": calculate_gp(annual_sales, annual_cost, annual_assets),
                "GPM": calculate_gpm(annual_sales, annual_cost),
                "ROA": calculate_roa(current_earnings, current_assets),
            }
        )
        return combine_descriptors_equal_weight(
            descriptors, market_cap
        ).reindex(symbols)
