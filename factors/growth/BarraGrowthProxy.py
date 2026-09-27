"""CNE6 Growth 的历史增长描述子代理。"""

import pandas as pd

from factor_base import FactorBase
from factors.barra.BarraCNE6Descriptors import (
    annual_report_values,
    calculate_egro,
    calculate_sgro,
    combine_descriptors_equal_weight,
    completed_year_end_values,
)


class BarraGrowthProxy(FactorBase):
    """以 EGRO 和 SGRO 代理；不包含分析师长期盈利增长预测。"""

    def __init__(self):
        super().__init__(
            window="7Y",
            data_fields=[
                "total_mv",
                "total_share",
                "report_end_date",
                "revenue_ttm",
                "n_income_attr_p_ttm",
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
        annual_earnings = annual_report_values(
            history["n_income_attr_p_ttm"], history["report_end_date"]
        )
        annual_sales = annual_report_values(
            history["revenue_ttm"], history["report_end_date"]
        )
        annual_shares = completed_year_end_values(
            history["total_share"], trade_date, count=7
        )
        annual_earnings.index = annual_earnings.index.astype(str).str[:4]
        annual_sales.index = annual_sales.index.astype(str).str[:4]
        annual_eps = annual_earnings.divide(annual_shares).tail(5)
        annual_sales_per_share = annual_sales.divide(annual_shares).tail(5)
        descriptors = pd.DataFrame(
            {
                "EGRO": calculate_egro(annual_eps),
                "SGRO": calculate_sgro(annual_sales_per_share),
            }
        )
        market_cap = history["total_mv"].loc[trade_date] * 10_000
        return combine_descriptors_equal_weight(
            descriptors, market_cap
        ).reindex(symbols)
