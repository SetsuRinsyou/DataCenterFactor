"""CNE6 Earnings Variability 的历史财务描述子代理。"""

import pandas as pd

from factor_base import FactorBase
from factors.barra.BarraCNE6Descriptors import (
    annual_report_values,
    calculate_vern,
    calculate_vsal,
    combine_descriptors_equal_weight,
)


class BarraEarningsVariabilityProxy(FactorBase):
    """以 VSAL 和 VERN 代理；不包含预测分歧及现金流波动描述子。"""

    def __init__(self):
        super().__init__(
            window="7Y",
            data_fields=[
                "total_mv",
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
        report_end_dates = market_data["report_end_date"].loc[
            history_start:trade_date
        ].reindex(columns=symbols)
        annual_sales = annual_report_values(
            market_data["revenue_ttm"].loc[history_start:trade_date].reindex(
                columns=symbols
            ),
            report_end_dates,
        )
        annual_earnings = annual_report_values(
            market_data["n_income_attr_p_ttm"]
            .loc[history_start:trade_date]
            .reindex(columns=symbols),
            report_end_dates,
        )
        descriptors = pd.DataFrame(
            {
                "VSAL": calculate_vsal(annual_sales),
                "VERN": calculate_vern(annual_earnings),
            }
        )
        market_cap = market_data["total_mv"].reindex(
            index=[trade_date], columns=symbols
        ).iloc[0] * 10_000
        return combine_descriptors_equal_weight(
            descriptors, market_cap
        ).reindex(symbols)
