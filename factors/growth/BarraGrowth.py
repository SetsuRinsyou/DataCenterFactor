"""CNE6 Growth 二级因子的公开资料等权复现。"""

import numpy as np
import pandas as pd

from factor_base import FactorBase
from factors.barra.BarraCNE6Descriptors import (
    calculate_egrlf,
    calculate_egro,
    calculate_sgro,
    combine_descriptors_equal_weight,
    completed_year_end_values,
    select_annual_report_window,
)


class BarraGrowth(FactorBase):
    """EGRLF、EGRO 和 SGRO 标准化后等权合成。"""

    def __init__(self):
        super().__init__(
            window="7Y",
            data_fields=[
                "total_mv",
                "total_share",
                "forecast_eps_fy1_fy3_cagr",
            ],
            financial_report_fields=["revenue_ttm", "n_income_attr_p_ttm"],
        )

    def calculate_daily_factor(
        self,
        trade_date: str,
        symbols: list[str],
        market_data: dict[str, pd.DataFrame],
        history_start: str,
        *,
        financial_report_data: pd.DataFrame,
    ) -> pd.Series:
        history = {
            field: market_data[field].loc[
                history_start:trade_date
            ].reindex(columns=symbols)
            for field in self.data_fields
        }
        annual, anchors = select_annual_report_window(
            financial_report_data, symbols, self.financial_report_fields
        )
        year_end_shares = completed_year_end_values(
            history["total_share"], trade_date, count=7
        )
        annual_shares = pd.DataFrame(np.nan, index=range(5), columns=symbols)
        for symbol in symbols:
            if pd.isna(anchors[symbol]):
                continue
            for offset in range(5):
                year = str(int(anchors[symbol]) - 4 + offset)
                if year in year_end_shares.index:
                    annual_shares.at[offset, symbol] = year_end_shares.at[
                        year, symbol
                    ]
        annual_eps = annual["n_income_attr_p_ttm"].divide(annual_shares)
        annual_sales_per_share = annual["revenue_ttm"].divide(annual_shares)
        descriptors = pd.DataFrame(
            {
                "EGRLF": calculate_egrlf(
                    history["forecast_eps_fy1_fy3_cagr"].loc[trade_date]
                ),
                "EGRO": calculate_egro(annual_eps),
                "SGRO": calculate_sgro(annual_sales_per_share),
            }
        )
        market_cap = history["total_mv"].loc[trade_date] * 10_000
        return combine_descriptors_equal_weight(
            descriptors, market_cap
        ).reindex(symbols)
