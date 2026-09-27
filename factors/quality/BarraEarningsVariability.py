"""CNE6 Earnings Variability 二级因子的公开资料等权复现。"""

import pandas as pd

from factor_base import FactorBase
from factors.barra.BarraCNE6Descriptors import (
    select_annual_report_window,
    calculate_etopf_std,
    calculate_vern,
    calculate_vflo,
    calculate_vsal,
    combine_descriptors_equal_weight,
)


class BarraEarningsVariability(FactorBase):
    """VSAL、VERN、VFLO 和 ETOPF_STD 标准化后等权合成。"""

    def __init__(self):
        super().__init__(
            window="7Y",
            data_fields=[
                "total_mv",
                "total_share",
                "forecast_eps_12m_std",
            ],
            financial_report_fields=[
                "revenue_ttm", "n_income_attr_p_ttm",
                "n_incr_cash_cash_equ_ttm",
            ],
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
        annual, _ = select_annual_report_window(
            financial_report_data, symbols, self.financial_report_fields
        )
        current_market_cap = history["total_mv"].loc[trade_date] * 10_000
        current_price = history["total_mv"].loc[trade_date].divide(
            history["total_share"].loc[trade_date]
        )
        descriptors = pd.DataFrame(
            {
                "VSAL": calculate_vsal(annual["revenue_ttm"]),
                "VERN": calculate_vern(annual["n_income_attr_p_ttm"]),
                "VFLO": calculate_vflo(annual["n_incr_cash_cash_equ_ttm"]),
                "ETOPF_STD": calculate_etopf_std(
                    history["forecast_eps_12m_std"].loc[trade_date],
                    current_price,
                ),
            }
        )
        return combine_descriptors_equal_weight(
            descriptors, current_market_cap
        ).reindex(symbols)
