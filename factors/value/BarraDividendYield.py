"""CNE6 Dividend Yield 二级因子的公开资料等权复现。"""

import pandas as pd

from factor_base import FactorBase
from factors.barra.BarraCNE6Descriptors import (
    calculate_dtop,
    calculate_dtopf,
    combine_descriptors_equal_weight,
)


class BarraDividendYield(FactorBase):
    """DTOP 和 DTOPF 标准化后等权合成。"""

    def __init__(self):
        super().__init__(
            window="1D",
            data_fields=[
                "total_mv",
                "cash_div_tax_ttm",
                "previous_month_end_raw_close",
                "forecast_dividend_yield_12m",
            ],
        )

    def calculate_daily_factor(
        self,
        trade_date: str,
        symbols: list[str],
        market_data: dict[str, pd.DataFrame],
        history_start: str,
    ) -> pd.Series:
        daily = {
            field: market_data[field].reindex(
                index=[trade_date], columns=symbols
            ).iloc[0]
            for field in self.data_fields
        }
        descriptors = pd.DataFrame(
            {
                "DTOP": calculate_dtop(
                    daily["cash_div_tax_ttm"],
                    daily["previous_month_end_raw_close"],
                ),
                "DTOPF": calculate_dtopf(
                    daily["forecast_dividend_yield_12m"]
                ),
            }
        )
        market_cap = daily["total_mv"] * 10_000
        return combine_descriptors_equal_weight(
            descriptors, market_cap
        ).reindex(symbols)
