"""CNE6 Industry Momentum 二级因子。"""

import pandas as pd

from factor_base import FactorBase
from factors.barra.BarraCNE6Descriptors import (
    calculate_indmom,
    standardize_descriptor,
)


class BarraIndustryMomentum(FactorBase):
    """以申万一级行业计算 INDMOM。"""

    def __init__(self):
        super().__init__(
            window="132D",
            data_fields=[
                "close",
                "circ_mv",
                "total_mv",
                "sw_l1_industry",
            ],
        )

    def calculate_daily_factor(
        self,
        trade_date: str,
        symbols: list[str],
        market_data: dict[str, pd.DataFrame],
        history_start: str,
    ) -> pd.Series:
        close = market_data["close"].loc[
            history_start:trade_date
        ].reindex(columns=symbols)
        industry = market_data["sw_l1_industry"].loc[
            history_start:trade_date
        ].reindex(columns=symbols)
        free_float_market_cap = market_data["circ_mv"].loc[
            history_start:trade_date
        ].reindex(columns=symbols)
        descriptor = calculate_indmom(
            close.pct_change(fill_method=None),
            industry,
            free_float_market_cap,
        )
        market_cap = market_data["total_mv"].loc[trade_date] * 10_000
        return standardize_descriptor(
            descriptor, market_cap
        ).reindex(symbols)
