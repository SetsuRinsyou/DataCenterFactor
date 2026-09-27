"""CNE6 Mid Cap 二级因子。"""

import pandas as pd

from factor_base import FactorBase
from factors.barra.BarraCNE6Descriptors import (
    calculate_lnsize,
    calculate_nlsize,
    standardize_descriptor,
)


class BarraMidCap(FactorBase):
    """LNSIZE 立方项对 LNSIZE 做市值加权回归后的残差。"""

    def __init__(self):
        super().__init__(window="1D", data_fields=["circ_mv", "total_mv"])

    def calculate_daily_factor(
        self,
        trade_date: str,
        symbols: list[str],
        market_data: dict[str, pd.DataFrame],
        history_start: str,
    ) -> pd.Series:
        circulating_market_cap = market_data["circ_mv"].reindex(
            index=[trade_date], columns=symbols
        ).iloc[0] * 10_000
        total_market_cap = market_data["total_mv"].reindex(
            index=[trade_date], columns=symbols
        ).iloc[0] * 10_000
        size_exposure = standardize_descriptor(
            calculate_lnsize(circulating_market_cap), total_market_cap
        )
        descriptor = calculate_nlsize(
            size_exposure, circulating_market_cap
        )
        return standardize_descriptor(descriptor, total_market_cap).reindex(
            symbols
        )
