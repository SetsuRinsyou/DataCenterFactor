"""CNE6 Size 二级因子。"""

import pandas as pd

from factor_base import FactorBase
from factors.barra.BarraCNE6Descriptors import (
    calculate_lnsize,
    standardize_descriptor,
)


class BarraSize(FactorBase):
    """流通市值自然对数，对应单一描述子 LNSIZE。"""

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
        ).iloc[0]
        total_market_cap = market_data["total_mv"].reindex(
            index=[trade_date], columns=symbols
        ).iloc[0] * 10_000
        descriptor = calculate_lnsize(circulating_market_cap * 10_000)
        return standardize_descriptor(descriptor, total_market_cap).reindex(symbols)
