"""CNE6 Short-Term Reversal 二级因子。"""

import pandas as pd

from factor_base import FactorBase
from factors.barra.BarraCNE6Descriptors import (
    calculate_strev,
    standardize_descriptor,
)


class BarraShortTermReversal(FactorBase):
    """过去 21 个交易日、半衰期 5 日的对数收益率加权和。"""

    def __init__(self):
        super().__init__(window="21D", data_fields=["close", "total_mv"])

    def calculate_daily_factor(
        self,
        trade_date: str,
        symbols: list[str],
        market_data: dict[str, pd.DataFrame],
        history_start: str,
    ) -> pd.Series:
        close = market_data["close"].loc[history_start:trade_date].reindex(
            columns=symbols
        )
        returns = close.pct_change(fill_method=None).tail(21)
        descriptor = calculate_strev(returns)
        market_cap = market_data["total_mv"].reindex(
            index=[trade_date], columns=symbols
        ).iloc[0] * 10_000
        return standardize_descriptor(descriptor, market_cap).reindex(symbols)
