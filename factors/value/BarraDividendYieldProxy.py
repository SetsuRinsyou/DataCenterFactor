"""CNE6 Dividend Yield 的历史股息率代理。"""

import pandas as pd

from factor_base import FactorBase
from factors.barra.BarraCNE6Descriptors import standardize_descriptor


class BarraDividendYieldProxy(FactorBase):
    """以 Tushare 滚动股息率代理 DTOP；不包含预测股息率 DTOPF。"""

    def __init__(self):
        super().__init__(window="1D", data_fields=["dv_ttm", "total_mv"])

    def calculate_daily_factor(
        self,
        trade_date: str,
        symbols: list[str],
        market_data: dict[str, pd.DataFrame],
        history_start: str,
    ) -> pd.Series:
        dividend_yield = market_data["dv_ttm"].reindex(
            index=[trade_date], columns=symbols
        ).iloc[0] / 100
        market_cap = market_data["total_mv"].reindex(
            index=[trade_date], columns=symbols
        ).iloc[0] * 10_000
        return standardize_descriptor(
            dividend_yield, market_cap
        ).reindex(symbols)
