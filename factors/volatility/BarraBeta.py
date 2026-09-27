"""CNE6 Beta 二级因子。"""

import pandas as pd

from factor_base import FactorBase
from factors.barra.BarraCNE6Descriptors import (
    calculate_hbeta,
    prepare_market_model_returns,
    standardize_descriptor,
)


class BarraBeta(FactorBase):
    """相对外部配置市场基准的历史 Beta。"""

    def __init__(self, index_code: str):
        if not isinstance(index_code, str) or not index_code.strip():
            raise ValueError("index_code must be a non-empty string")
        super().__init__(
            window="252D",
            data_fields=["close", "gc001_weight", "total_mv"],
            index_code=index_code.strip(),
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
        index_close = market_data["index_close"].loc[
            history_start:trade_date, self.index_code
        ].reindex(close.index)
        risk_free_rate = (
            market_data["gc001_weight"]
            .loc[history_start:trade_date]
            .max(axis=1)
            .reindex(close.index)
            / 100
            / 365
        )
        stock_excess, benchmark_excess, _ = prepare_market_model_returns(
            close, index_close, risk_free_rate
        )
        descriptor = calculate_hbeta(stock_excess, benchmark_excess)
        market_cap = market_data["total_mv"].reindex(
            index=[trade_date], columns=symbols
        ).iloc[0] * 10_000
        return standardize_descriptor(descriptor, market_cap).reindex(symbols)
