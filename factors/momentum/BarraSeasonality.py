"""CNE6 Seasonality 二级因子。"""

import pandas as pd

from factor_base import FactorBase
from factors.barra.BarraCNE6Descriptors import (
    calculate_season,
    standardize_descriptor,
)


class BarraSeasonality(FactorBase):
    """过去五年同季节窗口内首末可得价的月收益率均值。"""

    def __init__(self):
        super().__init__(window="6Y", data_fields=["close", "total_mv"])

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
        seasonal_returns = []
        for year in range(1, 6):
            start = len(close) - 1 - year * 252
            window = close.iloc[start:start + 22]
            if start < 0 or len(window) < 22:
                seasonal_returns.append(pd.Series(float("nan"), index=symbols))
                continue
            first_price = window.bfill().iloc[0]
            last_price = window.ffill().iloc[-1]
            seasonal_returns.append(
                (last_price / first_price - 1).where(
                    window.notna().sum(axis=0) >= 2
                )
            )
        lagged_returns = pd.DataFrame(seasonal_returns)
        descriptor = calculate_season(lagged_returns)
        market_cap = market_data["total_mv"].reindex(
            index=[trade_date], columns=symbols
        ).iloc[0] * 10_000
        return standardize_descriptor(descriptor, market_cap).reindex(symbols)
