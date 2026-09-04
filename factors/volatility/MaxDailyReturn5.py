"""过去21日前5大单日收益均值因子。"""

import numpy as np
import pandas as pd

from factor_base import FactorBase


class MaxDailyReturn5(FactorBase):
    """过去21个交易日前5大单日收益率的均值。"""

    def __init__(self):
        super().__init__(window="21D", data_fields=["close"])

    def calculate_daily_factor(
        self,
        trade_date: str,
        symbols: list[str],
        market_data: dict[str, pd.DataFrame],
        history_start: str,
    ) -> pd.Series:
        close_window = market_data["close"].loc[
            history_start:trade_date
        ].reindex(columns=symbols).tail(22)
        daily_returns = close_window.pct_change(fill_method=None).replace(
            [np.inf, -np.inf], np.nan
        ).tail(21)

        result = pd.Series(index=symbols, dtype=float)
        for symbol in symbols:
            valid_returns = daily_returns[symbol].dropna().to_numpy()
            if len(valid_returns) >= 5:
                result[symbol] = np.partition(valid_returns, -5)[-5:].mean()
        return result.reindex(symbols)
