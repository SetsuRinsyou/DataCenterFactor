"""基于近期极值点斜率的反转因子。"""

import numpy as np
import pandas as pd

from factor_base import FactorBase


class SLPReversal(FactorBase):
    """最近20日较近极值点到当前价格的单位时间斜率取负。"""

    def __init__(self):
        super().__init__(window="20D", data_fields=["close"])

    def calculate_daily_factor(
        self,
        trade_date: str,
        symbols: list[str],
        market_data: dict[str, pd.DataFrame],
        history_start: str,
    ) -> pd.Series:
        close_window = market_data["close"].loc[
            history_start:trade_date
        ].reindex(columns=symbols).tail(20)
        result = pd.Series(index=symbols, dtype=float)

        for symbol in symbols:
            prices = close_window[symbol].to_numpy(dtype=float)
            if np.isfinite(prices).sum() < 5 or not np.isfinite(prices[-1]):
                continue

            max_position = np.nanargmax(prices)
            min_position = np.nanargmin(prices)
            extreme_position = (
                max_position if max_position > min_position else min_position
            )
            elapsed_days = len(prices) - 1 - extreme_position
            if elapsed_days == 0:
                result[symbol] = 0.0
                continue

            extreme_price = prices[extreme_position]
            result[symbol] = -(
                (prices[-1] - extreme_price) / (extreme_price * elapsed_days)
            )
        return result.reindex(symbols)
