"""十日盘低回升价格形态因子。"""

import numpy as np
import pandas as pd

from factor_base import FactorBase


class DipRebound10(FactorBase):
    """最近10日从日内最低价到收盘价的平均对数回升幅度。"""

    def __init__(self):
        super().__init__(window="10D", data_fields=["close", "low"])

    def calculate_daily_factor(
        self,
        trade_date: str,
        symbols: list[str],
        market_data: dict[str, pd.DataFrame],
        history_start: str,
    ) -> pd.Series:
        close_window = market_data["close"].loc[
            history_start:trade_date
        ].reindex(columns=symbols).tail(10)
        low_window = market_data["low"].reindex(
            index=close_window.index,
            columns=symbols,
        )
        valid = (close_window > 0) & (low_window > 0) & (close_window >= low_window)
        daily_rebound = np.log(close_window.where(valid) / low_window.where(valid))
        return daily_rebound.mean().where(daily_rebound.count() >= 5).reindex(symbols)
