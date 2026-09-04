"""三个月价格反转因子。"""

import pandas as pd

from factor_base import FactorBase


class Reversal3M(FactorBase):
    """截至信号日的过去60交易日收益率取负。"""

    def __init__(self):
        super().__init__(window="60D", data_fields=["close"])

    def calculate_daily_factor(
        self,
        trade_date: str,
        symbols: list[str],
        market_data: dict[str, pd.DataFrame],
        history_start: str,
    ) -> pd.Series:
        close_window = market_data["close"].loc[
            history_start:trade_date,
            symbols,
        ].tail(61)
        start_close = close_window.iloc[0]
        current_close = close_window.iloc[-1]
        return (-(current_close / start_close.where(start_close > 0) - 1)).reindex(
            symbols
        )
