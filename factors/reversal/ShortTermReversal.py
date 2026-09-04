"""论文短期反转定义的21交易日映射。"""

import pandas as pd

from factor_base import FactorBase


class ShortTermReversal(FactorBase):
    """截至信号日的最近21交易日收益率。"""

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
        ].reindex(columns=symbols).dropna(how="all")
        return (close_window.iloc[-1] / close_window.iloc[0] - 1).reindex(symbols)
