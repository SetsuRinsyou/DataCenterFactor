"""论文长期反转定义的21交易日月度映射。"""

import pandas as pd

from factor_base import FactorBase


class LongTermReversal(FactorBase):
    """过去第13至第36个21交易日区段收益率之和。"""

    def __init__(self):
        super().__init__(window="756D", data_fields=["close"])

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
        ].dropna(how="all")
        block_returns = close_window.iloc[::21].pct_change(fill_method=None)
        return block_returns.iloc[1:25].sum(axis=0, min_count=24).reindex(symbols)
