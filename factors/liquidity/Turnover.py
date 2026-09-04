"""论文换手率定义的21交易日映射。"""

import pandas as pd

from factor_base import FactorBase


class Turnover(FactorBase):
    """最近21个交易日成交股数除以信号日总股本。"""

    def __init__(self):
        super().__init__(window="21D", data_fields=["vol", "total_share"])

    def calculate_daily_factor(
        self,
        trade_date: str,
        symbols: list[str],
        market_data: dict[str, pd.DataFrame],
        history_start: str,
    ) -> pd.Series:
        volume_window = market_data["vol"].loc[
            history_start:trade_date
        ].reindex(columns=symbols).tail(21)
        traded_shares = volume_window.where(volume_window >= 0).sum(
            axis=0,
            min_count=1,
        ) * 100
        total_shares = market_data["total_share"].reindex(
            index=[trade_date],
            columns=symbols,
        ).iloc[0] * 10_000
        return (traded_shares / total_shares.where(total_shares > 0)).reindex(symbols)
