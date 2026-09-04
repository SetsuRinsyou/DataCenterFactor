"""论文定义的日频总市值因子。"""

import pandas as pd

from factor_base import FactorBase


class MarketCap(FactorBase):
    """信号日总市值，以元为单位。"""

    def __init__(self):
        super().__init__(window="1D", data_fields=["total_mv"])

    def calculate_daily_factor(
        self,
        trade_date: str,
        symbols: list[str],
        market_data: dict[str, pd.DataFrame],
        history_start: str,
    ) -> pd.Series:
        total_mv = market_data["total_mv"].reindex(
            index=[trade_date],
            columns=symbols,
        ).iloc[0]
        return (total_mv * 10_000).where(total_mv > 0).reindex(symbols)
