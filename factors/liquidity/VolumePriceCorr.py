"""量价相关性因子。"""

import pandas as pd

from factor_base import FactorBase


class VolumePriceCorr(FactorBase):
    """最近10日复权收盘价与总股本换手率的相关系数。"""

    def __init__(self):
        super().__init__(
            window="10D",
            data_fields=["close", "vol", "total_share"],
        )

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
        volume_window = market_data["vol"].reindex(
            index=close_window.index,
            columns=symbols,
        )
        total_share_window = market_data["total_share"].reindex(
            index=close_window.index,
            columns=symbols,
        )
        daily_turnover = (
            volume_window.where(volume_window >= 0) * 100
            / (total_share_window.where(total_share_window > 0) * 10_000)
        )

        paired = close_window.notna() & daily_turnover.notna()
        correlation = close_window.where(paired).corrwith(
            daily_turnover.where(paired)
        )
        return correlation.where(paired.sum() >= 10).reindex(symbols)
