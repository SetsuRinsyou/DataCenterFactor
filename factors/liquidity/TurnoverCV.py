"""换手率变异系数因子。"""

import pandas as pd

from factor_base import FactorBase


class TurnoverCV(FactorBase):
    """最近20日总股本换手率的标准差除以均值。"""

    def __init__(self):
        super().__init__(window="20D", data_fields=["vol", "total_share"])

    def calculate_daily_factor(
        self,
        trade_date: str,
        symbols: list[str],
        market_data: dict[str, pd.DataFrame],
        history_start: str,
    ) -> pd.Series:
        volume_window = market_data["vol"].loc[
            history_start:trade_date
        ].reindex(columns=symbols).tail(20)
        total_share_window = market_data["total_share"].reindex(
            index=volume_window.index,
            columns=symbols,
        )
        daily_turnover = (
            volume_window.where(volume_window >= 0) * 100
            / (total_share_window.where(total_share_window > 0) * 10_000)
        )

        valid_count = daily_turnover.count()
        turnover_mean = daily_turnover.mean()
        turnover_std = daily_turnover.std()
        result = turnover_std / turnover_mean.where(turnover_mean > 0)
        return result.where(valid_count >= 15).reindex(symbols)
