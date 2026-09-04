"""十日开盘冲高价格形态因子。"""

import numpy as np
import pandas as pd

from factor_base import FactorBase


class OpenHighSurge10(FactorBase):
    """最近10日从开盘价到日内最高价的平均对数上冲幅度。"""

    def __init__(self):
        super().__init__(window="10D", data_fields=["open", "high"])

    def calculate_daily_factor(
        self,
        trade_date: str,
        symbols: list[str],
        market_data: dict[str, pd.DataFrame],
        history_start: str,
    ) -> pd.Series:
        open_window = market_data["open"].loc[
            history_start:trade_date
        ].reindex(columns=symbols).tail(10)
        high_window = market_data["high"].reindex(
            index=open_window.index,
            columns=symbols,
        )
        valid = (open_window > 0) & (high_window > 0) & (high_window >= open_window)
        daily_surge = np.log(high_window.where(valid) / open_window.where(valid))
        return daily_surge.mean().where(daily_surge.count() >= 5).reindex(symbols)
