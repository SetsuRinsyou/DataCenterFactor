"""论文定义的日频52周高点价格比因子。"""

import pandas as pd

from factor_base import FactorBase


class PriceRelative52WeekHigh(FactorBase):
    """信号日复权价格与最近52个有效周最高周收盘价之比。"""

    def __init__(self):
        super().__init__(window="260D", data_fields=["close"])

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
        dated_close = close_window.copy()
        dated_close.index = pd.to_datetime(dated_close.index, format="%Y%m%d")
        weekly_close = dated_close.resample("W-FRI").last()

        weekly_high = {}
        for symbol, values in weekly_close.items():
            valid_values = values.dropna().tail(52)
            weekly_high[symbol] = (
                valid_values.max() if len(valid_values) == 52 else float("nan")
            )
        weekly_high = pd.Series(weekly_high)

        current_close = close_window.reindex(
            index=[trade_date],
            columns=symbols,
        ).iloc[0]
        return (current_close / weekly_high.where(weekly_high > 0)).reindex(symbols)
