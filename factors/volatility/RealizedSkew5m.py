"""realized_skew_5m：当日5分钟对数收益的 realized skewness。"""

import numpy as np
import pandas as pd

from factor_base import FactorBase


class RealizedSkew5m(FactorBase):
    """单日日内偏度，收盘后可用，不进行跨日平滑。

    输入为收盘时刻标记的1分钟行情，仅使用(09:30,11:30]及
    (13:00,15:00]。每个5分钟桶取最后有效收盘价；允许桶内缺分钟，
    整桶缺失则不跨越该桶计算收益。午休衔接，隔夜不衔接。
    """

    def __init__(self, min_returns: int = 30):
        if (isinstance(min_returns, bool)
                or not isinstance(min_returns, (int, np.integer))
                or min_returns < 1):
            raise ValueError("min_returns must be a positive integer")
        super().__init__(window="1D", data_fields=[], minute_fields=["close"],
                         minute_window_days=1)
        self.min_returns = int(min_returns)

    def calculate_daily_factor(self, trade_date, symbols, market_data,
                               history_start, *, minute_data=None):
        if minute_data is None:
            raise ValueError("RealizedSkew5m requires minute_data")
        result = pd.Series(np.nan, index=symbols, name="realized_skew_5m")
        if minute_data.empty or not symbols:
            return result

        data = minute_data[["close"]].reset_index()
        day = pd.to_datetime(trade_date, format="%Y%m%d")
        times = data["trade_time"]
        morning = (times > day + pd.Timedelta(hours=9, minutes=30)) & (
            times <= day + pd.Timedelta(hours=11, minutes=30)
        )
        afternoon = (times > day + pd.Timedelta(hours=13)) & (
            times <= day + pd.Timedelta(hours=15)
        )
        data = data.loc[
            (morning | afternoon) & data["ts_code"].isin(symbols)
            & np.isfinite(data["close"]) & (data["close"] > 0)
        ].sort_values(["ts_code", "trade_time"], kind="stable")
        data = data.drop_duplicates(["ts_code", "trade_time"], keep="last")
        if data.empty:
            return result

        data["bar_time"] = data["trade_time"].dt.ceil("5min")
        closes = data.groupby(["bar_time", "ts_code"])["close"].last().unstack("ts_code")
        bars = pd.date_range(day + pd.Timedelta(hours=9, minutes=35),
                             day + pd.Timedelta(hours=11, minutes=30), freq="5min").append(
            pd.date_range(day + pd.Timedelta(hours=13, minutes=5),
                          day + pd.Timedelta(hours=15), freq="5min")
        )
        closes = closes.reindex(index=bars, columns=symbols)
        returns = np.log(closes).diff().replace([np.inf, -np.inf], np.nan)
        count = returns.count()
        sum_squares = returns.pow(2).sum()
        skew = np.sqrt(count) * returns.pow(3).sum() / sum_squares.where(sum_squares > 0).pow(1.5)
        return skew.where(count >= self.min_returns).rename("realized_skew_5m").reindex(symbols)
