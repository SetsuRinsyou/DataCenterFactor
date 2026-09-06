"""最高价相对前收盘价收益率的滚动样本标准差。"""

import re

import numpy as np

from factor_base import FactorBase


class HighReturnVolatility(FactorBase):
    """最高价相对前收盘价收益率的滚动样本标准差。默认至少 80% 的窗口观测有效，不年化。"""

    def __init__(self, window: str = "84D", min_periods: int | None = None):
        match = re.fullmatch(r"([1-9]\d*)D", window)
        if match is None:
            raise ValueError("window must be a trading-day window such as '84D'")
        window_days = int(match.group(1))
        if min_periods is None:
            min_periods = int(np.ceil(window_days * 0.8))
        if (
            isinstance(min_periods, bool)
            or not isinstance(min_periods, int)
            or not 2 <= min_periods <= window_days
        ):
            raise ValueError("min_periods must be an integer between 2 and window days")
        self.min_periods = min_periods
        super().__init__(
            window=window,
            data_fields=["close", "high", "vol"],
        )

    def calculate_daily_factor(self, trade_date, symbols, market_data, history_start):
        # 框架窗口含起点价格，多出的一个价格点用于首个交易日收益。
        close = market_data["close"].loc[history_start:trade_date].reindex(
            columns=symbols
        )
        volume = market_data["vol"].reindex_like(close)
        valid_close = (close > 0) & (volume > 0)
        previous_close = close.where(valid_close).shift(1)
        high = market_data["high"].reindex_like(close)
        high_returns = (
            high.where(valid_close & (high > 0)) / previous_close - 1
        ).iloc[1:].replace([np.inf, -np.inf], np.nan)
        high_std = high_returns.std(ddof=1).where(
            high_returns.count() >= self.min_periods
        )
        return high_std.reindex(symbols)
