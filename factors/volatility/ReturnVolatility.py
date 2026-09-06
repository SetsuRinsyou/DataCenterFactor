"""日收益率的滚动样本标准差。"""

import re

import numpy as np

from factor_base import FactorBase


class ReturnVolatility(FactorBase):
    """日收益率的滚动样本标准差。默认至少 80% 的窗口观测有效，不年化。"""

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
            data_fields=["close", "vol"],
        )

    def calculate_daily_factor(self, trade_date, symbols, market_data, history_start):
        # 框架窗口含起点价格，多出的一个价格点用于首个交易日收益。
        close = market_data["close"].loc[history_start:trade_date].reindex(
            columns=symbols
        )
        volume = market_data["vol"].reindex_like(close)
        valid_close = (close > 0) & (volume > 0)
        returns = close.where(valid_close).pct_change(fill_method=None).iloc[1:]
        returns = returns.replace([np.inf, -np.inf], np.nan)
        return returns.std(ddof=1).where(
            returns.count() >= self.min_periods
        ).reindex(symbols)
