"""中国市场映射的中证500 Market beta 日频实现。"""

import numpy as np

from factor_base import FactorBase


class MarketBetaCN(FactorBase):
    """一年相关系数乘以股票与市场五年三日收益波动率之比。"""

    def __init__(self):
        # 1260个重叠三日收益需要1263个价格点，即向前移动1262个交易日。
        super().__init__(
            window="1262D",
            data_fields=["close", "gc001_weight"],
            index_code="000905.SH",
        )

    def calculate_daily_factor(
        self,
        trade_date: str,
        symbols: list[str],
        market_data: dict,
        history_start: str,
    ):
        adjusted_close = market_data["close"].loc[
            history_start:trade_date, symbols
        ]
        index_close = market_data["index_close"].loc[
            history_start:trade_date, self.index_code
        ].reindex(adjusted_close.index)
        risk_free_rate = (
            market_data["gc001_weight"]
            .loc[history_start:trade_date]
            .max(axis=1)
            .reindex(adjusted_close.index)
            / 100
            / 365
        )

        stock_returns = adjusted_close.pct_change(fill_method=None).replace(
            [np.inf, -np.inf], np.nan
        )
        index_returns = index_close.pct_change(fill_method=None).replace(
            [np.inf, -np.inf], np.nan
        )
        stock_excess_returns = stock_returns.sub(risk_free_rate, axis="index")
        market_excess_returns = index_returns - risk_free_rate

        three_day_risk_free = (1 + risk_free_rate).rolling(
            3, min_periods=3
        ).apply(lambda values: values.prod(), raw=True) - 1
        stock_three_day_returns = (
            adjusted_close / adjusted_close.shift(3) - 1
        ).sub(three_day_risk_free, axis="index")
        market_three_day_returns = (
            index_close / index_close.shift(3) - 1 - three_day_risk_free
        )

        correlation = stock_excess_returns.tail(252).corrwith(
            market_excess_returns.tail(252)
        )
        stock_volatility = stock_three_day_returns.tail(1260).std()
        market_volatility = market_three_day_returns.tail(1260).std()
        if not market_volatility > 0:
            return correlation.reindex(symbols) * np.nan
        return (
            correlation * stock_volatility / market_volatility
        ).reindex(symbols)
