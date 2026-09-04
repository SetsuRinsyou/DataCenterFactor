"""论文未解释成交量定义的21交易日映射。"""

import pandas as pd

from factor_base import FactorBase


class UnexplainedVolume(FactorBase):
    """最近21日实际成交量减去前21日回归预测成交量的日均值。"""

    def __init__(self):
        super().__init__(window="42D", data_fields=["close", "vol"])

    def calculate_daily_factor(
        self,
        trade_date: str,
        symbols: list[str],
        market_data: dict[str, pd.DataFrame],
        history_start: str,
    ) -> pd.Series:
        close_window = market_data["close"].loc[
            history_start:trade_date
        ].reindex(columns=symbols).dropna(how="all")
        volume_window = market_data["vol"].reindex(
            index=close_window.index,
            columns=symbols,
        ).where(lambda values: values >= 0) * 100
        absolute_returns = close_window.pct_change(fill_method=None).abs().iloc[1:]
        aligned_volume = volume_window.iloc[1:]

        regression_returns = absolute_returns.iloc[:21]
        regression_volume = aligned_volume.iloc[:21]
        actual_returns = absolute_returns.iloc[21:42]
        actual_volume = aligned_volume.iloc[21:42]

        paired_regression = regression_returns.notna() & regression_volume.notna()
        x = regression_returns.where(paired_regression)
        y = regression_volume.where(paired_regression)
        x_mean = x.mean()
        y_mean = y.mean()
        covariance = ((x - x_mean) * (y - y_mean)).sum(min_count=2)
        variance = ((x - x_mean) ** 2).sum(min_count=2)
        slope = covariance / variance.where(variance > 0)
        intercept = y_mean - slope * x_mean

        paired_actual = actual_returns.notna() & actual_volume.notna()
        residuals = actual_volume.where(paired_actual) - (
            intercept + slope * actual_returns.where(paired_actual)
        )
        result = residuals.mean().where(paired_regression.sum() >= 2)
        return result.reindex(symbols)
