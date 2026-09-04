"""中国市场映射的中证500 FF3 特异波动率日频实现。"""

import numpy as np
import pandas as pd

from factor_base import FactorBase


class FF3IdiosyncraticVolatilityCN(FactorBase):
    """用滚动21交易日FF3回归残差标准差衡量特异波动率。"""

    def __init__(self):
        super().__init__(
            window="21D",
            data_fields=["close"],
            requires_ff3=True,
        )

    def calculate_daily_factor(
        self,
        trade_date: str,
        symbols: list[str],
        market_data: dict,
        history_start: str,
    ) -> pd.Series:
        adjusted_close = market_data["close"].loc[
            history_start:trade_date
        ].reindex(columns=symbols)
        stock_returns = adjusted_close.pct_change(fill_method=None).replace(
            [np.inf, -np.inf], np.nan
        )
        ff3_data = market_data["ff3"].reindex(stock_returns.index)
        stock_excess_returns = stock_returns.sub(ff3_data["rf"], axis="index")
        factor_returns = ff3_data[["mkt", "smb", "hml"]]

        result = pd.Series(index=symbols, dtype=float)
        for symbol in symbols:
            regression_data = pd.concat(
                [
                    stock_excess_returns[symbol].rename("stock"),
                    factor_returns,
                ],
                axis=1,
            ).dropna()
            if len(regression_data) < 15:
                continue
            design = np.column_stack(
                [
                    np.ones(len(regression_data)),
                    regression_data[["mkt", "smb", "hml"]],
                ]
            )
            response = regression_data["stock"].to_numpy()
            coefficients = np.linalg.lstsq(design, response, rcond=None)[0]
            residuals = response - design @ coefficients
            result[symbol] = residuals.std(ddof=1)
        return result.reindex(symbols)
