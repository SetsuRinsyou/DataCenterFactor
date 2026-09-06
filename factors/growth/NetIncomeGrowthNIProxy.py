"""归母净利润 TTM的交易日快照变化率。"""

import re

import numpy as np

from factor_base import FactorBase


class NetIncomeGrowthNIProxy(FactorBase):
    """(当前值 - 窗口起点值) / 起点值绝对值，不是报告期环比或同比。"""

    def __init__(self, window: str = "63D"):
        if re.fullmatch(r"[1-9]\d*D", window) is None:
            raise ValueError("window must be a positive trading-day window such as '63D'")
        super().__init__(window=window, data_fields=["n_income_attr_p_ttm"])

    def calculate_daily_factor(self, trade_date, symbols, market_data, history_start):
        # 精确匹配交易日端点，缺失时不向其他日期回退。
        values = market_data["n_income_attr_p_ttm"].reindex(
            index=[history_start, trade_date], columns=symbols
        )
        previous = values.iloc[0]
        current = values.iloc[1]
        denominator = previous.abs().where(previous != 0)
        return ((current - previous) / denominator).replace(
            [np.inf, -np.inf], np.nan
        ).reindex(symbols)
