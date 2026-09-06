"""Alpha158 成交均价除以收盘价，分子分母均采用未复权价格。"""

import numpy as np

from factor_base import FactorBase


class VWAP0(FactorBase):
    """Alpha158 成交均价除以收盘价，分子分母均采用未复权价格。"""

    def __init__(self):
        super().__init__(window="1D", data_fields=["amount","vol","pre_close","change","total_mv","total_share"])

    def calculate_daily_factor(self, trade_date, symbols, market_data, history_start):
        amount = market_data["amount"].reindex(index=[trade_date], columns=symbols).iloc[0]
        vol = market_data["vol"].reindex(index=[trade_date], columns=symbols).iloc[0]
        pre_close = market_data["pre_close"].reindex(index=[trade_date], columns=symbols).iloc[0]
        change = market_data["change"].reindex(index=[trade_date], columns=symbols).iloc[0]
        total_mv = market_data["total_mv"].reindex(index=[trade_date], columns=symbols).iloc[0]
        total_share = market_data["total_share"].reindex(index=[trade_date], columns=symbols).iloc[0]
        # 涨跌额与前收盘价均为未复权字段；用市值/股数交叉检查源数据异常。
        raw_close = pre_close + change
        implied_close = total_mv / total_share.where(total_share > 0)
        consistent = implied_close.isna() | ((raw_close - implied_close).abs() <= 0.011)
        raw_close = raw_close.where((raw_close > 0) & consistent)
        result = amount.where(amount >= 0) * 10 / vol.where(vol > 0) / raw_close
        return result.replace([np.inf, -np.inf], np.nan).reindex(symbols)
