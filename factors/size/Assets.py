"""论文 Assets 特征的日频实现。"""

from factor_base import FactorBase


class Assets(FactorBase):
    """信号日已生效最新报告的总资产 MRQ。"""

    def __init__(self):
        super().__init__(window="1D", data_fields=["total_assets_mrq"])

    def calculate_daily_factor(self, trade_date, symbols, market_data, history_start):
        return market_data["total_assets_mrq"].reindex(
            index=[trade_date], columns=symbols
        ).iloc[0]
