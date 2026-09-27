"""CNE6 Liquidity 二级因子的公开资料等权复现。"""

import pandas as pd

from factor_base import FactorBase
from factors.barra.BarraCNE6Descriptors import (
    calculate_atvr,
    calculate_stoa,
    calculate_stom,
    calculate_stoq,
    combine_descriptors_equal_weight,
)


class BarraLiquidity(FactorBase):
    """STOM、STOQ、STOA 和 ATVR 标准化后等权合成。"""

    def __init__(self):
        super().__init__(
            window="252D",
            data_fields=["turnover_rate_f", "total_mv"],
        )

    def calculate_daily_factor(
        self,
        trade_date: str,
        symbols: list[str],
        market_data: dict[str, pd.DataFrame],
        history_start: str,
    ) -> pd.Series:
        turnover = (
            market_data["turnover_rate_f"]
            .loc[history_start:trade_date]
            .reindex(columns=symbols)
            / 100
        )
        market_cap = market_data["total_mv"].reindex(
            index=[trade_date], columns=symbols
        ).iloc[0] * 10_000
        descriptors = pd.DataFrame(
            {
                "STOM": calculate_stom(turnover),
                "STOQ": calculate_stoq(turnover),
                "STOA": calculate_stoa(turnover),
                "ATVR": calculate_atvr(turnover),
            }
        )
        return combine_descriptors_equal_weight(
            descriptors, market_cap
        ).reindex(symbols)
