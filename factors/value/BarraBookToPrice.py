"""CNE6 Book-to-Price 二级因子。"""

import pandas as pd

from factor_base import FactorBase
from factors.barra.BarraCNE6Descriptors import (
    calculate_btop,
    standardize_descriptor,
)


class BarraBookToPrice(FactorBase):
    """最近报告期普通股账面价值除以当前总市值。"""

    def __init__(self):
        super().__init__(
            window="1D",
            data_fields=[
                "total_mv",
                "total_hldr_eqy_exc_min_int_mrq",
                "oth_eqt_tools_p_shr_mrq",
            ],
        )

    def calculate_daily_factor(
        self,
        trade_date: str,
        symbols: list[str],
        market_data: dict[str, pd.DataFrame],
        history_start: str,
    ) -> pd.Series:
        daily = {
            field: market_data[field].reindex(
                index=[trade_date], columns=symbols
            ).iloc[0]
            for field in self.data_fields
        }
        book_common_equity = (
            daily["total_hldr_eqy_exc_min_int_mrq"]
            - daily["oth_eqt_tools_p_shr_mrq"].fillna(0)
        )
        market_cap = daily["total_mv"] * 10_000
        descriptor = calculate_btop(book_common_equity, market_cap)
        return standardize_descriptor(descriptor, market_cap).reindex(symbols)
