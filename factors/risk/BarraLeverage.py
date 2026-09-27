"""CNE6 Leverage 二级因子的公开资料等权复现。"""

import pandas as pd

from factor_base import FactorBase
from factors.barra.BarraCNE6Descriptors import (
    calculate_blev,
    calculate_dtoa,
    calculate_mlev,
    combine_descriptors_equal_weight,
    latest_annual_value,
)


class BarraLeverage(FactorBase):
    """MLEV、BLEV 和 DTOA 标准化后等权合成。"""

    def __init__(self):
        super().__init__(
            window="7Y",
            data_fields=[
                "total_mv",
                "report_end_date",
                "total_hldr_eqy_exc_min_int_mrq",
                "oth_eqt_tools_p_shr_mrq",
                "total_liab_mrq",
                "total_cur_liab_mrq",
                "total_assets_mrq",
            ],
        )

    def calculate_daily_factor(
        self,
        trade_date: str,
        symbols: list[str],
        market_data: dict[str, pd.DataFrame],
        history_start: str,
    ) -> pd.Series:
        report_end_dates = market_data["report_end_date"].loc[
            history_start:trade_date
        ].reindex(columns=symbols)

        def annual(field: str, fill_value=None) -> pd.Series:
            values = market_data[field].loc[
                history_start:trade_date
            ].reindex(columns=symbols)
            if fill_value is not None:
                values = values.fillna(fill_value)
            return latest_annual_value(values, report_end_dates)

        preferred_equity = annual("oth_eqt_tools_p_shr_mrq", fill_value=0)
        total_liabilities = annual("total_liab_mrq")
        current_liabilities = annual("total_cur_liab_mrq")
        total_assets = annual("total_assets_mrq")
        book_common_equity = (
            annual("total_hldr_eqy_exc_min_int_mrq") - preferred_equity
        )
        long_term_debt = total_liabilities - current_liabilities

        market_cap_history = market_data["total_mv"].loc[
            history_start:trade_date
        ].reindex(columns=symbols)
        current_market_cap = market_cap_history.loc[trade_date] * 10_000
        previous_market_cap = market_cap_history.shift(1).loc[trade_date] * 10_000

        descriptors = pd.DataFrame(
            {
                "MLEV": calculate_mlev(
                    previous_market_cap, preferred_equity, long_term_debt
                ),
                "BLEV": calculate_blev(
                    book_common_equity, preferred_equity, long_term_debt
                ),
                "DTOA": calculate_dtoa(total_liabilities, total_assets),
            }
        )
        return combine_descriptors_equal_weight(
            descriptors, current_market_cap
        ).reindex(symbols)
