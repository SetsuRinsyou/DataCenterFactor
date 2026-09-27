"""CNE6 Earnings Yield 二级因子的公开资料等权复现。"""

import pandas as pd

from factor_base import FactorBase
from factors.barra.BarraCNE6Descriptors import (
    calculate_cetop,
    calculate_em,
    calculate_etop,
    calculate_etopf,
    combine_descriptors_equal_weight,
    select_annual_report_window,
)


class BarraEarningsYield(FactorBase):
    """ETOP、ETOPF、CETOP 和 EM 标准化后等权合成。"""

    DEBT_FIELDS = [
        "st_borr_mrq",
        "st_bonds_payable_mrq",
        "non_cur_liab_due_1y_mrq",
        "lt_borr_mrq",
        "bond_payable_mrq",
        "lease_liab_mrq",
    ]

    def __init__(self):
        super().__init__(
            window="7Y",
            data_fields=[
                "total_mv",
                "n_income_attr_p_ttm",
                "daa_ttm",
                "money_cap_mrq",
                "trad_asset_mrq",
                "oth_eqt_tools_p_shr_mrq",
                "forecast_np_12m",
                *self.DEBT_FIELDS,
            ],
            financial_report_fields=["ebit_ttm"],
        )

    def calculate_daily_factor(
        self,
        trade_date: str,
        symbols: list[str],
        market_data: dict[str, pd.DataFrame],
        history_start: str,
        *,
        financial_report_data: pd.DataFrame,
    ) -> pd.Series:
        history = {
            field: market_data[field].loc[
                history_start:trade_date
            ].reindex(columns=symbols)
            for field in self.data_fields
        }
        current = {
            field: values.loc[trade_date] for field, values in history.items()
        }
        market_cap = current["total_mv"] * 10_000
        debt = sum(
            (current[field].fillna(0) for field in self.DEBT_FIELDS),
            start=pd.Series(0.0, index=symbols),
        )
        enterprise_value = (
            market_cap
            + current["oth_eqt_tools_p_shr_mrq"].fillna(0)
            + debt
            - current["money_cap_mrq"]
            - current["trad_asset_mrq"].fillna(0)
        )
        annual, _ = select_annual_report_window(
            financial_report_data, symbols, self.financial_report_fields, count=1
        )
        annual_ebit = annual["ebit_ttm"].iloc[-1]
        cash_earnings = (
            current["n_income_attr_p_ttm"] + current["daa_ttm"].fillna(0)
        )
        descriptors = pd.DataFrame(
            {
                "ETOP": calculate_etop(
                    current["n_income_attr_p_ttm"], market_cap
                ),
                "ETOPF": calculate_etopf(
                    current["forecast_np_12m"] * 10_000, market_cap
                ),
                "CETOP": calculate_cetop(cash_earnings, market_cap),
                "EM": calculate_em(annual_ebit, enterprise_value),
            }
        )
        return combine_descriptors_equal_weight(
            descriptors, market_cap
        ).reindex(symbols)
