"""CNE6 Earnings Yield 的非预测描述子代理。"""

import pandas as pd

from factor_base import FactorBase
from factors.barra.BarraCNE6Descriptors import (
    calculate_cetop,
    calculate_em,
    calculate_etop,
    combine_descriptors_equal_weight,
    latest_annual_value,
)


class BarraEarningsYieldProxy(FactorBase):
    """以 ETOP、CETOP 和 EM 代理；不包含分析师预测 ETOFP。"""

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
                "report_end_date",
                "n_income_attr_p_ttm",
                "daa_ttm",
                "ebit_ttm",
                "money_cap_mrq",
                "trad_asset_mrq",
                "oth_eqt_tools_p_shr_mrq",
                *self.DEBT_FIELDS,
            ],
        )

    def calculate_daily_factor(
        self,
        trade_date: str,
        symbols: list[str],
        market_data: dict[str, pd.DataFrame],
        history_start: str,
    ) -> pd.Series:
        history = {
            field: market_data[field].loc[
                history_start:trade_date
            ].reindex(columns=symbols)
            for field in self.data_fields
        }
        current = {field: values.loc[trade_date] for field, values in history.items()}
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
        annual_ebit = latest_annual_value(
            history["ebit_ttm"], history["report_end_date"]
        )
        cash_earnings = (
            current["n_income_attr_p_ttm"] + current["daa_ttm"].fillna(0)
        )
        descriptors = pd.DataFrame(
            {
                "ETOP": calculate_etop(
                    current["n_income_attr_p_ttm"], market_cap
                ),
                "CETOP": calculate_cetop(cash_earnings, market_cap),
                "EM": calculate_em(annual_ebit, enterprise_value),
            }
        )
        return combine_descriptors_equal_weight(
            descriptors, market_cap
        ).reindex(symbols)
