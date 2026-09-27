"""CNE6 Earnings Quality 二级因子的公开资料等权复现。"""

import pandas as pd

from factor_base import FactorBase
from factors.barra.BarraCNE6Descriptors import (
    select_annual_report_window,
    calculate_abs,
    calculate_acf,
    combine_descriptors_equal_weight,
)


class BarraEarningsQuality(FactorBase):
    """ABS 和 ACF 标准化后等权合成。"""

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
                "total_assets_mrq",
                "n_income_attr_p_ttm",
                "n_cashflow_act_ttm",
                "n_cashflow_inv_act_ttm",
                "daa_ttm",
                "depreciation_ttm_pit",
            ],
            financial_report_fields=[
                "total_assets_mrq",
                "money_cap_mrq",
                "trad_asset_mrq",
                "total_liab_mrq",
                "daa_ttm",
                "depr_fa_coga_dpba_ttm",
                "amort_intang_assets_ttm",
                "lt_amort_deferred_exp_ttm",
                "use_right_asset_dep_ttm",
                *self.DEBT_FIELDS,
            ],
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
        annual, _ = select_annual_report_window(
            financial_report_data, symbols, self.financial_report_fields, count=2
        )
        debt = sum(
            (annual[field].fillna(0) for field in self.DEBT_FIELDS),
            start=pd.DataFrame(
                0.0,
                index=annual["total_assets_mrq"].index,
                columns=symbols,
            ),
        )
        net_operating_assets = (
            annual["total_assets_mrq"]
            - annual["money_cap_mrq"]
            - annual["trad_asset_mrq"].fillna(0)
            - annual["total_liab_mrq"]
            + debt
        )
        depreciation_fields = [
            "depr_fa_coga_dpba_ttm", "amort_intang_assets_ttm",
            "lt_amort_deferred_exp_ttm", "use_right_asset_dep_ttm",
        ]
        cashflow_depreciation = sum(
            (annual[field].fillna(0) for field in depreciation_fields),
            start=pd.DataFrame(0.0, index=net_operating_assets.index, columns=symbols),
        )
        cashflow_depreciation = cashflow_depreciation.where(
            sum(annual[field].notna() for field in depreciation_fields) > 0
        )
        annual_daa = annual["daa_ttm"].combine_first(cashflow_depreciation)
        abs_descriptor = calculate_abs(
            net_operating_assets.iloc[-1],
            net_operating_assets.iloc[-2],
            annual_daa.iloc[-1],
            annual["total_assets_mrq"].iloc[-1],
        )

        current = {
            field: values.loc[trade_date] for field, values in history.items()
        }
        acf_descriptor = calculate_acf(
            current["n_income_attr_p_ttm"],
            current["n_cashflow_act_ttm"],
            current["n_cashflow_inv_act_ttm"],
            current["daa_ttm"].combine_first(
                current["depreciation_ttm_pit"]
            ),
            current["total_assets_mrq"],
        )
        descriptors = pd.DataFrame(
            {"ABS": abs_descriptor, "ACF": acf_descriptor}
        )
        market_cap = current["total_mv"] * 10_000
        return combine_descriptors_equal_weight(
            descriptors, market_cap
        ).reindex(symbols)
