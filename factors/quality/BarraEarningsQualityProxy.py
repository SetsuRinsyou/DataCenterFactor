"""CNE6 Earnings Quality 的资产负债表应计代理。"""

import numpy as np
import pandas as pd

from factor_base import FactorBase
from factors.barra.BarraCNE6Descriptors import (
    annual_report_values,
    calculate_abs,
    standardize_descriptor,
)


class BarraEarningsQualityProxy(FactorBase):
    """仅使用 ABS 的兼容代理，不包含 ACF。"""

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
                "total_assets_mrq",
                "money_cap_mrq",
                "trad_asset_mrq",
                "total_liab_mrq",
                "daa_ttm",
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
        debt = sum(
            (history[field].fillna(0) for field in self.DEBT_FIELDS),
            start=pd.DataFrame(
                0.0,
                index=history["total_assets_mrq"].index,
                columns=symbols,
            ),
        )
        net_operating_assets = (
            history["total_assets_mrq"]
            - history["money_cap_mrq"]
            - history["trad_asset_mrq"].fillna(0)
            - history["total_liab_mrq"]
            + debt
        )
        annual_noa = annual_report_values(
            net_operating_assets, history["report_end_date"], count=2
        )
        descriptor = pd.Series(np.nan, index=symbols, dtype=float)
        if len(annual_noa) >= 2:
            annual_daa = annual_report_values(
                history["daa_ttm"], history["report_end_date"], count=2
            ).reindex(annual_noa.index)
            annual_assets = annual_report_values(
                history["total_assets_mrq"],
                history["report_end_date"],
                count=2,
            ).reindex(annual_noa.index)
            descriptor = calculate_abs(
                annual_noa.iloc[-1],
                annual_noa.iloc[-2],
                annual_daa.iloc[-1],
                annual_assets.iloc[-1],
            )
        market_cap = history["total_mv"].loc[trade_date] * 10_000
        return standardize_descriptor(descriptor, market_cap).reindex(symbols)
