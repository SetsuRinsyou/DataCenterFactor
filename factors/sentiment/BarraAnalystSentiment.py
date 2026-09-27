"""CNE6 Analyst Sentiment 二级因子的公开资料等权复现。"""

import numpy as np
import pandas as pd

from factor_base import FactorBase
from factors.barra.BarraCNE6Descriptors import (
    calculate_epsf_c,
    calculate_etopf_c,
    calculate_rr,
    combine_descriptors_equal_weight,
)


class BarraAnalystSentiment(FactorBase):
    """RR、ETOPF_C 和 EPSF_C 标准化后等权合成。"""

    def __init__(self):
        super().__init__(
            window="252D",
            data_fields=[
                "total_mv",
                "total_share",
                "forecast_eps_12m",
                "forecast_revision_up_count",
                "forecast_revision_down_count",
                "forecast_revision_total_count",
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
        up_revisions = history["forecast_revision_up_count"].tail(63)
        down_revisions = history["forecast_revision_down_count"].tail(63)
        total_revisions = history["forecast_revision_total_count"].tail(63)
        period_up = pd.DataFrame(
            [
                up_revisions.iloc[start:start + 21].sum(
                    axis=0, min_count=1
                )
                for start in range(0, 63, 21)
            ]
        )
        period_down = pd.DataFrame(
            [
                down_revisions.iloc[start:start + 21].sum(
                    axis=0, min_count=1
                )
                for start in range(0, 63, 21)
            ]
        )
        period_total = pd.DataFrame(
            [
                total_revisions.iloc[start:start + 21].sum(
                    axis=0, min_count=1
                )
                for start in range(0, 63, 21)
            ]
        )
        rr = calculate_rr(
            period_up,
            period_down,
            period_total,
            [1.0, 2.0, 3.0],
        )

        forecast_eps = history["forecast_eps_12m"]
        price = history["total_mv"].divide(history["total_share"])
        forecast_ep = forecast_eps.divide(price.where(price != 0)).replace(
            [np.inf, -np.inf], np.nan
        )
        snapshot_offsets = [252, 189, 126, 63, 0]
        eps_levels = pd.DataFrame(
            [forecast_eps.iloc[-offset - 1] for offset in snapshot_offsets]
        )
        ep_levels = pd.DataFrame(
            [forecast_ep.iloc[-offset - 1] for offset in snapshot_offsets]
        )
        previous_eps = eps_levels.shift(1)
        previous_ep = ep_levels.shift(1)
        eps_denominator = (
            eps_levels.abs() + previous_eps.abs()
        ) / 2
        ep_denominator = (
            ep_levels.abs() + previous_ep.abs()
        ) / 2
        quarterly_eps_changes = (
            (eps_levels - previous_eps) / eps_denominator.where(
                eps_denominator != 0
            )
        ).iloc[1:]
        quarterly_ep_changes = (
            (ep_levels - previous_ep) / ep_denominator.where(
                ep_denominator != 0
            )
        ).iloc[1:]
        epsf_c = calculate_epsf_c(
            quarterly_eps_changes, [3.0, 5.0, 7.0, 9.0]
        )
        etopf_c = calculate_etopf_c(
            quarterly_ep_changes, [3.0, 5.0, 7.0, 9.0]
        )

        descriptors = pd.DataFrame(
            {"RR": rr, "ETOPF_C": etopf_c, "EPSF_C": epsf_c}
        )
        market_cap = history["total_mv"].loc[trade_date] * 10_000
        return combine_descriptors_equal_weight(
            descriptors, market_cap
        ).reindex(symbols)
