"""增长、规模与估值等权合成的基本面复合因子。"""

import numpy as np
import pandas as pd

from factor_base import FactorBase


class GrowthSizeValueComposite(FactorBase):
    """高增长、小市值和低估值的日频横截面复合因子。"""

    def __init__(self):
        super().__init__(
            window="1D",
            data_fields=[
                "tr_yoy",
                "dt_netprofit_yoy",
                "ocf_yoy",
                "total_mv",
                "pe_ttm",
                "pb",
                "ps_ttm",
            ],
        )

    @staticmethod
    def _winsorize_mad(values: pd.Series) -> pd.Series:
        result = pd.to_numeric(values, errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        valid = result.dropna()
        if valid.empty:
            return result

        median = valid.median()
        mad = (valid - median).abs().median()
        if not np.isfinite(mad) or mad == 0:
            return result

        limit = 5 * 1.4826 * mad
        return result.clip(lower=median - limit, upper=median + limit)

    @staticmethod
    def _zscore(values: pd.Series) -> pd.Series:
        result = pd.to_numeric(values, errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        valid = result.dropna()
        if len(valid) < 2:
            return pd.Series(np.nan, index=result.index, dtype=float)

        standard_deviation = valid.std(ddof=0)
        if not np.isfinite(standard_deviation) or standard_deviation == 0:
            return pd.Series(np.nan, index=result.index, dtype=float)
        return (result - valid.mean()) / standard_deviation

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

        growth_components = pd.concat(
            [
                self._zscore(self._winsorize_mad(daily[field]))
                for field in ("tr_yoy", "dt_netprofit_yoy", "ocf_yoy")
            ],
            axis=1,
        )
        growth = growth_components.mean(axis=1).where(
            growth_components.notna().sum(axis=1) >= 2
        )

        log_market_value = np.log(daily["total_mv"].where(daily["total_mv"] > 0))
        size = -self._zscore(self._winsorize_mad(log_market_value))

        value_components = pd.concat(
            [
                self._zscore(
                    self._winsorize_mad(
                        1 / daily[field].where(daily[field] > 0)
                    )
                )
                for field in ("pe_ttm", "pb", "ps_ttm")
            ],
            axis=1,
        )
        value = value_components.mean(axis=1).where(
            value_components.notna().sum(axis=1) >= 2
        )

        standardized_subfactors = pd.concat(
            [self._zscore(growth), self._zscore(size), self._zscore(value)],
            axis=1,
        )
        composite = standardized_subfactors.mean(axis=1).where(
            standardized_subfactors.notna().all(axis=1)
        )
        return composite.reindex(symbols)
