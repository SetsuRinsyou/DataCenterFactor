"""基于多周期价格和成交量趋势的日频截面因子。"""

import numpy as np
import pandas as pd

from factor_base import FactorBase


class TrendScore(FactorBase):
    """用最近已实现的5日盈亏回归系数评估最新趋势特征。"""

    WINDOWS = (3, 5, 10, 20, 50, 100, 200, 300, 400)
    PREDICTION_DAYS = 5
    SIGNAL_LAG_DAYS = PREDICTION_DAYS + 1
    HISTORY_DAYS = max(WINDOWS) + SIGNAL_LAG_DAYS - 1
    MIN_REGRESSION_SAMPLES = 100

    def __init__(self):
        super().__init__(
            window=f"{self.HISTORY_DAYS}D",
            data_fields=["close", "vol"],
        )

    def calculate_daily_factor(
        self,
        trade_date: str,
        symbols: list[str],
        market_data: dict[str, pd.DataFrame],
        history_start: str,
    ) -> pd.Series:
        result = pd.Series(index=symbols, dtype=float)
        close_window = market_data["close"].loc[
            history_start:trade_date
        ].reindex(columns=symbols).tail(self.HISTORY_DAYS + 1)
        if (
            len(close_window) < self.HISTORY_DAYS + 1
            or close_window.index[-1] != trade_date
        ):
            return result

        volume_window = market_data["vol"].reindex(
            index=close_window.index,
            columns=symbols,
        )
        close_values = close_window.to_numpy(dtype=float)
        volume_values = volume_window.to_numpy(dtype=float)
        close_values[~np.isfinite(close_values) | (close_values <= 0)] = np.nan
        volume_values[~np.isfinite(volume_values) | (volume_values <= 0)] = np.nan

        feature_values = []
        for values in (close_values, volume_values):
            valid = np.isfinite(values)
            cumulative_sum = np.vstack(
                [
                    np.zeros(values.shape[1]),
                    np.cumsum(np.where(valid, values, 0.0), axis=0),
                ]
            )
            cumulative_count = np.vstack(
                [np.zeros(values.shape[1], dtype=int), np.cumsum(valid, axis=0)]
            )
            end = len(values)
            training_end = end - self.SIGNAL_LAG_DAYS
            denominators = (
                values[-1],
                values[-(self.SIGNAL_LAG_DAYS + 1)],
            )

            current_features = []
            training_features = []
            for window in self.WINDOWS:
                current_count = (
                    cumulative_count[end] - cumulative_count[end - window]
                )
                training_count = (
                    cumulative_count[training_end]
                    - cumulative_count[training_end - window]
                )
                current_mean = (
                    cumulative_sum[end] - cumulative_sum[end - window]
                ) / window
                training_mean = (
                    cumulative_sum[training_end]
                    - cumulative_sum[training_end - window]
                ) / window
                current_features.append(
                    np.where(
                        current_count == window,
                        current_mean / denominators[0],
                        np.nan,
                    )
                )
                training_features.append(
                    np.where(
                        training_count == window,
                        training_mean / denominators[1],
                        np.nan,
                    )
                )
            feature_values.append((current_features, training_features))

        current_columns = []
        training_columns = []
        for window_position in range(len(self.WINDOWS)):
            for field_position in range(2):
                current_columns.append(
                    feature_values[field_position][0][window_position]
                )
                training_columns.append(
                    feature_values[field_position][1][window_position]
                )
        current_features = np.column_stack(current_columns)
        training_features = np.column_stack(training_columns)
        realized_returns = (
            close_values[-1] / close_values[-(self.PREDICTION_DAYS + 1)] - 1
        )

        regression_mask = np.isfinite(realized_returns) & np.isfinite(
            training_features
        ).all(axis=1)
        if regression_mask.sum() < self.MIN_REGRESSION_SAMPLES:
            return result

        design = np.column_stack(
            [
                np.ones(regression_mask.sum()),
                training_features[regression_mask],
            ]
        )
        coefficients = np.linalg.lstsq(
            design,
            realized_returns[regression_mask],
            rcond=None,
        )[0][1:]

        prediction_mask = np.isfinite(current_features).all(axis=1)
        result.iloc[np.flatnonzero(prediction_mask)] = (
            current_features[prediction_mask] @ coefficients
        )
        return result
