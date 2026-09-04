"""因子 RankIC 和分组收益的可视化工具。"""

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter


DEFAULT_PLOT_DIR = Path(__file__).resolve().parent / "tmp" / "plots"


def validate_forward_days(forward_days: int) -> None:
    if (
        isinstance(forward_days, bool)
        or not isinstance(forward_days, (int, np.integer))
        or forward_days <= 0
    ):
        raise ValueError("forward_days must be a positive integer")


def prepare_output_path(output_path: str | Path) -> Path:
    path = Path(output_path).expanduser()
    if not path.suffix:
        path = path.with_suffix(".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def prepare_plot_index(index: pd.Index) -> pd.Index:
    if isinstance(index, pd.DatetimeIndex):
        return index

    compact_dates = pd.to_datetime(
        index.astype(str), format="%Y%m%d", errors="coerce"
    )
    if compact_dates.notna().all():
        return pd.DatetimeIndex(compact_dates)

    general_dates = pd.to_datetime(index, errors="coerce")
    if general_dates.notna().all():
        return pd.DatetimeIndex(general_dates)
    return index


def plot_ic(
    rank_ic: pd.Series,
    forward_days: int,
    output_path: str | Path = DEFAULT_PLOT_DIR / "ic.png",
    moving_average_window: int = 20,
) -> Path:
    """以柱状图绘制逐期 RankIC，并叠加滚动均线和汇总指标。"""
    if not isinstance(rank_ic, pd.Series):
        raise TypeError("rank_ic must be a pandas Series")
    validate_forward_days(forward_days)
    if (
        isinstance(moving_average_window, bool)
        or not isinstance(moving_average_window, (int, np.integer))
        or moving_average_window <= 0
    ):
        raise ValueError("moving_average_window must be a positive integer")

    values = pd.to_numeric(rank_ic, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    valid = values.dropna()
    if valid.empty:
        raise ValueError("rank_ic must contain at least one finite value")

    ic_mean = float(valid.mean())
    ic_std = float(valid.std(ddof=1)) if len(valid) >= 2 else np.nan
    annualized_icir = (
        ic_mean / ic_std * np.sqrt(252 / forward_days)
        if np.isfinite(ic_std) and ic_std > 0
        else np.nan
    )
    moving_average = values.rolling(
        window=int(moving_average_window), min_periods=1
    ).mean()
    x_values = prepare_plot_index(values.index)

    figure = Figure(figsize=(12, 6))
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    axis.bar(
        x_values,
        values,
        width=1.0,
        color="tab:blue",
        alpha=0.45,
        label=f"IC (mean={ic_mean:.4f}, annualized ICIR={annualized_icir:.4f})",
    )
    axis.plot(
        x_values,
        moving_average,
        color="tab:orange",
        linewidth=1.8,
        label=f"{moving_average_window}-period moving average",
    )
    axis.axhline(0, color="black", linewidth=0.8, alpha=0.6)
    axis.set_title(f"RankIC ({forward_days}-day forward return)")
    axis.set_xlabel("Trade date")
    axis.set_ylabel("RankIC")
    axis.grid(alpha=0.25)
    axis.legend(loc="best")
    figure.autofmt_xdate()
    figure.tight_layout()

    path = prepare_output_path(output_path)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    return path


def plot_group_cumulative_returns(
    group_returns: pd.DataFrame,
    forward_days: int,
    output_path: str | Path = DEFAULT_PLOT_DIR / "group_cumulative_returns.png",
) -> Path:
    """按 1/N 采样非重叠 N 日收益，绘制各组累计收益及绩效指标。"""
    if not isinstance(group_returns, pd.DataFrame):
        raise TypeError("group_returns must be a pandas DataFrame")
    validate_forward_days(forward_days)
    if group_returns.empty or group_returns.shape[1] == 0:
        raise ValueError("group_returns must contain at least one group")

    numeric_returns = group_returns.apply(pd.to_numeric, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    sampled_returns = numeric_returns.iloc[::forward_days]
    periods_per_year = 252 / forward_days

    figure = Figure(figsize=(12, 6))
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    plotted_groups = 0

    for group_name in sampled_returns.columns:
        returns = sampled_returns[group_name].dropna()
        if returns.empty:
            continue
        if (returns < -1).any():
            raise ValueError(f"{group_name} contains a return below -100%")

        wealth = (1 + returns).cumprod()
        cumulative_returns = wealth - 1
        total_growth = float(wealth.iloc[-1])
        annualized_return = (
            total_growth ** (periods_per_year / len(returns)) - 1
            if total_growth >= 0
            else np.nan
        )
        running_peak = wealth.cummax().clip(lower=1.0)
        max_drawdown = float((wealth / running_peak - 1).min())
        return_std = float(returns.std(ddof=1)) if len(returns) >= 2 else np.nan
        sharpe = (
            float(returns.mean()) / return_std * np.sqrt(periods_per_year)
            if np.isfinite(return_std) and return_std > 0
            else np.nan
        )

        axis.plot(
            prepare_plot_index(cumulative_returns.index),
            cumulative_returns,
            linewidth=1.5,
            label=(
                f"{group_name} (annualized={annualized_return:.2%}, "
                f"max drawdown={max_drawdown:.2%}, Sharpe={sharpe:.2f})"
            ),
        )
        plotted_groups += 1

    if plotted_groups == 0:
        raise ValueError("group_returns has no finite values after 1/N sampling")

    axis.axhline(0, color="black", linewidth=0.8, alpha=0.6)
    axis.set_title(
        f"Group cumulative returns (sampled every {forward_days} observations)"
    )
    axis.set_xlabel("Trade date")
    axis.set_ylabel("Cumulative return")
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.grid(alpha=0.25)
    axis.legend(loc="best", fontsize=8)
    figure.autofmt_xdate()
    figure.tight_layout()

    path = prepare_output_path(output_path)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    return path
