"""因子 RankIC 和分组收益的可视化工具。"""

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
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
    group_net_values: pd.DataFrame,
    output_path: str | Path = DEFAULT_PLOT_DIR / "group_cumulative_returns.png",
) -> Path:
    """绘制每日持仓净值对应的累计收益，并计算每日绩效指标。"""
    if not isinstance(group_net_values, pd.DataFrame):
        raise TypeError("group_net_values must be a pandas DataFrame")
    if group_net_values.empty or group_net_values.shape[1] == 0:
        raise ValueError("group_net_values must contain at least one group")
    wealth_frame = group_net_values.apply(pd.to_numeric, errors="coerce")
    if (not np.isfinite(wealth_frame.to_numpy()).all()
            or (wealth_frame <= 0).any().any()):
        raise ValueError("group_net_values must contain finite positive daily values")
    if not wealth_frame.index.is_unique or not wealth_frame.index.is_monotonic_increasing:
        raise ValueError("group_net_values dates must be unique and increasing")
    if not np.allclose(wealth_frame.iloc[0], 1.0):
        raise ValueError("group_net_values must start at 1")

    figure = Figure(figsize=(12, 6))
    FigureCanvasAgg(figure)
    axis = figure.subplots()

    for group_name in wealth_frame.columns:
        wealth = wealth_frame[group_name]
        returns = wealth.pct_change(fill_method=None).iloc[1:]
        cumulative_returns = wealth - 1
        total_growth = float(wealth.iloc[-1])
        annualized_return = (
            total_growth ** (252 / len(returns)) - 1 if len(returns) else np.nan
        )
        max_drawdown = float((wealth / wealth.cummax() - 1).min())
        return_std = float(returns.std(ddof=1)) if len(returns) >= 2 else np.nan
        sharpe = (
            float(returns.mean()) / return_std * np.sqrt(252)
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

    axis.axhline(0, color="black", linewidth=0.8, alpha=0.6)
    axis.set_title("Group cumulative returns (daily valuation)")
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


def plot_style_correlations(
    summary: pd.DataFrame, factor_name: str, output_path: str | Path,
) -> Path:
    """绘制日截面 Pearson 相关系数的时间均值；无有效数据明确标记 N/A。"""
    means = summary["mean_correlation"]
    figure = Figure(figsize=(max(12, len(means) * 0.55), 7))
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    positions = np.arange(len(means))
    bars = axis.bar(positions, means.fillna(0), color="lightcoral")
    for bar, value in zip(bars, means):
        label = "N/A" if pd.isna(value) else f"{0 if abs(value) < 1e-10 else value:.3f}"
        y = 0 if pd.isna(value) else value
        axis.annotate(label, (bar.get_x() + bar.get_width() / 2, y),
                      xytext=(0, 4 if y >= 0 else -4), textcoords="offset points",
                      ha="center", va="bottom" if y >= 0 else "top", fontsize=8)
    axis.axhline(0, color="grey", linewidth=0.8)
    axis.set_ylim(-1, 1)
    labels = summary.get("display_label", pd.Series(
        [f"Other{i + 1}" for i in positions], index=means.index
    ))
    axis.set_xticks(positions, labels, rotation=45, ha="right", fontsize=9)
    axis.set_ylabel("Mean daily cross-sectional Pearson correlation")
    axis.set_title(f"{factor_name} | Style Correlation", fontsize=11)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    path = prepare_output_path(output_path)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    return path


def plot_report_series(axis, values: pd.DataFrame, title: str) -> None:
    """在报告子图中绘制按交易日对齐的时间序列。"""
    x = prepare_plot_index(values.index)
    for name in values:
        axis.plot(x, values[name], label=str(name), linewidth=1)
    axis.set_title(title, fontsize=10)
    axis.grid(alpha=0.2)
    if isinstance(x, pd.DatetimeIndex):
        locator = AutoDateLocator(minticks=3, maxticks=5)
        axis.xaxis.set_major_locator(locator)
        axis.xaxis.set_major_formatter(ConciseDateFormatter(locator))
    axis.legend(fontsize=7, loc="best")


def plot_report_table(axis, frame: pd.DataFrame) -> None:
    """绘制汇总或年度指标表；缺失统计明确显示 N/A。"""
    axis.axis("off")
    display = frame.map(
        lambda value: f"{value:.4f}" if isinstance(value, (float, np.floating))
        and np.isfinite(value) else ("N/A" if pd.isna(value) else str(value))
    )
    display = display.rename_axis("Metric" if len(frame.columns) == 1 else "Year").reset_index()
    table = axis.table(
        cellText=display.to_numpy(), colLabels=display.columns,
        loc="center", bbox=[0, 0, 1, 1],
        colWidths=([0.6, 0.4] if len(frame.columns) == 1 else None),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#e0e6ef")
        if row == 0:
            cell.set_facecolor("#e9eefb")


def plot_report_cards(figure: Figure, cards: dict) -> None:
    """在报告顶部展示关键指标。"""
    for position, (label, value) in enumerate(cards.items()):
        x = (position + 0.5) / len(cards)
        figure.text(x, 0.91, str(value), ha="center", fontsize=11, weight="bold",
                    bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#dce3ef"})
        figure.text(x, 0.883, label, ha="center", fontsize=8, color="#64748b")


def plot_excess_drawdown(axis, nav: pd.Series) -> None:
    """超额净值曲线及完整每日路径回撤。"""
    plot_report_series(axis, (nav - 1).to_frame("Excess"), "Excess Return & Drawdown")
    right = axis.twinx()
    drawdown = nav / nav.cummax() - 1
    right.fill_between(prepare_plot_index(nav.index), drawdown.to_numpy(), 0,
                       alpha=0.15, color="tab:red", label="Excess drawdown")
    right.yaxis.set_major_formatter(PercentFormatter(1))
    axis.yaxis.set_major_formatter(PercentFormatter(1))


def plot_report_rolling(axis, rolling: dict, series: str, window: int) -> None:
    """滚动 Sharpe 与胜率使用不同纵轴。"""
    plot_report_series(axis, rolling["sharpe"][[series]],
                       f"Rolling Stability ({window} days; full window)")
    axis.legend([f"{window}D Sharpe"], loc="upper left", fontsize=7)
    right = axis.twinx()
    right.plot(prepare_plot_index(rolling["win_rate"].index),
               rolling["win_rate"][series], color="tab:green", linewidth=0.8,
               label="Win rate")
    right.yaxis.set_major_formatter(PercentFormatter(1))
    right.legend(loc="upper right", fontsize=7)


def plot_group_report(report: dict, output_path: str | Path) -> Path:
    """绘制分组报告；接收主流程已经计算并对齐的净值和统计。"""
    figure = Figure(figsize=(17, 11))
    FigureCanvasAgg(figure)
    grid = figure.add_gridspec(3, 3, top=0.83, bottom=0.09, hspace=0.45, wspace=0.35)
    figure.suptitle(report["title"] + " | Close Group Backtest", fontsize=12)
    plot_report_cards(figure, report["group_cards"])
    groups = report["groups"]
    cumulative = groups.subtract(1)
    cumulative["Benchmark"] = report["nav"]["Benchmark"] - 1
    axis = figure.add_subplot(grid[0, :2])
    plot_report_series(axis, cumulative, "Cumulative Return by Group")
    axis.legend(fontsize=6, loc="upper left", ncol=5)
    right = axis.twinx()
    spread = report["nav"]["Spread"] - 1
    right.plot(prepare_plot_index(spread.index), spread, color="red", linewidth=1,
               label="High-low spread")
    right.legend(loc="lower right", fontsize=7)
    axis.yaxis.set_major_formatter(PercentFormatter(1))
    right.yaxis.set_major_formatter(PercentFormatter(1))
    plot_excess_drawdown(figure.add_subplot(grid[0, 2]), report["nav"]["Excess"])
    axis = figure.add_subplot(grid[1, 0])
    report["group_metrics"]["annual_return"].plot.bar(ax=axis, color="tab:cyan")
    axis.set_title("Annualized Return by Group", fontsize=10)
    axis.yaxis.set_major_formatter(PercentFormatter(1))
    axis = figure.add_subplot(grid[1, 1])
    yearly = report["yearly"]["return"].unstack("series")
    yearly[["Excess", "Spread"]].plot.bar(ax=axis)
    axis.set_title("Yearly Excess / High-low Spread", fontsize=10)
    axis.yaxis.set_major_formatter(PercentFormatter(1))
    plot_report_rolling(figure.add_subplot(grid[1, 2]), report["rolling"],
                        "Strategy", report["window"])
    axis = figure.add_subplot(grid[2, :2])
    ic = report["rank_ic"]
    x = prepare_plot_index(ic.index)
    axis.bar(x, ic, color="lightgrey", width=1, alpha=0.6, label="Daily RankIC")
    axis.plot(x, ic.rolling(report["window"], min_periods=1).mean(),
              color="tab:purple", label="Rolling mean")
    axis.axhline(ic.mean(), color="red", linestyle="--", label="Mean IC")
    right = axis.twinx()
    right.plot(x, ic.cumsum(), color="teal", label="Cumulative IC")
    right.set_ylabel("Cumulative IC")
    locator = AutoDateLocator(minticks=3, maxticks=5)
    axis.xaxis.set_major_locator(locator)
    axis.xaxis.set_major_formatter(ConciseDateFormatter(locator))
    axis.set_title("RankIC", fontsize=10)
    axis.legend(fontsize=7, loc="upper left")
    right.legend(fontsize=7, loc="lower right")
    plot_report_table(figure.add_subplot(grid[2, 2]), report["summary"])
    figure.text(0.5, 0.025, report["conventions"], ha="center", fontsize=8)
    path = prepare_output_path(output_path)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    return path


def plot_strategy_report(report: dict, output_path: str | Path) -> Path:
    """绘制单组策略报告，复用分组回测结果而不重新回测。"""
    figure = Figure(figsize=(17, 11))
    FigureCanvasAgg(figure)
    grid = figure.add_gridspec(3, 3, top=0.83, bottom=0.09, hspace=0.45, wspace=0.35)
    figure.suptitle(report["title"] + " | Strategy Performance", fontsize=12)
    plot_report_cards(figure, report["strategy_cards"])
    nav = report["nav"]
    axis = figure.add_subplot(grid[0, :2])
    plot_report_series(axis, nav[["Strategy", "Benchmark"]] - 1, "Cumulative Return")
    right = axis.twinx()
    for name, color in (("Strategy", "tab:blue"), ("Benchmark", "grey")):
        drawdown = nav[name] / nav[name].cummax() - 1
        right.fill_between(prepare_plot_index(nav.index), drawdown.to_numpy(), 0,
                           color=color, alpha=0.08)
    right.yaxis.set_major_formatter(PercentFormatter(1))
    axis.yaxis.set_major_formatter(PercentFormatter(1))
    plot_excess_drawdown(figure.add_subplot(grid[0, 2]), nav["Excess"])
    axis = figure.add_subplot(grid[1, 0])
    yearly = report["yearly"]["return"].unstack("series")
    yearly[["Strategy", "Benchmark", "Excess"]].plot.bar(ax=axis)
    axis.set_title("Yearly Return", fontsize=10)
    axis.yaxis.set_major_formatter(PercentFormatter(1))
    plot_report_rolling(figure.add_subplot(grid[1, 1]), report["rolling"],
                        "Excess", report["window"])
    axis = figure.add_subplot(grid[1, 2])
    drawdown = nav[["Strategy", "Excess"]] / nav[["Strategy", "Excess"]].cummax() - 1
    plot_report_series(axis, drawdown, "Drawdown")
    for name in drawdown:
        axis.fill_between(prepare_plot_index(nav.index), drawdown[name].to_numpy(), 0,
                          alpha=0.1)
    axis.yaxis.set_major_formatter(PercentFormatter(1))
    plot_report_table(figure.add_subplot(grid[2, 0]), report["summary"])
    annual = report["yearly"]
    table = pd.DataFrame()
    for name, prefix in (("Strategy", ""), ("Excess", "Excess ")):
        part = annual.xs(name, level="series")
        for field in ("return", "sharpe", "max_drawdown", "calmar"):
            label = {"return": "Ret", "sharpe": "Sharpe",
                     "max_drawdown": "Max DD", "calmar": "Calmar"}[field]
            table[prefix + label] = part[field]
    axis = figure.add_subplot(grid[2, 1:])
    plot_report_table(axis, table)
    axis.set_title("Yearly Strategy & Excess Metrics", fontsize=10)
    figure.text(0.5, 0.025, report["conventions"], ha="center", fontsize=8)
    path = prepare_output_path(output_path)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    return path
