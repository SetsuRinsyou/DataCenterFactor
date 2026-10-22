"""通用因子计算、评价和落库入口。"""

import argparse
import importlib
import json
from pathlib import Path
from time import perf_counter

import duckdb
import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS

from data_manager import DataManager
from factor_base import FactorBase
from eval_plot import (
    plot_group_report, plot_strategy_report,
    plot_style_correlations,
)


SCRIPT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行因子计算、评价并保存因子值")
    parser.add_argument(
        "--db-path",
        default=str(SCRIPT_ROOT / "data" / "src_data.duckdb"),
        help="行情数据库路径",
    )
    parser.add_argument(
        "--output-db",
        default=str(SCRIPT_ROOT / "data" / "factor_results.duckdb"),
        help="保存因子值的新 DuckDB 文件路径",
    )
    parser.add_argument(
        "--factor-db",
        help="基础因子数据库路径，默认使用 output-db，仅声明 factor_fields 时读取",
    )
    parser.add_argument(
        "--minute-db-path",
        default=str(SCRIPT_ROOT / "data" / "market_1min.duckdb"),
        help="分钟行情数据库路径，仅分钟因子使用",
    )
    parser.add_argument("--pool-name", default="zz500", help="股票池名称")
    parser.add_argument("--start-date", required=True, help="开始日期，格式 YYYYMMDD")
    parser.add_argument("--end-date", required=True, help="结束日期，格式 YYYYMMDD")
    parser.add_argument(
        "--forward-days",
        required=True,
        type=int,
        help="未来收益持有的交易日数量",
    )
    parser.add_argument(
        "--factor-module",
        default="factors.momentum.STM",
        help="因子类所在的 Python 模块",
    )
    parser.add_argument(
        "--factor-class",
        default="StandardMomentum",
        help="继承 FactorBase 的因子类名",
    )
    parser.add_argument(
        "--factor-params",
        default="{}",
        help='因子构造参数 JSON，例如 {"window":"252D"}',
    )
    parser.add_argument("--n-groups", type=int, default=5, help="因子分组数，默认五组")
    parser.add_argument(
        "--plot-dir", type=Path, default=SCRIPT_ROOT / "tmp" / "plots",
        help="图表输出目录，实验时应指定实验目录内的路径",
    )
    parser.add_argument("--benchmark", help="基准指数代码；指定后生成完整报告")
    parser.add_argument("--strategy-group", type=int, help="策略组编号（从1开始），默认最高组")
    parser.add_argument("--rolling-window", type=int, default=60, help="滚动统计窗口")
    parser.add_argument("--style-db", help="风格因子数据库，启用风格相关性分析")
    parser.add_argument("--style-fields", nargs="+", help="风格数据库中的因子列名")
    return parser.parse_args()


def load_factor(
    module_name: str,
    class_name: str,
    params_json: str,
) -> FactorBase:
    """从指定模块加载并实例化 FactorBase 子类。"""
    params = json.loads(params_json)
    if not isinstance(params, dict):
        raise ValueError("factor-params must be a JSON object")

    module = importlib.import_module(module_name)
    factor_class = getattr(module, class_name)
    if not isinstance(factor_class, type) or not issubclass(factor_class, FactorBase):
        raise TypeError(f"{module_name}.{class_name} must inherit FactorBase")
    return factor_class(**params)


def calculate_ic_statistics(
    rank_ic: pd.Series,
    forward_days: int,
) -> dict[str, float | int]:
    """计算平均 IC、周期修正 ICIR 和 Newey-West t 检验。"""
    if forward_days <= 0:
        raise ValueError("forward_days must be positive")

    values = rank_ic.dropna().to_numpy(dtype=float)
    if len(values) < 2:
        raise ValueError("at least two valid IC observations are required")

    observation_count = len(values)
    mean_ic = float(values.mean())
    standard_deviation = float(values.std(ddof=1))
    annualized_icir = (
        mean_ic / standard_deviation * np.sqrt(252 / forward_days)
        if standard_deviation > 0
        else np.nan
    )

    # 连续信号日的 forward_days 日收益互相重叠，使用仅含常数项的 OLS
    # 和 Bartlett 核 HAC 协方差估计均值的 Newey-West 检验。
    nw_lags = min(forward_days - 1, observation_count - 1)
    hac_result = OLS(values, np.ones((observation_count, 1))).fit(
        cov_type="HAC",
        cov_kwds={
            "maxlags": nw_lags,
            "kernel": "bartlett",
            "use_correction": True,
        },
        use_t=True,
    )
    nw_standard_error = float(hac_result.bse[0])
    nw_t_stat = float(hac_result.tvalues[0])
    nw_p_value = float(hac_result.pvalues[0])
    return {
        "observations": observation_count,
        "mean_ic": mean_ic,
        "ic_std": standard_deviation,
        "annualized_icir": float(annualized_icir),
        "nw_lags": nw_lags,
        "nw_standard_error": float(nw_standard_error),
        "nw_t_stat": float(nw_t_stat),
        "nw_p_value": nw_p_value,
    }


def calculate_performance(net_values: pd.DataFrame) -> pd.DataFrame:
    """每日净值的年化绩效；零无风险利率，回撤使用完整路径。"""
    if (len(net_values) < 2 or not net_values.index.is_unique
            or not net_values.index.is_monotonic_increasing
            or not np.isfinite(net_values.to_numpy()).all()
            or (net_values <= 0).any().any()):
        raise ValueError("performance requires >= 2 aligned positive daily NAV rows")
    returns = net_values.pct_change(fill_method=None).iloc[1:]
    annual = (net_values.iloc[-1] / net_values.iloc[0]) ** (252 / len(returns)) - 1
    volatility = returns.std(ddof=1) * np.sqrt(252)
    drawdown = (net_values / net_values.cummax() - 1).min()
    return pd.DataFrame({
        "annual_return": annual,
        "volatility": volatility,
        "sharpe": returns.mean() * 252 / volatility.replace(0, np.nan),
        "max_drawdown": drawdown,
        "calmar": annual / (-drawdown).replace(0, np.nan),
        "win_rate": (returns > 0).mean(),
    })


def calculate_yearly_performance(net_values: pd.DataFrame) -> pd.DataFrame:
    """年度首日收益包含上一年末至该日的变化；首年从初始净值开始。"""
    years = pd.DatetimeIndex(pd.to_datetime(net_values.index.astype(str))).year
    results = []
    for year in sorted(set(years)):
        positions = np.flatnonzero(years == year)
        frame = net_values.iloc[max(0, positions[0] - 1):positions[-1] + 1]
        if len(frame) < 2:
            continue
        metrics = calculate_performance(frame)
        metrics["return"] = frame.iloc[-1] / frame.iloc[0] - 1
        metrics["year"] = year
        results.append(metrics.rename_axis("series").reset_index())
    if not results:
        return pd.DataFrame(columns=["year", "series", "return"]).set_index(
            ["year", "series"]
        )
    return pd.concat(results, ignore_index=True).set_index(["year", "series"])


def calculate_rolling_performance(net_values: pd.DataFrame, window: int) -> dict:
    """按完整交易日窗口计算年化 Sharpe 和日收益胜率。"""
    if isinstance(window, bool) or not isinstance(window, int) or window < 2:
        raise ValueError("rolling window must be an integer >= 2")
    returns = net_values.pct_change(fill_method=None)
    rolling = returns.rolling(window, min_periods=window)
    return {
        "sharpe": rolling.mean() / rolling.std(ddof=1).replace(0, np.nan) * np.sqrt(252),
        "win_rate": returns.gt(0).where(returns.notna()).rolling(
            window, min_periods=window
        ).mean(),
    }


def load_benchmark(data_manager: DataManager, index: pd.Index, code: str) -> pd.Series:
    """严格按每日净值日期读取基准 close，不填补缺失行情。"""
    prices = data_manager.get_index_data(code, index.tolist(), "1D")
    close = prices["close"].droplevel("ts_code").reindex(index)
    if close.empty or not np.isfinite(close.to_numpy()).all() or (close <= 0).any():
        missing = close.index[close.isna() | (close <= 0)].tolist()
        raise ValueError(f"benchmark {code} has missing/invalid close: {missing[:10]}")
    return (close / close.iloc[0]).rename("Benchmark")


def create_backtest_reports(
    group_net_values: pd.DataFrame,
    benchmark: pd.Series,
    rank_ic: pd.Series,
    trade_records: list[dict],
    forward_days: int,
    strategy_group: int,
    rolling_window: int,
    title: str,
    output_dir: Path,
) -> tuple[Path, Path]:
    """生成两类 close 报告并保存其底层数据；不写入源数据库。

    超额为净值比值；Spread 为最高组减最低组每日收益差的复利研究曲线，
    相当于每日重置两腿名义敞口，不代表真实融券组合。
    换手率为双边成交额/调仓前净资产，均值含首次建仓及零交易日。
    """
    if not 1 <= strategy_group <= group_net_values.shape[1]:
        raise ValueError("strategy_group must be between 1 and n_groups")
    if not benchmark.index.equals(group_net_values.index):
        raise ValueError("benchmark dates must equal daily NAV dates")
    groups = group_net_values
    group_metrics = calculate_performance(groups)
    strategy_name = groups.columns[strategy_group - 1]
    nav = pd.DataFrame({"Strategy": groups[strategy_name], "Benchmark": benchmark})
    nav["Excess"] = nav["Strategy"] / nav["Benchmark"]
    daily = groups.pct_change(fill_method=None)
    spread_returns = daily.iloc[:, -1] - daily.iloc[:, 0]
    spread_returns.iloc[0] = 0
    nav["Spread"] = (1 + spread_returns).cumprod()
    metrics = calculate_performance(nav)
    yearly = calculate_yearly_performance(nav)
    rolling = calculate_rolling_performance(nav, rolling_window)
    trades = pd.DataFrame(trade_records)
    turnover = pd.DataFrame(0.0, index=groups.index, columns=groups.columns)
    if not trades.empty:
        trades["turnover"] = (trades["buy_amount"] + trades["sell_amount"]) / trades["wealth"]
        recorded = trades.pivot(index="trade_date", columns="group", values="turnover")
        turnover.update(recorded)
    mean_turnover = float(turnover[strategy_name].mean())
    ic_stats = calculate_ic_statistics(rank_ic, forward_days)
    strategy = metrics.loc["Strategy"]
    excess = metrics.loc["Excess"]
    best_group = group_metrics["annual_return"].idxmax()
    best_excess_nav = groups[best_group] / benchmark
    best_excess = calculate_performance(best_excess_nav.to_frame("best")).loc["best"]
    group_cards = {
        "Best Group (in sample)": best_group,
        "Best Excess Ann.": f"{best_excess['annual_return']:.2%}",
        "Best Excess Sharpe": f"{best_excess['sharpe']:.2f}",
        "Best Excess Max DD": f"{best_excess['max_drawdown']:.2%}",
        "Best Excess Calmar": f"{best_excess['calmar']:.2f}",
        "Best Excess Win": f"{best_excess['win_rate']:.2%}",
        "IC Mean": f"{ic_stats['mean_ic']:.2%}",
        "Annualized ICIR": f"{ic_stats['annualized_icir']:.2f}",
    }
    strategy_cards = {
        "Strategy Ann.": f"{strategy['annual_return']:.2%}",
        "Strategy Sharpe": f"{strategy['sharpe']:.2f}",
        "Strategy Max DD": f"{strategy['max_drawdown']:.2%}",
        "Excess Ann.": f"{excess['annual_return']:.2%}",
        "Excess Sharpe": f"{excess['sharpe']:.2f}",
        "Excess Max DD": f"{excess['max_drawdown']:.2%}",
        "Strategy Win": f"{strategy['win_rate']:.2%}",
        "Excess Win": f"{excess['win_rate']:.2%}",
        "Daily Turnover": f"{mean_turnover:.2%}",
    }
    returns = nav.pct_change(fill_method=None).iloc[1:]
    selected_trades = (trades.loc[trades["group"] == strategy_name]
                       if not trades.empty else pd.DataFrame())
    summary = {
        "Date Range": f"{groups.index[0]} - {groups.index[-1]}",
        "Valuation Days": len(groups), "Return Days": len(returns),
        "Rolling Window": rolling_window, "Strategy Group": strategy_name,
        "Best Group (in sample)": best_group,
        "Avg Holdings at Rebalance": (
            float(selected_trades["holdings"].mean()) if not selected_trades.empty else np.nan
        ),
        "Daily Turnover (two-way)": f"{mean_turnover:.2%}",
        "Rebalance Turnover": (
            f"{selected_trades['turnover'].mean():.2%}" if not selected_trades.empty else "N/A"
        ),
        "Benchmark Ann.": f"{metrics.loc['Benchmark', 'annual_return']:.2%}",
        "Benchmark Sharpe": f"{metrics.loc['Benchmark', 'sharpe']:.2f}",
        "Return Correlation": returns["Strategy"].corr(returns["Benchmark"]),
        "Excess Volatility": f"{excess['volatility']:.2%}",
        "IC Positive": f"{(rank_ic.dropna() > 0).mean():.2%}",
    }
    report = {
        "title": title + f" | Strategy {strategy_name}", "groups": groups,
        "nav": nav, "rank_ic": rank_ic, "group_metrics": group_metrics,
        "yearly": yearly, "rolling": rolling, "window": rolling_window,
        "summary": pd.DataFrame({"Value": summary}),
        "group_cards": group_cards, "strategy_cards": strategy_cards,
        "conventions": (
            "T+1 close; no fees; RF=0; excess NAV=strategy/benchmark; "
            "Spread=compounded daily high-low return spread (research only); "
            "turnover=two-way, includes initial entry and zero-trade days"
        ),
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = (
        plot_group_report(report, output_dir / "group_report.png"),
        plot_strategy_report(report, output_dir / "strategy_report.png"),
    )
    for name, frame in {
        "group_nav": groups, "report_nav": nav, "rank_ic": rank_ic,
        "group_metrics": group_metrics, "performance": metrics,
        "yearly_metrics": yearly, "turnover": turnover,
        "rolling_sharpe": rolling["sharpe"], "rolling_win_rate": rolling["win_rate"],
        "summary": report["summary"],
    }.items():
        frame.to_csv(output_dir / f"{name}.csv")
    trades.to_csv(output_dir / "trades.csv", index=False)
    (output_dir / "conventions.txt").write_text(report["conventions"], encoding="utf-8")
    return paths


def calculate_style_correlations(
    signal_frame: pd.DataFrame,
    style_data: dict[str, pd.DataFrame],
    min_stocks: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """逐日截面 Pearson 相关性及时间均值，不混合不同日期的股票。

    每个风格按日期、股票对齐，成对剔除非有限值；至少 min_stocks 只股票且
    两侧均非恒定时才计算。比较原始/中性化结果时，调用方应传入同样本矩阵。
    返回每日相关、每日配对股票数、各风格汇总；缺失值不填充。
    """
    if (signal_frame.empty or not signal_frame.index.is_unique
            or not signal_frame.columns.is_unique or not style_data):
        raise ValueError("signals must be nonempty with unique dates/stocks; styles required")
    if isinstance(min_stocks, bool) or not isinstance(min_stocks, int) or min_stocks < 3:
        raise ValueError("min_stocks must be an integer >= 3")
    signals = signal_frame.replace([np.inf, -np.inf], np.nan)
    daily = pd.DataFrame(index=signals.index)
    counts = pd.DataFrame(index=signals.index)
    for name, values in style_data.items():
        values = values.reindex(index=signals.index, columns=signals.columns)
        values = values.replace([np.inf, -np.inf], np.nan)
        paired = signals.notna() & values.notna()
        left, right = signals.where(paired), values.where(paired)
        counts[name] = paired.sum(axis=1)
        eligible = ((counts[name] >= min_stocks)
                    & (left.std(axis=1) > 0) & (right.std(axis=1) > 0))
        daily[name] = np.nan
        daily.loc[eligible, name] = left.loc[eligible].corrwith(
            right.loc[eligible], axis=1, method="pearson"
        )
    summary = pd.DataFrame({
        "mean_correlation": daily.mean(),
        "mean_abs_correlation": daily.abs().mean(),
        "valid_days": daily.count(),
        "mean_stocks": counts.where(daily.notna()).mean(),
    })
    summary.index.name = "style"
    return daily, counts, summary


def create_style_correlation_report(
    signal_frame: pd.DataFrame,
    style_data: dict[str, pd.DataFrame],
    factor_name: str,
    output_dir: Path,
    style_groups: dict[str, str] | None = None,
) -> Path:
    """保存相关性图及 CSV；横轴按风格及组内顺序编号，未分类为 Other。"""
    daily, counts, summary = calculate_style_correlations(signal_frame, style_data)
    summary["style_group"] = [
        (style_groups or {}).get(name) or "Other" for name in summary.index
    ]
    numbers = summary.groupby("style_group", sort=False).cumcount() + 1
    summary["display_label"] = summary["style_group"] + numbers.astype(str)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    daily.to_csv(output_dir / "style_daily_correlations.csv")
    counts.to_csv(output_dir / "style_daily_counts.csv")
    summary.to_csv(output_dir / "style_correlation_summary.csv")
    return plot_style_correlations(
        summary, factor_name, output_dir / "style_correlations.png"
    )


def quote_identifier(identifier: str) -> str:
    """将动态列名安全地转换为 DuckDB 标识符。"""
    return '"' + identifier.replace('"', '""') + '"'


def save_factor_values(
    signal_frame: pd.DataFrame,
    output_db: Path,
    factor_name: str,
) -> int:
    """将非空因子值写入 factor 表，并保留表中已有的其他因子列。"""
    if factor_name in {"trade_date", "code"}:
        raise ValueError(f"factor class name conflicts with key column: {factor_name}")
    if signal_frame.empty:
        return 0

    values = signal_frame.to_numpy()
    row_positions, column_positions = np.nonzero(~pd.isna(values))
    factor_values = pd.DataFrame(
        {
            "trade_date": signal_frame.index.to_numpy()[row_positions].astype(str),
            "code": signal_frame.columns.to_numpy()[column_positions].astype(str),
            "factor_value": values[row_positions, column_positions],
        }
    )

    output_db.parent.mkdir(parents=True, exist_ok=True)
    factor_column = quote_identifier(factor_name)
    connection = duckdb.connect(str(output_db))
    try:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS factor (
                trade_date VARCHAR NOT NULL,
                code VARCHAR NOT NULL,
                PRIMARY KEY (trade_date, code)
            )
            """
        )
        existing_columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info('factor')").fetchall()
        }
        if factor_name not in existing_columns:
            connection.execute(
                f"ALTER TABLE factor ADD COLUMN {factor_column} DOUBLE"
            )

        connection.register("factor_values_to_save", factor_values)
        connection.execute("BEGIN TRANSACTION")
        try:
            # 重跑同一日期区间时先清空旧值，避免本次 NaN 位置残留历史结果。
            connection.execute(
                f"""
                UPDATE factor
                SET {factor_column} = NULL
                WHERE trade_date BETWEEN ? AND ?
                """,
                [str(signal_frame.index.min()), str(signal_frame.index.max())],
            )
            connection.execute(
                f"""
                INSERT INTO factor (trade_date, code, {factor_column})
                SELECT trade_date, code, factor_value
                FROM factor_values_to_save
                ON CONFLICT (trade_date, code) DO UPDATE
                SET {factor_column} = excluded.{factor_column}
                """
            )
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.unregister("factor_values_to_save")
    finally:
        connection.close()
    return len(factor_values)


def main() -> None:
    args = parse_args()
    if bool(args.style_db) != bool(args.style_fields):
        raise ValueError("style-db and style-fields must be specified together")
    if args.forward_days <= 0:
        raise ValueError("forward-days must be positive")

    FactorBase.validate_n_groups(args.n_groups)
    if args.rolling_window < 2:
        raise ValueError("rolling-window must be >= 2")
    strategy_group = args.n_groups if args.strategy_group is None else args.strategy_group
    if not 1 <= strategy_group <= args.n_groups:
        raise ValueError("strategy-group must be between 1 and n-groups")

    input_db = Path(args.db_path).expanduser().resolve()
    output_db = Path(args.output_db).expanduser().resolve()
    if input_db == output_db:
        raise ValueError("output-db must be different from the market database")
    if args.style_db and output_db == Path(args.style_db).expanduser().resolve():
        raise ValueError("output-db must be different from the style database")

    factor = load_factor(
        args.factor_module,
        args.factor_class,
        args.factor_params,
    )
    factor_name = f"{factor.__class__.__name__}_{factor.window}"
    minute_db = Path(args.minute_db_path).expanduser().resolve() if factor.minute_fields else None
    factor_db = Path(args.factor_db or output_db).expanduser().resolve() if factor.factor_fields else None
    if minute_db is not None and output_db == minute_db:
        raise ValueError("output-db must be different from the minute database")
    print(f"因子: {factor_name}")
    print(f"区间: {args.start_date} - {args.end_date}")
    print(f"预测周期: {args.forward_days} 个交易日")

    calculation_started = perf_counter()
    data_manager = DataManager(
        str(input_db),
        args.pool_name,
        args.start_date,
        args.end_date,
        minute_db_path=str(minute_db) if minute_db is not None else None,
        factor_db_path=str(factor_db) if factor_db is not None else None,
    )
    trade_records = []
    benchmark = None
    try:
        signal_frame, rank_ic, group_returns = factor.compute_eval(
            data_manager,
            args.forward_days,
            args.n_groups,
        )
        group_net_values = factor.calculate_group_net_values(
            signal_frame, data_manager, args.forward_days, args.n_groups,
            trade_records=trade_records if args.benchmark else None,
        )
        if args.benchmark:
            benchmark = load_benchmark(data_manager, group_net_values.index, args.benchmark)
    finally:
        data_manager.close()
    calculation_seconds = perf_counter() - calculation_started

    if args.style_db:
        with DataManager(
            str(input_db), args.pool_name, args.start_date, args.end_date,
            factor_db_path=args.style_db,
        ) as style_manager:
            style_values = style_manager.get_factor_data(
                signal_frame.index.tolist(), "1D", args.style_fields,
                signal_frame.columns.tolist(),
            )
            tables = {row[0] for row in style_manager.factor_connection.execute(
                "SHOW TABLES"
            ).fetchall()}
            style_groups = dict(style_manager.factor_connection.execute(
                'SELECT factor, style FROM factor_metadata WHERE style IS NOT NULL'
            ).fetchall()) if "factor_metadata" in tables else {}
        path = create_style_correlation_report(
            signal_frame,
            {field: style_values[field].unstack("ts_code") for field in args.style_fields},
            factor_name, args.plot_dir / factor_name,
            style_groups=style_groups,
        )
        print(f"风格相关性图: {path}")

    ic_statistics = calculate_ic_statistics(rank_ic, args.forward_days)
    if benchmark is not None:
        report_paths = create_backtest_reports(
            group_net_values, benchmark, rank_ic, trade_records, args.forward_days,
            strategy_group, args.rolling_window,
            f"{factor_name} | Benchmark {args.benchmark}", args.plot_dir / factor_name,
        )
        for path in report_paths:
            print(f"报告: {path}")
    save_started = perf_counter()
    saved_rows = save_factor_values(
        signal_frame,
        output_db,
        factor_name,
    )
    save_seconds = perf_counter() - save_started

    print("\n计算结果")
    print(f"因子形状: {signal_frame.shape}")
    print(f"RankIC形状: {rank_ic.shape}")
    print(f"{args.n_groups}分层收益形状: {group_returns.shape}")
    print(f"有效IC数量: {ic_statistics['observations']}")
    print(f"平均RankIC: {ic_statistics['mean_ic']:.8f}")
    print(f"IC标准差: {ic_statistics['ic_std']:.8f}")
    print(f"周期修正年化ICIR: {ic_statistics['annualized_icir']:.8f}")
    print(
        f"Newey-West t值: {ic_statistics['nw_t_stat']:.8f} "
        f"(lag={ic_statistics['nw_lags']}, "
        f"p={ic_statistics['nw_p_value']:.8f})"
    )
    print(f"{args.n_groups}分层平均收益:")
    for group_name, group_return in group_returns.mean().items():
        print(f"  {group_name}: {group_return:.8%}")
    print(f"每日净值形状: {group_net_values.shape}")
    print(f"写入数据库: {output_db}")
    print(f"写入非空因子值: {saved_rows}")
    print(f"计算耗时: {calculation_seconds:.3f} 秒")
    print(f"写入耗时: {save_seconds:.3f} 秒")


if __name__ == "__main__":
    main()
