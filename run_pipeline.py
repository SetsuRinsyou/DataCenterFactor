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
from eval_plot import plot_group_cumulative_returns


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
    if args.forward_days <= 0:
        raise ValueError("forward-days must be positive")

    input_db = Path(args.db_path).expanduser().resolve()
    output_db = Path(args.output_db).expanduser().resolve()
    if input_db == output_db:
        raise ValueError("output-db must be different from the market database")

    factor = load_factor(
        args.factor_module,
        args.factor_class,
        args.factor_params,
    )
    factor_name = f"{factor.__class__.__name__}_{factor.window}"
    minute_db = Path(args.minute_db_path).expanduser().resolve() if factor.minute_fields else None
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
    )
    try:
        signal_frame, rank_ic, group_returns = factor.compute_eval(
            data_manager,
            args.forward_days,
        )
        group_net_values = factor.calculate_group_net_values(
            signal_frame, data_manager, args.forward_days
        )
    finally:
        data_manager.close()
    calculation_seconds = perf_counter() - calculation_started

    ic_statistics = calculate_ic_statistics(rank_ic, args.forward_days)
    save_started = perf_counter()
    saved_rows = save_factor_values(
        signal_frame,
        output_db,
        factor_name,
    )
    save_seconds = perf_counter() - save_started
    plot_path = plot_group_cumulative_returns(
        group_net_values,
        SCRIPT_ROOT / "tmp" / "plots" / f"{factor_name}_group_cumulative_returns.png",
    )

    print("\n计算结果")
    print(f"因子形状: {signal_frame.shape}")
    print(f"RankIC形状: {rank_ic.shape}")
    print(f"五分层收益形状: {group_returns.shape}")
    print(f"有效IC数量: {ic_statistics['observations']}")
    print(f"平均RankIC: {ic_statistics['mean_ic']:.8f}")
    print(f"IC标准差: {ic_statistics['ic_std']:.8f}")
    print(f"周期修正年化ICIR: {ic_statistics['annualized_icir']:.8f}")
    print(
        f"Newey-West t值: {ic_statistics['nw_t_stat']:.8f} "
        f"(lag={ic_statistics['nw_lags']}, "
        f"p={ic_statistics['nw_p_value']:.8f})"
    )
    print("五分层平均收益:")
    for group_name, group_return in group_returns.mean().items():
        print(f"  {group_name}: {group_return:.8%}")
    print(f"每日净值形状: {group_net_values.shape}")
    print(f"收益曲线: {plot_path}")
    print(f"写入数据库: {output_db}")
    print(f"写入非空因子值: {saved_rows}")
    print(f"计算耗时: {calculation_seconds:.3f} 秒")
    print(f"写入耗时: {save_seconds:.3f} 秒")


if __name__ == "__main__":
    main()
