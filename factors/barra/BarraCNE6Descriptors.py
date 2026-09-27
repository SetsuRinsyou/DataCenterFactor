"""CNE6 三级描述子的纯计算函数。

函数只接收已经按信号日截断并完成口径对齐的数据，不读取数据库，也不负责
截面去极值、标准化或二级因子合成。横截面输入使用 ``Series``，历史输入使用
日期为行、股票为列的 ``DataFrame``；所有函数均返回股票索引的 ``Series``。
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce")
    return (numerator / denominator.where(denominator != 0)).replace(
        [np.inf, -np.inf], np.nan
    )


def _exponential_weights(length: int, half_life: float) -> np.ndarray:
    ages = np.arange(length - 1, -1, -1, dtype=float)
    weights = np.power(0.5, ages / half_life)
    return weights / weights.sum()


def _weighted_sum(
    values: pd.DataFrame,
    window: int,
    half_life: float,
    min_observations: int,
) -> pd.Series:
    sample = values.tail(window).apply(pd.to_numeric, errors="coerce")
    weights = _exponential_weights(len(sample), half_life)
    result = {}
    for symbol in sample.columns:
        observations = sample[symbol].to_numpy(dtype=float)
        valid = np.isfinite(observations)
        result[symbol] = (
            np.dot(observations[valid], weights[valid])
            if valid.sum() >= min_observations
            else np.nan
        )
    return pd.Series(result, dtype=float)


def _weighted_regression(
    stock_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    window: int,
    half_life: float,
    min_observations: int,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    stocks = stock_returns.tail(window).apply(pd.to_numeric, errors="coerce")
    benchmark = pd.to_numeric(
        benchmark_returns.reindex(stocks.index), errors="coerce"
    )
    weights = _exponential_weights(len(stocks), half_life)
    alpha = {}
    beta = {}
    residual_sigma = {}
    explanatory = benchmark.to_numpy(dtype=float)
    for symbol in stocks.columns:
        response = stocks[symbol].to_numpy(dtype=float)
        valid = np.isfinite(response) & np.isfinite(explanatory)
        if valid.sum() < min_observations:
            alpha[symbol] = np.nan
            beta[symbol] = np.nan
            residual_sigma[symbol] = np.nan
            continue
        design = np.column_stack(
            [np.ones(valid.sum(), dtype=float), explanatory[valid]]
        )
        sqrt_weights = np.sqrt(weights[valid])
        coefficients = np.linalg.lstsq(
            design * sqrt_weights[:, None],
            response[valid] * sqrt_weights,
            rcond=None,
        )[0]
        residuals = response[valid] - design @ coefficients
        alpha[symbol] = coefficients[0]
        beta[symbol] = coefficients[1]
        residual_sigma[symbol] = np.sqrt(
            np.average(np.square(residuals), weights=weights[valid])
        )
    return (
        pd.Series(alpha, dtype=float),
        pd.Series(beta, dtype=float),
        pd.Series(residual_sigma, dtype=float),
    )


def _normalized_slope(
    annual_values: pd.DataFrame,
    negate: bool = False,
    min_observations: int = 4,
) -> pd.Series:
    values = annual_values.tail(5).apply(pd.to_numeric, errors="coerce")
    result = {}
    for symbol in values.columns:
        observations = values[symbol].to_numpy(dtype=float)
        valid = np.isfinite(observations)
        if valid.sum() < min_observations:
            result[symbol] = np.nan
            continue
        time = np.arange(len(observations), dtype=float)[valid]
        sample = observations[valid]
        average = sample.mean()
        if not np.isfinite(average) or average == 0:
            result[symbol] = np.nan
            continue
        slope = np.polyfit(time, sample, 1)[0] / average
        result[symbol] = -slope if negate else slope
    return pd.Series(result, dtype=float)


def _variability(annual_values: pd.DataFrame) -> pd.Series:
    values = annual_values.tail(5).apply(pd.to_numeric, errors="coerce")
    average = values.mean(axis=0, skipna=True)
    standard_deviation = values.std(axis=0, ddof=1, skipna=True)
    result = _safe_divide(standard_deviation, average)
    return result.where(values.notna().sum(axis=0) >= 4)


def _weighted_changes(
    history: pd.DataFrame,
    weights: Sequence[float],
) -> pd.Series:
    values = history.apply(pd.to_numeric, errors="coerce")
    weight_array = np.asarray(weights, dtype=float)
    if values.shape[0] != len(weight_array):
        raise ValueError("history rows must match weights")
    if not np.isfinite(weight_array).all() or (weight_array < 0).any():
        raise ValueError("weights must be finite and non-negative")
    weighted = values.mul(weight_array, axis=0)
    present_weight = values.notna().mul(weight_array, axis=0).sum(axis=0)
    return weighted.sum(axis=0, min_count=1) / present_weight.where(
        present_weight > 0
    )


def standardize_descriptor(
    values: pd.Series,
    market_cap: pd.Series,
) -> pd.Series:
    """按公开复现口径做 MAD 去极值和市值加权截面标准化。"""
    result = pd.to_numeric(values, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    capitalization = pd.to_numeric(market_cap, errors="coerce")
    valid = result.notna() & capitalization.gt(0)
    if valid.sum() < 2:
        return pd.Series(np.nan, index=result.index, dtype=float)

    median = result.loc[valid].median()
    mad = (result.loc[valid] - median).abs().median()
    if np.isfinite(mad) and mad > 0:
        limit = 3 * 1.4826 * mad
        result = result.clip(median - limit, median + limit)

    valid = result.notna() & capitalization.gt(0)
    center = np.average(
        result.loc[valid], weights=capitalization.loc[valid]
    )
    scale = np.sqrt(np.mean(np.square(result.loc[valid] - center)))
    if not np.isfinite(scale) or scale == 0:
        return pd.Series(np.nan, index=result.index, dtype=float)
    return ((result - center) / scale).reindex(values.index)


def combine_descriptors_equal_weight(
    descriptors: pd.DataFrame,
    market_cap: pd.Series,
) -> pd.Series:
    """各描述子标准化后等权合成，要求每个组成描述子均有效。"""
    if descriptors.empty or descriptors.shape[1] == 0:
        raise ValueError("descriptors must contain at least one column")
    standardized = pd.DataFrame(
        {
            name: standardize_descriptor(descriptors[name], market_cap)
            for name in descriptors.columns
        }
    )
    complete = standardized.notna().all(axis=1)
    composite = standardized.mean(axis=1).where(complete)
    return standardize_descriptor(composite, market_cap)


def annual_report_values(
    values: pd.DataFrame,
    report_end_dates: pd.DataFrame,
    count: int = 5,
) -> pd.DataFrame:
    """从 PIT 日快照中提取每只股票最近 ``count`` 个年度报告值。"""
    if count < 1:
        raise ValueError("count must be positive")
    extracted = {}
    for symbol in values.columns:
        frame = pd.DataFrame(
            {
                "value": pd.to_numeric(values[symbol], errors="coerce"),
                "report_end_date": report_end_dates[symbol].astype("string"),
            }
        )
        frame = frame.loc[
            frame["report_end_date"].str.fullmatch(r"\d{4}1231", na=False)
        ].dropna(subset=["value"])
        if frame.empty:
            continue
        annual = frame.groupby("report_end_date")["value"].last().tail(count)
        extracted[symbol] = annual
    if not extracted:
        return pd.DataFrame(columns=values.columns, dtype=float)
    return pd.DataFrame(extracted).sort_index().reindex(columns=values.columns)


def select_annual_report_window(
    report_data: pd.DataFrame,
    symbols: list[str],
    fields: list[str],
    count: int = 5,
) -> tuple[dict[str, pd.DataFrame], pd.Series]:
    """Align each stock's latest disclosed annual report and prior fiscal years.

    Rows are fiscal-year offsets from oldest to newest. Missing reports or
    fields stay NaN instead of shifting the stock to an older complete window.
    """
    if count < 1:
        raise ValueError("count must be positive")
    result = {
        field: pd.DataFrame(np.nan, index=range(count), columns=symbols)
        for field in fields
    }
    anchors = pd.Series(pd.NA, index=symbols, dtype="Int64")
    if report_data.empty:
        return result, anchors
    stock_reports = {
        symbol: frame.droplevel("ts_code")
        for symbol, frame in report_data.groupby(level="ts_code", sort=False)
    }
    for symbol in symbols:
        if symbol not in stock_reports:
            continue
        stock = stock_reports[symbol]
        annual_dates = stock.index.astype(str)
        stock = stock.loc[annual_dates.str.fullmatch(r"\d{4}1231")]
        if stock.empty:
            continue
        anchor = int(stock.index.max()[:4])
        anchors[symbol] = anchor
        for offset, year in enumerate(range(anchor - count + 1, anchor + 1)):
            period = f"{year}1231"
            if period not in stock.index:
                continue
            for field in fields:
                result[field].at[offset, symbol] = stock.at[period, field]
    return result, anchors


def latest_annual_value(
    values: pd.DataFrame,
    report_end_dates: pd.DataFrame,
) -> pd.Series:
    """返回每只股票截至信号日最新可得的年度报告值。"""
    annual = annual_report_values(values, report_end_dates, count=1)
    return annual.ffill().iloc[-1].reindex(values.columns)


def completed_year_end_values(
    values: pd.DataFrame,
    trade_date: str,
    count: int = 5,
) -> pd.DataFrame:
    """提取信号年前最近 ``count`` 个自然年末的日频字段值。"""
    years = pd.Index(values.index.astype(str).str[:4], name="year")
    completed = values.loc[years < trade_date[:4]].copy()
    completed.index = years[years < trade_date[:4]]
    return completed.groupby(level="year").last().tail(count)


def prepare_market_model_returns(
    stock_close: pd.DataFrame,
    benchmark_close: pd.Series,
    daily_risk_free_rate: pd.Series,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """从价格构造市场模型超额收益和个股相对市场对数收益。"""
    stock_prices = stock_close.apply(pd.to_numeric, errors="coerce")
    benchmark_prices = pd.to_numeric(
        benchmark_close.reindex(stock_prices.index), errors="coerce"
    )
    risk_free_rate = pd.to_numeric(
        daily_risk_free_rate.reindex(stock_prices.index), errors="coerce"
    )

    stock_returns = stock_prices.pct_change(fill_method=None).iloc[1:]
    benchmark_returns = benchmark_prices.pct_change(fill_method=None).iloc[1:]
    risk_free_rate = risk_free_rate.iloc[1:]
    stock_excess_returns = stock_returns.sub(risk_free_rate, axis="index")
    benchmark_excess_returns = benchmark_returns - risk_free_rate
    relative_log_returns = np.log1p(stock_returns).sub(
        np.log1p(benchmark_returns), axis="index"
    )
    return (
        stock_excess_returns.replace([np.inf, -np.inf], np.nan),
        benchmark_excess_returns.replace([np.inf, -np.inf], np.nan),
        relative_log_returns.replace([np.inf, -np.inf], np.nan),
    )


# Size


def calculate_lnsize(free_float_market_cap: pd.Series) -> pd.Series:
    """流通市值的自然对数。"""
    market_cap = pd.to_numeric(free_float_market_cap, errors="coerce")
    return np.log(market_cap.where(market_cap > 0))


def calculate_nlsize(
    size_exposure: pd.Series,
    market_cap: pd.Series,
) -> pd.Series:
    """Size 立方项对 Size 做市值加权回归后的残差。"""
    size = pd.to_numeric(size_exposure, errors="coerce")
    capitalization = pd.to_numeric(market_cap, errors="coerce")
    valid = size.notna() & capitalization.gt(0)
    result = pd.Series(np.nan, index=size.index, dtype=float)
    if valid.sum() < 2:
        return result
    design = np.column_stack(
        [np.ones(valid.sum(), dtype=float), size.loc[valid].to_numpy()]
    )
    response = size.loc[valid].pow(3).to_numpy()
    sqrt_weights = np.sqrt(capitalization.loc[valid].to_numpy())
    coefficients = np.linalg.lstsq(
        design * sqrt_weights[:, None],
        response * sqrt_weights,
        rcond=None,
    )[0]
    result.loc[valid] = response - design @ coefficients
    return result


# Volatility


def calculate_hbeta(
    stock_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
) -> pd.Series:
    """252 日、半衰期 63 日加权回归的市场 Beta。"""
    return _weighted_regression(
        stock_returns, benchmark_returns, 252, 63, 126
    )[1]


def calculate_hsigma(
    stock_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
) -> pd.Series:
    """HBETA 同一回归的残差波动率。"""
    return _weighted_regression(
        stock_returns, benchmark_returns, 252, 63, 126
    )[2]


def calculate_dastd(excess_returns: pd.DataFrame) -> pd.Series:
    """过去 252 日、半衰期 42 日的日超额收益波动率。"""
    sample = excess_returns.tail(252).apply(pd.to_numeric, errors="coerce")
    weights = _exponential_weights(len(sample), 42)
    result = {}
    for symbol in sample.columns:
        observations = sample[symbol].to_numpy(dtype=float)
        valid = np.isfinite(observations)
        if valid.sum() < 126:
            result[symbol] = np.nan
            continue
        mean = np.average(observations[valid], weights=weights[valid])
        result[symbol] = np.sqrt(
            np.average(
                np.square(observations[valid] - mean), weights=weights[valid]
            )
        )
    return pd.Series(result, dtype=float)


def calculate_cmra(excess_returns: pd.DataFrame) -> pd.Series:
    """过去 12 个 21 日月度累计对数超额收益的极差。"""
    log_returns = np.log1p(
        excess_returns.tail(252).apply(pd.to_numeric, errors="coerce")
    )
    result = {}
    for symbol in log_returns.columns:
        observations = log_returns[symbol]
        cumulative = []
        for months in range(1, 13):
            period = observations.tail(21 * months)
            cumulative.append(period.sum(min_count=int(np.ceil(len(period) * 0.8))))
        valid = pd.Series(cumulative, dtype=float).dropna()
        result[symbol] = valid.max() - valid.min() if len(valid) >= 10 else np.nan
    return pd.Series(result, dtype=float)


# Liquidity


def calculate_stom(turnover: pd.DataFrame) -> pd.Series:
    """最近 21 日换手率之和的自然对数。"""
    total = turnover.tail(21).sum(axis=0, min_count=11)
    return np.log(total.where(total > 0))


def calculate_stoq(turnover: pd.DataFrame) -> pd.Series:
    """最近三个 21 日月换手率算术平均值的自然对数。"""
    sample = turnover.tail(63)
    monthly = [sample.iloc[start:start + 21].sum(axis=0, min_count=11)
               for start in range(0, 63, 21)]
    values = pd.DataFrame(monthly)
    average = values.mean(axis=0).where(values.notna().sum(axis=0) == 3)
    return np.log(average.where(average > 0))


def calculate_stoa(turnover: pd.DataFrame) -> pd.Series:
    """最近十二个 21 日月换手率算术平均值的自然对数。"""
    sample = turnover.tail(252)
    monthly = [sample.iloc[start:start + 21].sum(axis=0, min_count=11)
               for start in range(0, 252, 21)]
    values = pd.DataFrame(monthly)
    average = values.mean(axis=0).where(values.notna().sum(axis=0) == 12)
    return np.log(average.where(average > 0))


def calculate_atvr(turnover: pd.DataFrame) -> pd.Series:
    """过去 252 日、半衰期 63 日的换手率加权和。"""
    return _weighted_sum(turnover, 252, 63, 126)


# Momentum


def calculate_strev(stock_returns: pd.DataFrame) -> pd.Series:
    """过去 21 日、半衰期 5 日的对数收益率加权和。"""
    log_returns = np.log1p(stock_returns.apply(pd.to_numeric, errors="coerce"))
    return _weighted_sum(log_returns, 21, 5, 21)


def calculate_season(lagged_month_returns: pd.DataFrame) -> pd.Series:
    """过去五年同一季节月份的已实现次月收益率均值。"""
    values = lagged_month_returns.tail(5).apply(pd.to_numeric, errors="coerce")
    return values.mean(axis=0).where(values.notna().sum(axis=0) == 5)


def calculate_indmom(
    stock_returns: pd.DataFrame,
    industry: pd.DataFrame,
    free_float_market_cap: pd.DataFrame,
) -> pd.Series:
    """计算滞后 3 日并平滑 3 日的行业动量。"""
    returns = stock_returns.apply(pd.to_numeric, errors="coerce")
    log_returns = np.log1p(returns).replace([np.inf, -np.inf], np.nan)
    if len(log_returns) < 131:
        return pd.Series(np.nan, index=returns.columns, dtype=float)

    lagged_values = []
    for lag in range(3, 6):
        end = len(log_returns) - lag
        strength = _weighted_sum(log_returns.iloc[:end], 126, 21, 63)
        exposure_date = log_returns.index[end - 1]
        daily_industry = industry.reindex(
            index=[exposure_date], columns=returns.columns
        ).iloc[0]
        capitalization = free_float_market_cap.reindex(
            index=[exposure_date], columns=returns.columns
        ).iloc[0]
        weight = np.sqrt(
            pd.to_numeric(capitalization, errors="coerce").where(
                capitalization > 0
            )
        )
        frame = pd.concat(
            [
                strength.rename("strength"),
                daily_industry.rename("industry"),
                weight.rename("weight"),
            ],
            axis=1,
        ).dropna()
        weight_sum = frame["weight"].groupby(
            frame["industry"]
        ).transform("sum")
        weighted_strength = (
            frame["strength"] * frame["weight"]
        ).groupby(frame["industry"]).transform("sum") / weight_sum
        own_contribution = (
            frame["strength"] * frame["weight"] / weight_sum
        )
        lagged_values.append(
            (weighted_strength - own_contribution).reindex(returns.columns)
        )
    return pd.DataFrame(lagged_values).mean(axis=0)


def calculate_rstr(relative_log_returns: pd.DataFrame) -> pd.Series:
    """252 日、半衰期 126 日相对强度的 11 日滞后平滑值。"""
    if len(relative_log_returns) < 273:
        return pd.Series(np.nan, index=relative_log_returns.columns, dtype=float)
    values = []
    for lag in range(11, 22):
        end = len(relative_log_returns) - lag
        values.append(
            _weighted_sum(relative_log_returns.iloc[:end], 252, 126, 126)
        )
    return pd.DataFrame(values).mean(axis=0)


def calculate_halpha(
    stock_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
) -> pd.Series:
    """HBETA 同一回归的截距项。"""
    return _weighted_regression(
        stock_returns, benchmark_returns, 252, 63, 126
    )[0]


# Quality


def calculate_mlev(
    market_equity: pd.Series,
    preferred_equity: pd.Series,
    long_term_debt: pd.Series,
) -> pd.Series:
    """市场杠杆率：(ME + PE + LD) / ME。"""
    return _safe_divide(
        market_equity + preferred_equity + long_term_debt, market_equity
    )


def calculate_blev(
    book_common_equity: pd.Series,
    preferred_equity: pd.Series,
    long_term_debt: pd.Series,
) -> pd.Series:
    """账面杠杆率：(BE + PE + LD) / BE。"""
    return _safe_divide(
        book_common_equity + preferred_equity + long_term_debt,
        book_common_equity,
    )


def calculate_dtoa(
    total_liabilities: pd.Series,
    total_assets: pd.Series,
) -> pd.Series:
    """上一财年总负债除以总资产。"""
    return _safe_divide(total_liabilities, total_assets)


def calculate_vsal(annual_sales: pd.DataFrame) -> pd.Series:
    """过去五财年营业收入标准差除以均值。"""
    return _variability(annual_sales)


def calculate_vern(annual_earnings: pd.DataFrame) -> pd.Series:
    """过去五财年净利润标准差除以均值。"""
    return _variability(annual_earnings)


def calculate_vflo(annual_cash_increase: pd.DataFrame) -> pd.Series:
    """过去五财年现金及现金等价物净增加额标准差除以均值。"""
    return _variability(annual_cash_increase)


def calculate_etopf_std(
    forecast_eps_std: pd.Series,
    current_price: pd.Series,
) -> pd.Series:
    """分析师未来 12 月 EPS 预测的截面标准差除以当前股价。"""
    return _safe_divide(forecast_eps_std, current_price)


def calculate_abs(
    current_noa: pd.Series,
    previous_noa: pd.Series,
    depreciation_and_amortization: pd.Series,
    total_assets: pd.Series,
) -> pd.Series:
    """负的资产负债表应计项目除以总资产。"""
    accrual = current_noa - previous_noa - depreciation_and_amortization
    return _safe_divide(-accrual, total_assets)


def calculate_acf(
    net_income: pd.Series,
    operating_cash_flow: pd.Series,
    investing_cash_flow: pd.Series,
    depreciation_and_amortization: pd.Series,
    total_assets: pd.Series,
) -> pd.Series:
    """负的现金流量表应计项目除以总资产。"""
    accrual = (
        net_income
        - (operating_cash_flow + investing_cash_flow)
        + depreciation_and_amortization
    )
    return _safe_divide(-accrual, total_assets)


def calculate_ato(sales_ttm: pd.Series, total_assets_mrq: pd.Series) -> pd.Series:
    """过去 12 月营业收入除以最近报告期总资产。"""
    return _safe_divide(sales_ttm, total_assets_mrq)


def calculate_gp(
    annual_sales: pd.Series,
    annual_cost_of_goods_sold: pd.Series,
    annual_total_assets: pd.Series,
) -> pd.Series:
    """上一财年毛利润除以总资产。"""
    return _safe_divide(
        annual_sales - annual_cost_of_goods_sold, annual_total_assets
    )


def calculate_gpm(
    annual_sales: pd.Series,
    annual_cost_of_goods_sold: pd.Series,
) -> pd.Series:
    """上一财年毛利润除以营业收入。"""
    return _safe_divide(annual_sales - annual_cost_of_goods_sold, annual_sales)


def calculate_roa(
    earnings_ttm: pd.Series,
    total_assets_mrq: pd.Series,
) -> pd.Series:
    """过去 12 月盈利除以最近报告期总资产。"""
    return _safe_divide(earnings_ttm, total_assets_mrq)


def calculate_agro(annual_total_assets: pd.DataFrame) -> pd.Series:
    """过去五财年总资产归一化趋势斜率的相反数。"""
    return _normalized_slope(annual_total_assets, negate=True)


def calculate_igro(annual_shares_outstanding: pd.DataFrame) -> pd.Series:
    """过去五财年流通股本归一化趋势斜率的相反数。"""
    return _normalized_slope(annual_shares_outstanding, negate=True)


def calculate_cxgro(annual_capital_expenditure: pd.DataFrame) -> pd.Series:
    """过去五财年资本支出归一化趋势斜率的相反数。"""
    return _normalized_slope(annual_capital_expenditure, negate=True)


# Value and growth


def calculate_btop(
    book_common_equity: pd.Series,
    current_market_cap: pd.Series,
) -> pd.Series:
    """最近报告期普通股账面价值除以当前总市值。"""
    return _safe_divide(book_common_equity, current_market_cap)


def calculate_etop(
    earnings_ttm: pd.Series,
    current_market_cap: pd.Series,
) -> pd.Series:
    """过去 12 月盈利除以当前总市值。"""
    return _safe_divide(earnings_ttm, current_market_cap)


def calculate_etopf(
    forecast_earnings_12m: pd.Series,
    current_market_cap: pd.Series,
) -> pd.Series:
    """分析师预测未来 12 月盈利除以当前总市值。"""
    return _safe_divide(forecast_earnings_12m, current_market_cap)


def calculate_cetop(
    cash_earnings_ttm: pd.Series,
    current_market_cap: pd.Series,
) -> pd.Series:
    """过去 12 月现金盈利除以当前总市值。"""
    return _safe_divide(cash_earnings_ttm, current_market_cap)


def calculate_em(
    previous_fiscal_year_ebit: pd.Series,
    current_enterprise_value: pd.Series,
) -> pd.Series:
    """上一财年 EBIT 除以当前企业价值。"""
    return _safe_divide(previous_fiscal_year_ebit, current_enterprise_value)


def calculate_ltrstr(relative_log_returns: pd.DataFrame) -> pd.Series:
    """长期相对强度滞后 273 日后的 11 日均值并取反。"""
    if len(relative_log_returns) < 1323:
        return pd.Series(np.nan, index=relative_log_returns.columns, dtype=float)
    values = []
    for lag in range(273, 284):
        end = len(relative_log_returns) - lag
        values.append(
            _weighted_sum(relative_log_returns.iloc[:end], 1040, 260, 520)
        )
    return -pd.DataFrame(values).mean(axis=0)


def calculate_lthalpha(
    stock_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
) -> pd.Series:
    """长期 CAPM Alpha 滞后 273 日后的 11 日均值并取反。"""
    if len(stock_returns) < 1323:
        return pd.Series(np.nan, index=stock_returns.columns, dtype=float)
    values = []
    for lag in range(273, 284):
        end = len(stock_returns) - lag
        values.append(
            _weighted_regression(
                stock_returns.iloc[:end],
                benchmark_returns.iloc[:end],
                1040,
                260,
                520,
            )[0]
        )
    return -pd.DataFrame(values).mean(axis=0)


def calculate_egrlf(long_term_earnings_growth_forecast: pd.Series) -> pd.Series:
    """分析师预测的长期（3-5 年）盈利增长率。"""
    return pd.to_numeric(long_term_earnings_growth_forecast, errors="coerce")


def calculate_egro(annual_eps: pd.DataFrame) -> pd.Series:
    """过去五财年每股收益的归一化趋势斜率。"""
    return _normalized_slope(annual_eps)


def calculate_sgro(annual_sales_per_share: pd.DataFrame) -> pd.Series:
    """过去五财年每股营业收入的归一化趋势斜率。"""
    return _normalized_slope(annual_sales_per_share)


# Analyst sentiment and dividend yield


def calculate_rr(
    period_up_revisions: pd.DataFrame,
    period_down_revisions: pd.DataFrame,
    period_total_forecasts: pd.DataFrame,
    weights: Sequence[float],
) -> pd.Series:
    """最近三个 21 日区间盈利预测上调减下调比例的加权均值。"""
    revision_ratio = (period_up_revisions - period_down_revisions).divide(
        period_total_forecasts.where(period_total_forecasts != 0)
    )
    return _weighted_changes(revision_ratio, weights)


def calculate_etopf_c(
    quarterly_forecast_ep_changes: pd.DataFrame,
    weights: Sequence[float],
) -> pd.Series:
    """最近四个季度分析师预测 EP 比相对变化的加权均值。"""
    return _weighted_changes(quarterly_forecast_ep_changes, weights)


def calculate_epsf_c(
    quarterly_forecast_eps_changes: pd.DataFrame,
    weights: Sequence[float],
) -> pd.Series:
    """最近四个季度分析师预测 EPS 相对变化的加权均值。"""
    return _weighted_changes(quarterly_forecast_eps_changes, weights)


def calculate_dtop(
    dividends_per_share_ttm: pd.Series,
    previous_month_end_price: pd.Series,
) -> pd.Series:
    """最近 12 月每股股息除以上月月末股价。"""
    return _safe_divide(dividends_per_share_ttm, previous_month_end_price)


def calculate_dtopf(
    forecast_dividend_yield_12m: pd.Series,
) -> pd.Series:
    """将 ``report_rc.rd`` 的未来 12 月预测股息率转换为小数。"""
    return pd.to_numeric(
        forecast_dividend_yield_12m, errors="coerce"
    ) / 100
