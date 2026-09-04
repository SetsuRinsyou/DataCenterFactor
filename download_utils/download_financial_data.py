#!/usr/bin/env python3
"""Build the point-in-time quarterly financial input table in DuckDB."""

import argparse
import time
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd
import tushare as ts

from download_utils.download_market_data import TUSHARE_API_KEY


DATABASE_PATH = Path(__file__).resolve().parents[1] / "data" / "src_data.duckdb"


FIRST_PERIOD = "20030930"
REQUEST_INTERVAL_SECONDS = 1.0
MAX_RETRIES = 5
RF_TS_CODE = "204001.SH"

META_FIELDS = [
    "ts_code",
    "ann_date",
    "f_ann_date",
    "end_date",
    "report_type",
    "comp_type",
    "update_flag",
]
INCOME_FIELDS = [
    "revenue",
    "oper_cost",
    "biz_tax_surchg",
    "sell_exp",
    "admin_exp",
    "rd_exp",
    "n_income_attr_p",
]
BALANCE_FIELDS = [
    "total_assets",
    "total_hldr_eqy_exc_min_int",
    "money_cap",
    "trad_asset",
    "fix_assets",
    "fix_assets_total",
    "total_cur_assets",
    "total_cur_liab",
    "total_liab",
    "st_borr",
    "st_bonds_payable",
    "non_cur_liab_due_1y",
    "lt_borr",
    "bond_payable",
    "lease_liab",
    "defer_tax_liab",
]
CASHFLOW_FIELDS = [
    "depr_fa_coga_dpba",
    "amort_intang_assets",
    "lt_amort_deferred_exp",
    "use_right_asset_dep",
]
GROWTH_FIELDS = ["tr_yoy", "dt_netprofit_yoy", "ocf_yoy"]

INCOME_OUTPUTS = [f"{column}_ttm" for column in INCOME_FIELDS]
BALANCE_OUTPUTS = [f"{column}_mrq" for column in BALANCE_FIELDS]
CASHFLOW_OUTPUTS = [f"{column}_ttm" for column in CASHFLOW_FIELDS]
FINANCIAL_COLUMNS = [
    "trade_date",
    "ts_code",
    "report_end_date",
    "income_effective_date",
    "bs_effective_date",
    "cf_effective_date",
    "fi_effective_date",
    "comp_type",
    *INCOME_OUTPUTS,
    *BALANCE_OUTPUTS,
    *CASHFLOW_OUTPUTS,
    "daa_ttm",
    "depreciation_ttm_pit",
    "rf_ts_code",
    "gc001_weight",
    *GROWTH_FIELDS,
]

SOURCE_CONFIG = {
    "income": {
        "vip": "income_vip",
        "ordinary": "income",
        "fields": INCOME_FIELDS,
        "meta": META_FIELDS,
    },
    "balancesheet": {
        "vip": "balancesheet_vip",
        "ordinary": "balancesheet",
        "fields": BALANCE_FIELDS,
        "meta": META_FIELDS,
    },
    "cashflow": {
        "vip": "cashflow_vip",
        "ordinary": "cashflow",
        "fields": CASHFLOW_FIELDS,
        "meta": META_FIELDS,
    },
    "fina_indicator": {
        "vip": "fina_indicator_vip",
        "ordinary": "fina_indicator",
        "fields": ["daa", *GROWTH_FIELDS],
        "meta": ["ts_code", "ann_date", "end_date", "update_flag"],
    },
}


@dataclass
class ApiCounter:
    calls: int = 0
    retries: int = 0
    fallback_calls: int = 0
    capped_responses: int = 0


def quarter_periods(first_period: str, last_date: str) -> list[str]:
    periods = []
    for year in range(int(first_period[:4]), int(last_date[:4]) + 1):
        for suffix in ("0331", "0630", "0930", "1231"):
            period = f"{year}{suffix}"
            if first_period <= period <= last_date:
                periods.append(period)
    return periods


def previous_year_period(period: str) -> str:
    return f"{int(period[:4]) - 1}{period[4:]}"


def create_financial_table(connection: duckdb.DuckDBPyConnection, table_name: str) -> None:
    connection.execute(f'DROP TABLE IF EXISTS "{table_name}"')
    connection.execute(
        f"""
        CREATE TABLE "{table_name}" (
            trade_date VARCHAR NOT NULL,
            ts_code VARCHAR NOT NULL,
            report_end_date VARCHAR,
            income_effective_date VARCHAR,
            bs_effective_date VARCHAR,
            cf_effective_date VARCHAR,
            fi_effective_date VARCHAR,
            comp_type VARCHAR,
            revenue_ttm DOUBLE,
            oper_cost_ttm DOUBLE,
            biz_tax_surchg_ttm DOUBLE,
            sell_exp_ttm DOUBLE,
            admin_exp_ttm DOUBLE,
            rd_exp_ttm DOUBLE,
            n_income_attr_p_ttm DOUBLE,
            total_assets_mrq DOUBLE,
            total_hldr_eqy_exc_min_int_mrq DOUBLE,
            money_cap_mrq DOUBLE,
            trad_asset_mrq DOUBLE,
            fix_assets_mrq DOUBLE,
            fix_assets_total_mrq DOUBLE,
            total_cur_assets_mrq DOUBLE,
            total_cur_liab_mrq DOUBLE,
            total_liab_mrq DOUBLE,
            st_borr_mrq DOUBLE,
            st_bonds_payable_mrq DOUBLE,
            non_cur_liab_due_1y_mrq DOUBLE,
            lt_borr_mrq DOUBLE,
            bond_payable_mrq DOUBLE,
            lease_liab_mrq DOUBLE,
            defer_tax_liab_mrq DOUBLE,
            depr_fa_coga_dpba_ttm DOUBLE,
            amort_intang_assets_ttm DOUBLE,
            lt_amort_deferred_exp_ttm DOUBLE,
            use_right_asset_dep_ttm DOUBLE,
            daa_ttm DOUBLE,
            depreciation_ttm_pit DOUBLE,
            rf_ts_code VARCHAR DEFAULT '{RF_TS_CODE}',
            gc001_weight DOUBLE,
            tr_yoy DOUBLE,
            dt_netprofit_yoy DOUBLE,
            ocf_yoy DOUBLE,
            PRIMARY KEY (trade_date, ts_code)
        )
        """
    )


def financial_schema_matches(connection: duckdb.DuckDBPyConnection) -> bool:
    tables = {row[0] for row in connection.execute("SHOW TABLES").fetchall()}
    if "financial" not in tables:
        return False
    info = connection.execute("PRAGMA table_info('financial')").fetchall()
    columns = [row[1] for row in info]
    primary_key = [row[1] for row in info if row[5]]
    if columns != FINANCIAL_COLUMNS or primary_key != ["trade_date", "ts_code"]:
        raise RuntimeError(
            f"financial exists but its {len(FINANCIAL_COLUMNS)}-column schema "
            "or primary key differs; "
            "a full rebuild is required"
        )
    return True


def financial_needs_fix_assets_total(connection: duckdb.DuckDBPyConnection) -> bool:
    """识别仅缺少新增固定资产合计列的旧版 financial 表。"""
    tables = {row[0] for row in connection.execute("SHOW TABLES").fetchall()}
    if "financial" not in tables:
        return False
    info = connection.execute("PRAGMA table_info('financial')").fetchall()
    columns = [row[1] for row in info]
    primary_key = [row[1] for row in info if row[5]]
    legacy_columns = [
        column for column in FINANCIAL_COLUMNS if column != "fix_assets_total_mrq"
    ]
    return columns == legacy_columns and primary_key == ["trade_date", "ts_code"]


def financial_needs_growth_fields(connection: duckdb.DuckDBPyConnection) -> bool:
    """识别仅缺少三个财务指标同比字段的旧版 financial 表。"""
    tables = {row[0] for row in connection.execute("SHOW TABLES").fetchall()}
    if "financial" not in tables:
        return False
    info = connection.execute("PRAGMA table_info('financial')").fetchall()
    columns = [row[1] for row in info]
    primary_key = [row[1] for row in info if row[5]]
    legacy_columns = [
        column for column in FINANCIAL_COLUMNS if column not in GROWTH_FIELDS
    ]
    return columns == legacy_columns and primary_key == ["trade_date", "ts_code"]


def financial_needs_depreciation_ttm_pit(
    connection: duckdb.DuckDBPyConnection,
) -> bool:
    """识别仅缺少最近有效折旧摊销 TTM 列的旧版 financial 表。"""
    tables = {row[0] for row in connection.execute("SHOW TABLES").fetchall()}
    if "financial" not in tables:
        return False
    info = connection.execute("PRAGMA table_info('financial')").fetchall()
    columns = [row[1] for row in info]
    primary_key = [row[1] for row in info if row[5]]
    legacy_columns = [
        column for column in FINANCIAL_COLUMNS if column != "depreciation_ttm_pit"
    ]
    return columns == legacy_columns and primary_key == ["trade_date", "ts_code"]


def call_api(pro, endpoint: str, counters: dict[str, ApiCounter], **kwargs) -> pd.DataFrame:
    counter = counters[endpoint]
    method = getattr(pro, endpoint)
    for retry in range(MAX_RETRIES + 1):
        counter.calls += 1
        try:
            frame = method(**kwargs)
            time.sleep(REQUEST_INTERVAL_SECONDS)
            if frame is None:
                return pd.DataFrame()
            if len(frame) >= 6000 and endpoint.endswith("_vip"):
                counter.capped_responses += 1
            if len(frame) == 100 and not endpoint.endswith("_vip") and endpoint != "repo_daily":
                counter.capped_responses += 1
            return frame
        except Exception as exc:
            if retry == MAX_RETRIES:
                raise RuntimeError(
                    f"{endpoint} failed after {MAX_RETRIES} retries; parameters={kwargs}"
                ) from exc
            counter.retries += 1
            time.sleep(2 ** (retry + 1))
    raise AssertionError("unreachable")


def normalize_source(
    frame: pd.DataFrame,
    source: str,
    periods: set[str],
    universe: set[str],
) -> pd.DataFrame:
    config = SOURCE_CONFIG[source]
    required = [*config["meta"], *config["fields"]]
    if frame.empty:
        return pd.DataFrame(columns=["ts_code", "end_date", "effective_date", "comp_type", *config["fields"]])
    missing = set(required) - set(frame.columns)
    if missing:
        raise RuntimeError(f"{source} response is missing fields: {sorted(missing)}")

    data = frame[required].copy()
    data["ts_code"] = data["ts_code"].astype(str)
    data["end_date"] = data["end_date"].astype(str)
    data = data[data["ts_code"].isin(universe) & data["end_date"].isin(periods)].copy()
    if source != "fina_indicator":
        data = data[data["report_type"].astype(str) == "1"].copy()
        data["effective_date"] = data["f_ann_date"].where(
            data["f_ann_date"].notna() & (data["f_ann_date"].astype(str) != ""),
            data["ann_date"],
        )
    else:
        data["effective_date"] = data["ann_date"]
        data["comp_type"] = None

    data = data[data["effective_date"].notna()].copy()
    data["effective_date"] = data["effective_date"].astype(str)
    data = data[data["effective_date"].str.fullmatch(r"\d{8}", na=False)].copy()
    data["_update_priority"] = (
        pd.to_numeric(data["update_flag"], errors="coerce").fillna(0).eq(1).astype(int)
    )
    data["_sequence"] = range(len(data))
    data = data.sort_values(
        ["ts_code", "end_date", "effective_date", "_update_priority", "_sequence"]
    ).drop_duplicates(["ts_code", "end_date", "effective_date"], keep="last")
    for column in config["fields"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return data[["ts_code", "end_date", "effective_date", "comp_type", *config["fields"]]].reset_index(drop=True)


def concat_source_frames(frames: list[pd.DataFrame], columns: list[str]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(columns=columns)
    prepared = [frame.dropna(axis=1, how="all") for frame in frames]
    return pd.concat(prepared, ignore_index=True).reindex(columns=columns)


def interior_gap_codes(frame: pd.DataFrame, periods: list[str]) -> list[str]:
    if frame.empty:
        return []
    position = {period: index for index, period in enumerate(periods)}
    gaps = []
    for ts_code, group in frame.groupby("ts_code", sort=False):
        observed = sorted({position[period] for period in group["end_date"] if period in position})
        if len(observed) < 2:
            continue
        if any(index not in observed for index in range(observed[0] + 1, observed[-1])):
            gaps.append(ts_code)
    return sorted(gaps)


def fetch_source(
    pro,
    source: str,
    periods: list[str],
    universe: set[str],
    market_max_date: str,
    counters: dict[str, ApiCounter],
) -> pd.DataFrame:
    config = SOURCE_CONFIG[source]
    fields = ",".join([*config["meta"], *config["fields"]])
    frames = []
    for index, period in enumerate(periods, start=1):
        kwargs = {"period": period, "fields": fields}
        if source != "fina_indicator":
            kwargs["report_type"] = "1"
        frame = call_api(pro, config["vip"], counters, **kwargs)
        if not frame.empty:
            frame = frame[frame["ts_code"].astype(str).isin(universe)].copy()
            frames.append(frame)
        print(f"[{source} {index}/{len(periods)}] {period}: {len(frame):,} target rows")

    raw = concat_source_frames(frames, fields.split(","))
    normalized = normalize_source(raw, source, set(periods), universe)
    gap_codes = interior_gap_codes(normalized, periods)
    if gap_codes:
        print(f"{source}: {len(gap_codes):,} stocks have interior quarter gaps; using one fallback call per stock")
    fallback_frames = []
    for index, ts_code in enumerate(gap_codes, start=1):
        kwargs = {"ts_code": ts_code, "fields": fields}
        if source == "fina_indicator":
            kwargs.update(start_date=periods[0], end_date=periods[-1])
        else:
            kwargs.update(
                start_date=f"{int(periods[0][:4]) - 1}0101",
                end_date=market_max_date,
                report_type="1",
            )
        counters[config["ordinary"]].fallback_calls += 1
        fallback = call_api(pro, config["ordinary"], counters, **kwargs)
        if not fallback.empty:
            fallback_frames.append(fallback)
        print(f"[{source} fallback {index}/{len(gap_codes)}] {ts_code}: {len(fallback):,} rows")

    if fallback_frames:
        combined = concat_source_frames([raw, *fallback_frames], fields.split(","))
        normalized = normalize_source(combined, source, set(periods), universe)
    return normalized


def latest_record(records: list[dict], effective_date: str) -> dict | None:
    dates = [record["effective_date"] for record in records]
    index = bisect_right(dates, effective_date) - 1
    return records[index] if index >= 0 else None


def build_ttm_events(
    frame: pd.DataFrame,
    value_fields: list[str],
    output_fields: list[str],
    calculate_periods: set[str],
) -> pd.DataFrame:
    output_columns = ["ts_code", "report_end_date", "effective_date", "comp_type", *output_fields]
    if frame.empty:
        return pd.DataFrame(columns=output_columns)

    record_index = {}
    for key, group in frame.sort_values("effective_date").groupby(["ts_code", "end_date"], sort=False):
        record_index[key] = group.to_dict("records")

    events = []
    for (ts_code, period), current_records in record_index.items():
        if period not in calculate_periods:
            continue
        if period.endswith("1231"):
            dependency_records = []
        else:
            dependency_records = [
                record_index.get((ts_code, f"{int(period[:4]) - 1}1231"), []),
                record_index.get((ts_code, previous_year_period(period)), []),
            ]
        first_current_date = current_records[0]["effective_date"]
        candidate_dates = sorted(
            {
                record["effective_date"]
                for records in [current_records, *dependency_records]
                for record in records
                if record["effective_date"] >= first_current_date
            }
        )
        for effective_date in candidate_dates:
            current = latest_record(current_records, effective_date)
            dependencies = [latest_record(records, effective_date) for records in dependency_records]
            event = {
                "ts_code": ts_code,
                "report_end_date": period,
                "effective_date": effective_date,
                "comp_type": current.get("comp_type") if current else None,
            }
            for source_field, output_field in zip(value_fields, output_fields):
                current_value = current.get(source_field) if current else None
                if period.endswith("1231"):
                    value = current_value
                else:
                    previous_annual = dependencies[0].get(source_field) if dependencies[0] else None
                    previous_same = dependencies[1].get(source_field) if dependencies[1] else None
                    if any(pd.isna(item) for item in (current_value, previous_annual, previous_same)):
                        value = None
                    else:
                        value = current_value + previous_annual - previous_same
                event[output_field] = value
            events.append(event)
    return pd.DataFrame(events, columns=output_columns).sort_values(
        ["ts_code", "report_end_date", "effective_date"], ignore_index=True
    )


def build_growth_events(
    frame: pd.DataFrame, calculate_periods: set[str]
) -> pd.DataFrame:
    columns = ["ts_code", "report_end_date", "effective_date", *GROWTH_FIELDS]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    data = frame[frame["end_date"].isin(calculate_periods)].copy()
    data = data.rename(columns={"end_date": "report_end_date"})
    return data[columns].sort_values(
        ["ts_code", "report_end_date", "effective_date"], ignore_index=True
    )


def build_indicator_events(
    frame: pd.DataFrame, calculate_periods: set[str]
) -> pd.DataFrame:
    columns = [
        "ts_code",
        "report_end_date",
        "effective_date",
        "comp_type",
        "daa_ttm",
        *GROWTH_FIELDS,
    ]
    daa_events = build_ttm_events(
        frame, ["daa"], ["daa_ttm"], calculate_periods
    )
    growth_events = build_growth_events(frame, calculate_periods)
    if daa_events.empty and growth_events.empty:
        return pd.DataFrame(columns=columns)

    event_index = {}
    for name, events in (("daa", daa_events), ("growth", growth_events)):
        for key, group in events.groupby(
            ["ts_code", "report_end_date"], sort=False
        ):
            event_index.setdefault(key, {})[name] = group.sort_values(
                "effective_date"
            ).to_dict("records")

    combined = []
    for (ts_code, report_end_date), sources in event_index.items():
        daa_records = sources.get("daa", [])
        growth_records = sources.get("growth", [])
        effective_dates = sorted(
            {
                record["effective_date"]
                for records in (daa_records, growth_records)
                for record in records
            }
        )
        for effective_date in effective_dates:
            daa = latest_record(daa_records, effective_date)
            growth = latest_record(growth_records, effective_date)
            combined.append(
                {
                    "ts_code": ts_code,
                    "report_end_date": report_end_date,
                    "effective_date": effective_date,
                    "comp_type": daa.get("comp_type") if daa else None,
                    "daa_ttm": daa.get("daa_ttm") if daa else None,
                    **{
                        field: growth.get(field) if growth else None
                        for field in GROWTH_FIELDS
                    },
                }
            )
    return pd.DataFrame(combined, columns=columns).sort_values(
        ["ts_code", "report_end_date", "effective_date"], ignore_index=True
    )


def build_balance_events(frame: pd.DataFrame, calculate_periods: set[str]) -> pd.DataFrame:
    columns = ["ts_code", "report_end_date", "effective_date", "comp_type", *BALANCE_OUTPUTS]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    data = frame[frame["end_date"].isin(calculate_periods)].copy()
    data = data.rename(
        columns={"end_date": "report_end_date", **dict(zip(BALANCE_FIELDS, BALANCE_OUTPUTS))}
    )
    return data[columns].sort_values(
        ["ts_code", "report_end_date", "effective_date"], ignore_index=True
    )


def build_depreciation_events(
    cashflow: pd.DataFrame,
    indicator: pd.DataFrame,
) -> pd.DataFrame:
    """合并折旧摊销来源，仅保留能够更新最近有效值的公告事件。"""
    frames = []
    if not cashflow.empty:
        cashflow_values = cashflow[CASHFLOW_OUTPUTS].sum(axis=1, min_count=1)
        cashflow_events = cashflow[
            ["ts_code", "report_end_date", "effective_date"]
        ].copy()
        cashflow_events["depreciation_ttm_pit"] = cashflow_values
        cashflow_events["_source_priority"] = 0
        frames.append(cashflow_events)
    if not indicator.empty:
        indicator_events = indicator[
            ["ts_code", "report_end_date", "effective_date", "daa_ttm"]
        ].rename(columns={"daa_ttm": "depreciation_ttm_pit"})
        indicator_events = indicator_events.copy()
        indicator_events["_source_priority"] = 1
        frames.append(indicator_events)
    if not frames:
        return pd.DataFrame(
            columns=["ts_code", "effective_date", "depreciation_ttm_pit"]
        )

    events = pd.concat(frames, ignore_index=True)
    events = events[events["depreciation_ttm_pit"].notna()].copy()
    events = events.sort_values(
        [
            "ts_code",
            "effective_date",
            "report_end_date",
            "_source_priority",
        ]
    ).drop_duplicates(["ts_code", "effective_date"], keep="last")
    return events[
        ["ts_code", "effective_date", "depreciation_ttm_pit"]
    ].reset_index(drop=True)


def build_activation_events(
    income: pd.DataFrame,
    balance: pd.DataFrame,
    calculate_periods: set[str],
) -> pd.DataFrame:
    frames = []
    for frame in (income, balance):
        if not frame.empty:
            part = frame[frame["end_date"].isin(calculate_periods)][
                ["ts_code", "end_date", "effective_date"]
            ].copy()
            frames.append(part)
    if not frames:
        return pd.DataFrame(columns=["ts_code", "report_end_date", "effective_date"])
    events = pd.concat(frames, ignore_index=True)
    events = events.groupby(["ts_code", "end_date"], as_index=False)["effective_date"].min()
    return events.rename(columns={"end_date": "report_end_date"}).sort_values(
        ["ts_code", "effective_date", "report_end_date"], ignore_index=True
    )


def repo_ranges(start_date: str, end_date: str) -> list[tuple[str, str]]:
    ranges = []
    start_year = int(start_date[:4])
    end_year = int(end_date[:4])
    for year in range(start_year, end_year + 1, 5):
        ranges.append((max(start_date, f"{year}0101"), min(end_date, f"{year + 4}1231")))
    return ranges


def fetch_repo_daily(
    pro,
    start_date: str,
    end_date: str,
    counters: dict[str, ApiCounter],
) -> pd.DataFrame:
    frames = []
    for period_start, period_end in repo_ranges(start_date, end_date):
        frame = call_api(
            pro,
            "repo_daily",
            counters,
            ts_code=RF_TS_CODE,
            start_date=period_start,
            end_date=period_end,
            fields="ts_code,trade_date,weight",
        )
        if len(frame) >= 2000:
            raise RuntimeError(
                f"repo_daily returned {len(frame):,} rows for {period_start}-{period_end}; "
                "the response may have reached its 2,000-row limit"
            )
        frames.append(frame)
        print(f"[repo_daily] {period_start}-{period_end}: {len(frame):,} rows")
    if not frames:
        return pd.DataFrame(columns=["trade_date", "gc001_weight"])
    data = pd.concat(frames, ignore_index=True)
    missing = {"ts_code", "trade_date", "weight"} - set(data.columns)
    if missing:
        raise RuntimeError(f"repo_daily response is missing fields: {sorted(missing)}")
    data = data[data["ts_code"].astype(str) == RF_TS_CODE].copy()
    data["trade_date"] = data["trade_date"].astype(str)
    data["gc001_weight"] = pd.to_numeric(data["weight"], errors="coerce")
    return data[["trade_date", "gc001_weight"]].drop_duplicates("trade_date", keep="last")


def register_events(
    connection: duckdb.DuckDBPyConnection,
    activation: pd.DataFrame,
    income: pd.DataFrame,
    balance: pd.DataFrame,
    cashflow: pd.DataFrame,
    indicator: pd.DataFrame,
    depreciation: pd.DataFrame,
    repo: pd.DataFrame,
) -> None:
    for name, frame in {
        "activation_events": activation,
        "income_events": income,
        "balance_events": balance,
        "cashflow_events": cashflow,
        "indicator_events": indicator,
        "depreciation_events": depreciation,
        "repo_events": repo,
    }.items():
        connection.register(name, frame)


def unregister_events(connection: duckdb.DuckDBPyConnection) -> None:
    for name in (
        "activation_events",
        "income_events",
        "balance_events",
        "cashflow_events",
        "indicator_events",
        "depreciation_events",
        "repo_events",
    ):
        connection.unregister(name)


def event_select_sql(incremental: bool) -> str:
    new_values = {
        "report_end_date": "a.report_end_date",
        "income_effective_date": "i.effective_date",
        "bs_effective_date": "b.effective_date",
        "cf_effective_date": "c.effective_date",
        "fi_effective_date": "f.effective_date",
        "comp_type": "COALESCE(i.comp_type, b.comp_type)",
        **{column: f"i.{column}" for column in INCOME_OUTPUTS},
        **{column: f"b.{column}" for column in BALANCE_OUTPUTS},
        **{column: f"c.{column}" for column in CASHFLOW_OUTPUTS},
        "daa_ttm": "f.daa_ttm",
        "depreciation_ttm_pit": "d.depreciation_ttm_pit",
        **{column: f"f.{column}" for column in GROWTH_FIELDS},
    }
    expressions = ["m.trade_date", "m.ts_code"]
    for column in FINANCIAL_COLUMNS[2:]:
        if column == "rf_ts_code":
            expression = f"'{RF_TS_CODE}'"
        elif column == "gc001_weight":
            if incremental:
                expression = (
                    "CASE WHEN r.gc001_weight IS NOT NULL THEN r.gc001_weight "
                    "WHEN old.trade_date = m.trade_date THEN old.gc001_weight END"
                )
            else:
                expression = "r.gc001_weight"
        else:
            expression = new_values[column]
            if incremental and column == "depreciation_ttm_pit":
                expression = f"COALESCE({expression}, old.{column})"
            elif incremental:
                expression = (
                    f"CASE WHEN a.report_end_date IS NULL THEN old.{column} "
                    f"ELSE {expression} END"
                )
        expressions.append(f"{expression} AS {column}")

    old_join = """
        ASOF LEFT JOIN financial AS old
            ON m.ts_code = old.ts_code AND m.trade_date >= old.trade_date
    """ if incremental else ""
    return f"""
        SELECT {', '.join(expressions)}
        FROM market AS m
        {old_join}
        ASOF LEFT JOIN activation_events AS a
            ON m.ts_code = a.ts_code AND m.trade_date > a.effective_date
        ASOF LEFT JOIN income_events AS i
            ON m.ts_code = i.ts_code
           AND a.report_end_date = i.report_end_date
           AND m.trade_date > i.effective_date
        ASOF LEFT JOIN balance_events AS b
            ON m.ts_code = b.ts_code
           AND a.report_end_date = b.report_end_date
           AND m.trade_date > b.effective_date
        ASOF LEFT JOIN cashflow_events AS c
            ON m.ts_code = c.ts_code
           AND a.report_end_date = c.report_end_date
           AND m.trade_date > c.effective_date
        ASOF LEFT JOIN indicator_events AS f
            ON m.ts_code = f.ts_code
           AND a.report_end_date = f.report_end_date
           AND m.trade_date > f.effective_date
        ASOF LEFT JOIN depreciation_events AS d
            ON m.ts_code = d.ts_code
           AND m.trade_date > d.effective_date
        LEFT JOIN repo_events AS r ON m.trade_date = r.trade_date
    """


def validate_key_state(connection: duckdb.DuckDBPyConnection) -> str:
    financial_max = connection.execute("SELECT MAX(trade_date) FROM financial").fetchone()[0]
    if financial_max is None:
        raise RuntimeError("financial exists but is empty; a full rebuild is required")
    extra = connection.execute(
        """
        SELECT COUNT(*) FROM financial AS f
        ANTI JOIN market AS m USING (trade_date, ts_code)
        """
    ).fetchone()[0]
    missing_historical = connection.execute(
        """
        SELECT COUNT(*) FROM market AS m
        ANTI JOIN financial AS f USING (trade_date, ts_code)
        WHERE m.trade_date <= ?
        """,
        [financial_max],
    ).fetchone()[0]
    if extra or missing_historical:
        raise RuntimeError(
            "financial and market have historical key differences "
            f"(extra={extra:,}, missing_through_{financial_max}={missing_historical:,}); "
            "a full rebuild is required"
        )
    return financial_max


def build_table(
    connection: duckdb.DuckDBPyConnection,
    incremental: bool,
    cutoff_effective_date: str | None,
) -> None:
    create_financial_table(connection, "financial__build")
    query = event_select_sql(incremental)
    if incremental and cutoff_effective_date:
        connection.execute(
            "INSERT INTO financial__build SELECT * FROM financial WHERE trade_date <= ?",
            [cutoff_effective_date],
        )
        query += " WHERE m.trade_date > ?"
        connection.execute(f"INSERT INTO financial__build {query}", [cutoff_effective_date])
    else:
        connection.execute(f"INSERT INTO financial__build {query}")


def validate_table(connection: duckdb.DuckDBPyConnection, table_name: str) -> None:
    info = connection.execute(f"PRAGMA table_info('{table_name}')").fetchall()
    columns = [row[1] for row in info]
    primary_key = [row[1] for row in info if row[5]]
    if columns != FINANCIAL_COLUMNS or primary_key != ["trade_date", "ts_code"]:
        raise RuntimeError(
            f"{table_name} does not have the expected "
            f"{len(FINANCIAL_COLUMNS)}-column schema and key"
        )

    market_count = connection.execute("SELECT COUNT(*) FROM market").fetchone()[0]
    table_count, first_date, last_date = connection.execute(
        f"SELECT COUNT(*), MIN(trade_date), MAX(trade_date) FROM {table_name}"
    ).fetchone()
    missing = connection.execute(
        f"SELECT COUNT(*) FROM market ANTI JOIN {table_name} USING (trade_date, ts_code)"
    ).fetchone()[0]
    extra = connection.execute(
        f"SELECT COUNT(*) FROM {table_name} ANTI JOIN market USING (trade_date, ts_code)"
    ).fetchone()[0]
    if table_count != market_count or missing or extra:
        raise RuntimeError(
            f"{table_name} keys differ from market: rows={table_count:,}/{market_count:,}, "
            f"missing={missing:,}, extra={extra:,}"
        )

    leak_conditions = [
        "(income_effective_date IS NOT NULL AND trade_date <= income_effective_date)",
        "(bs_effective_date IS NOT NULL AND trade_date <= bs_effective_date)",
        "(cf_effective_date IS NOT NULL AND trade_date <= cf_effective_date)",
        "(fi_effective_date IS NOT NULL AND trade_date <= fi_effective_date)",
    ]
    leaks = connection.execute(
        f"SELECT COUNT(*) FROM {table_name} WHERE {' OR '.join(leak_conditions)}"
    ).fetchone()[0]
    if leaks:
        raise RuntimeError(f"{table_name} contains {leaks:,} rows that use data too early")
    growth_without_date = connection.execute(
        f"""
        SELECT COUNT(*) FROM {table_name}
        WHERE ({' OR '.join(f'{column} IS NOT NULL' for column in GROWTH_FIELDS)})
          AND fi_effective_date IS NULL
        """
    ).fetchone()[0]
    if growth_without_date:
        raise RuntimeError(
            f"{table_name} contains {growth_without_date:,} growth rows without "
            "a financial-indicator effective date"
        )
    print(
        f"Validated {table_name}: {table_count:,} rows, "
        f"{len(FINANCIAL_COLUMNS)} columns, "
        f"range {first_date}-{last_date}, exact market key match, no disclosure-date leakage"
    )


def add_growth_fields(
    connection: duckdb.DuckDBPyConnection,
    growth_events: pd.DataFrame,
) -> None:
    """保留现有财务列，仅按公告日向后增加三个同比指标。"""
    connection.register("growth_events", growth_events)
    try:
        connection.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE financial_growth_daily AS
            SELECT
                f.trade_date,
                f.ts_code,
                g.effective_date AS growth_effective_date,
                {', '.join(f'g.{column}' for column in GROWTH_FIELDS)}
            FROM financial AS f
            ASOF LEFT JOIN growth_events AS g
                ON f.ts_code = g.ts_code
               AND f.report_end_date = g.report_end_date
               AND f.trade_date > g.effective_date
            WHERE g.effective_date IS NOT NULL
            """
        )
    finally:
        connection.unregister("growth_events")

    connection.execute("BEGIN TRANSACTION")
    try:
        for column in GROWTH_FIELDS:
            connection.execute(
                f"ALTER TABLE financial ADD COLUMN {column} DOUBLE"
            )
        connection.execute(
            f"""
            UPDATE financial AS f
            SET
                fi_effective_date = CASE
                    WHEN f.fi_effective_date IS NULL
                      OR d.growth_effective_date > f.fi_effective_date
                    THEN d.growth_effective_date
                    ELSE f.fi_effective_date
                END,
                {', '.join(f'{column} = d.{column}' for column in GROWTH_FIELDS)}
            FROM financial_growth_daily AS d
            WHERE f.trade_date = d.trade_date
              AND f.ts_code = d.ts_code
            """
        )
        validate_table(connection, "financial")
        connection.execute("COMMIT")
    except Exception:
        connection.execute("ROLLBACK")
        raise
    connection.execute("CHECKPOINT")


def add_fix_assets_total_column(
    connection: duckdb.DuckDBPyConnection,
    balance_events: pd.DataFrame,
) -> None:
    """保留旧表全部数据，仅增加按公告日生效的固定资产合计列。"""
    events = balance_events[
        ["ts_code", "report_end_date", "effective_date", "fix_assets_total_mrq"]
    ]
    connection.register("fix_assets_total_events", events)
    try:
        connection.execute(
            """
            CREATE OR REPLACE TEMP TABLE fix_assets_total_daily AS
            SELECT
                f.trade_date,
                f.ts_code,
                b.fix_assets_total_mrq
            FROM financial AS f
            ASOF LEFT JOIN fix_assets_total_events AS b
                ON f.ts_code = b.ts_code
               AND f.report_end_date = b.report_end_date
               AND f.trade_date > b.effective_date
            """
        )
    finally:
        connection.unregister("fix_assets_total_events")

    create_financial_table(connection, "financial__build")
    select_columns = [
        "d.fix_assets_total_mrq"
        if column == "fix_assets_total_mrq"
        else f'f."{column}"'
        for column in FINANCIAL_COLUMNS
    ]
    connection.execute(
        f"""
        INSERT INTO financial__build
        SELECT {', '.join(select_columns)}
        FROM financial AS f
        LEFT JOIN fix_assets_total_daily AS d
          USING (trade_date, ts_code)
        """
    )
    validate_table(connection, "financial__build")

    connection.execute("BEGIN TRANSACTION")
    try:
        connection.execute("DROP TABLE financial")
        connection.execute("ALTER TABLE financial__build RENAME TO financial")
        connection.execute("COMMIT")
    except Exception:
        connection.execute("ROLLBACK")
        raise
    connection.execute("CHECKPOINT")


def add_depreciation_ttm_pit_column(
    connection: duckdb.DuckDBPyConnection,
    depreciation_events: pd.DataFrame,
) -> None:
    """保留原始财务字段，仅增加截至当日最近已披露的有效折旧摊销。"""
    connection.register("depreciation_events", depreciation_events)
    try:
        connection.execute(
            """
            CREATE OR REPLACE TEMP TABLE depreciation_ttm_pit_daily AS
            SELECT
                f.trade_date,
                f.ts_code,
                d.depreciation_ttm_pit
            FROM financial AS f
            ASOF LEFT JOIN depreciation_events AS d
                ON f.ts_code = d.ts_code
               AND f.trade_date > d.effective_date
            """
        )
    finally:
        connection.unregister("depreciation_events")

    create_financial_table(connection, "financial__build")
    select_columns = [
        "d.depreciation_ttm_pit"
        if column == "depreciation_ttm_pit"
        else f'f."{column}"'
        for column in FINANCIAL_COLUMNS
    ]
    connection.execute(
        f"""
        INSERT INTO financial__build
        SELECT {', '.join(select_columns)}
        FROM financial AS f
        LEFT JOIN depreciation_ttm_pit_daily AS d
          USING (trade_date, ts_code)
        """
    )
    validate_table(connection, "financial__build")

    connection.execute("BEGIN TRANSACTION")
    try:
        connection.execute("DROP TABLE financial")
        connection.execute("ALTER TABLE financial__build RENAME TO financial")
        connection.execute("COMMIT")
    except Exception:
        connection.execute("ROLLBACK")
        raise
    connection.execute("CHECKPOINT")


def report_coverage(connection: duckdb.DuckDBPyConnection) -> None:
    value_columns = FINANCIAL_COLUMNS[2:]
    aggregates = []
    for column in value_columns:
        aggregates.extend(
            [
                f"MIN(trade_date) FILTER (WHERE {column} IS NOT NULL)",
                f"COUNT({column})",
            ]
        )
    row = connection.execute(f"SELECT COUNT(*), {', '.join(aggregates)} FROM financial").fetchone()
    total = row[0]
    print("Column coverage:")
    for index, column in enumerate(value_columns):
        first_date = row[1 + index * 2]
        count = row[2 + index * 2]
        missing_rate = (total - count) / total if total else 0.0
        print(
            f"  {column}: first={first_date or 'NULL'}, non_null={count:,}, "
            f"missing={missing_rate:.2%}"
        )

    early_columns = [*CASHFLOW_OUTPUTS, "daa_ttm"]
    early = connection.execute(
        "SELECT " + ", ".join(f"COUNT({column})" for column in early_columns) +
        " FROM financial WHERE trade_date < '20100101'"
    ).fetchone()
    print("Early cash-flow coverage (trade_date < 20100101):")
    for column, count in zip(early_columns, early):
        print(f"  {column}: {count:,} non-null rows")


def run_self_tests() -> None:
    rows = []
    values = {
        "20230331": 8.0,
        "20230630": 35.0,
        "20230930": 72.0,
        "20231231": 100.0,
        "20240331": 10.0,
        "20240630": 40.0,
        "20240930": 80.0,
        "20241231": 110.0,
    }
    for period, value in values.items():
        rows.append(
            {
                "ts_code": "TEST.SH",
                "end_date": period,
                "effective_date": f"{int(period[:4])}{'0430' if period.endswith('0331') else '0831' if period.endswith('0630') else '1031' if period.endswith('0930') else '0430'}",
                "comp_type": "1",
                "value": value,
            }
        )
    frame = pd.DataFrame(rows)
    events = build_ttm_events(
        frame,
        ["value"],
        ["value_ttm"],
        {"20240331", "20240630", "20240930", "20241231"},
    )
    actual = events.groupby("report_end_date").tail(1).set_index("report_end_date")["value_ttm"].to_dict()
    assert actual == {"20240331": 102.0, "20240630": 105.0, "20240930": 108.0, "20241231": 110.0}

    indicator_frame = frame.rename(columns={"value": "daa"}).copy()
    indicator_frame["tr_yoy"] = indicator_frame["daa"]
    indicator_frame["dt_netprofit_yoy"] = indicator_frame["daa"] + 1.0
    indicator_frame["ocf_yoy"] = indicator_frame["daa"] + 2.0
    indicator_events = build_indicator_events(
        indicator_frame,
        {"20240331", "20240630", "20240930", "20241231"},
    )
    latest_indicator = indicator_events.groupby("report_end_date").tail(1).set_index(
        "report_end_date"
    )
    assert latest_indicator.loc["20240331", "daa_ttm"] == 102.0
    assert latest_indicator.loc["20240331", "tr_yoy"] == 10.0
    assert latest_indicator.loc["20240331", "dt_netprofit_yoy"] == 11.0
    assert latest_indicator.loc["20240331", "ocf_yoy"] == 12.0

    missing = frame[frame["end_date"] != "20230331"]
    missing_events = build_ttm_events(missing, ["value"], ["value_ttm"], {"20240331"})
    assert missing_events["value_ttm"].isna().all()
    assert previous_year_period("20240630") == "20230630"

    cashflow_events = pd.DataFrame(
        [
            {
                "ts_code": "TEST.SH",
                "report_end_date": "20231231",
                "effective_date": "20240430",
                **{column: 1.0 for column in CASHFLOW_OUTPUTS},
            },
            {
                "ts_code": "TEST.SH",
                "report_end_date": "20240331",
                "effective_date": "20240501",
                **{column: None for column in CASHFLOW_OUTPUTS},
            },
        ]
    )
    indicator_events = pd.DataFrame(
        [
            {
                "ts_code": "TEST.SH",
                "report_end_date": "20231231",
                "effective_date": "20240430",
                "daa_ttm": 10.0,
            },
            {
                "ts_code": "TEST.SH",
                "report_end_date": "20240331",
                "effective_date": "20240501",
                "daa_ttm": None,
            },
        ]
    )
    depreciation_events = build_depreciation_events(
        cashflow_events, indicator_events
    )
    assert depreciation_events.to_dict("records") == [
        {
            "ts_code": "TEST.SH",
            "effective_date": "20240430",
            "depreciation_ttm_pit": 10.0,
        }
    ]

    balance = pd.DataFrame(
        [{"ts_code": "TEST.SH", "end_date": "20240331", "effective_date": "20240430", "comp_type": "1", **{field: float(i) for i, field in enumerate(BALANCE_FIELDS)}}]
    )
    balance_events = build_balance_events(balance, {"20240331"})
    assert balance_events.iloc[0]["total_assets_mrq"] == 0.0

    connection = duckdb.connect(":memory:")
    try:
        connection.execute("CREATE TABLE dates(trade_date VARCHAR, ts_code VARCHAR)")
        connection.execute("INSERT INTO dates VALUES ('20240430', 'TEST.SH'), ('20240501', 'TEST.SH')")
        connection.execute("CREATE TABLE events(effective_date VARCHAR, ts_code VARCHAR, value DOUBLE)")
        connection.execute("INSERT INTO events VALUES ('20240430', 'TEST.SH', 1.0)")
        result = connection.execute(
            """
            SELECT d.trade_date, e.value FROM dates AS d
            ASOF LEFT JOIN events AS e
              ON d.ts_code = e.ts_code AND d.trade_date > e.effective_date
            ORDER BY d.trade_date
            """
        ).fetchall()
        assert result == [("20240430", None), ("20240501", 1.0)]
        create_financial_table(connection, "financial_test")
        assert len(connection.execute("PRAGMA table_info('financial_test')").fetchall()) == len(
            FINANCIAL_COLUMNS
        )

        connection.execute(
            "CREATE TABLE market(trade_date VARCHAR, ts_code VARCHAR, "
            "PRIMARY KEY (trade_date, ts_code))"
        )
        connection.execute(
            "INSERT INTO market VALUES "
            "('20240430', 'TEST.SH'), ('20240501', 'TEST.SH')"
        )
        create_financial_table(connection, "financial")
        for column in GROWTH_FIELDS:
            connection.execute(f"ALTER TABLE financial DROP COLUMN {column}")
        connection.execute(
            "INSERT INTO financial (trade_date, ts_code, report_end_date) VALUES "
            "('20240430', 'TEST.SH', '20240331'), "
            "('20240501', 'TEST.SH', '20240331')"
        )
        growth = pd.DataFrame(
            [
                {
                    "ts_code": "TEST.SH",
                    "report_end_date": "20240331",
                    "effective_date": "20240430",
                    "tr_yoy": 1.0,
                    "dt_netprofit_yoy": 2.0,
                    "ocf_yoy": 3.0,
                }
            ]
        )
        add_growth_fields(connection, growth)
        growth_result = connection.execute(
            "SELECT trade_date, fi_effective_date, tr_yoy, "
            "dt_netprofit_yoy, ocf_yoy FROM financial ORDER BY trade_date"
        ).fetchall()
        assert growth_result == [
            ("20240430", None, None, None, None),
            ("20240501", "20240430", 1.0, 2.0, 3.0),
        ]
    finally:
        connection.close()
    print(
        "Self-tests passed: Q1-Q4 TTM, direct YoY indicators, missing "
        "components, MRQ, q-4, strict effective date, schema"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="run offline formula and schema checks")
    parser.add_argument(
        "--full-rebuild",
        action="store_true",
        help="rebuild financial from the complete current market key set",
    )
    args = parser.parse_args()
    if args.self_test:
        run_self_tests()
        return
    if not TUSHARE_API_KEY.strip():
        raise SystemExit("Please set the TUSHARE_API_KEY environment variable")
    if not DATABASE_PATH.exists():
        raise SystemExit(f"Database does not exist: {DATABASE_PATH}")

    counters = defaultdict(ApiCounter)
    connection = duckdb.connect(str(DATABASE_PATH))
    try:
        run_self_tests()
        market_rows, market_min, market_max = connection.execute(
            "SELECT COUNT(*), MIN(trade_date), MAX(trade_date) FROM market"
        ).fetchone()
        universe = {row[0] for row in connection.execute("SELECT DISTINCT ts_code FROM market").fetchall()}
        all_periods = quarter_periods(FIRST_PERIOD, market_max)
        if not args.full_rebuild and financial_needs_growth_fields(connection):
            validate_key_state(connection)
            print(
                f"Mode: add growth fields only; market={market_rows:,} rows, "
                f"{len(universe):,} stocks, {market_min}-{market_max}; "
                f"fetching {len(all_periods)} quarters "
                f"({all_periods[0]}-{all_periods[-1]})"
            )
            pro = ts.pro_api(TUSHARE_API_KEY.strip())
            raw_indicator = fetch_source(
                pro, "fina_indicator", all_periods, universe, market_max, counters
            )
            growth_events = build_growth_events(raw_indicator, set(all_periods))
            add_growth_fields(connection, growth_events)
            print("API summary:")
            for endpoint in sorted(counters):
                counter = counters[endpoint]
                print(
                    f"  {endpoint}: calls={counter.calls:,}, retries={counter.retries:,}, "
                    f"fallback_calls={counter.fallback_calls:,}, "
                    f"capped_responses={counter.capped_responses:,}"
                )
            report_coverage(connection)
            print(
                "Done. financial now includes tr_yoy, dt_netprofit_yoy, and "
                "ocf_yoy; existing columns were preserved."
            )
            return

        if not args.full_rebuild and financial_needs_fix_assets_total(connection):
            validate_key_state(connection)
            print(
                f"Mode: add fix_assets_total only; market={market_rows:,} rows, "
                f"{len(universe):,} stocks, {market_min}-{market_max}; "
                f"fetching {len(all_periods)} quarters ({all_periods[0]}-{all_periods[-1]})"
            )
            pro = ts.pro_api(TUSHARE_API_KEY.strip())
            raw_balance = fetch_source(
                pro,
                "balancesheet",
                all_periods,
                universe,
                market_max,
                counters,
            )
            balance_events = build_balance_events(raw_balance, set(all_periods))
            add_fix_assets_total_column(connection, balance_events)
            print("API summary:")
            for endpoint in sorted(counters):
                counter = counters[endpoint]
                print(
                    f"  {endpoint}: calls={counter.calls:,}, retries={counter.retries:,}, "
                    f"fallback_calls={counter.fallback_calls:,}, "
                    f"capped_responses={counter.capped_responses:,}"
                )
            report_coverage(connection)
            print("Done. financial now includes fix_assets_total_mrq; existing columns were preserved.")
            return

        if not args.full_rebuild and financial_needs_depreciation_ttm_pit(connection):
            validate_key_state(connection)
            print(
                f"Mode: add depreciation_ttm_pit only; market={market_rows:,} rows, "
                f"{len(universe):,} stocks, {market_min}-{market_max}; "
                f"fetching {len(all_periods)} quarters ({all_periods[0]}-{all_periods[-1]})"
            )
            pro = ts.pro_api(TUSHARE_API_KEY.strip())
            raw_cashflow = fetch_source(
                pro, "cashflow", all_periods, universe, market_max, counters
            )
            raw_indicator = fetch_source(
                pro, "fina_indicator", all_periods, universe, market_max, counters
            )
            calculate_periods = set(all_periods)
            cashflow_events = build_ttm_events(
                raw_cashflow, CASHFLOW_FIELDS, CASHFLOW_OUTPUTS, calculate_periods
            )
            indicator_events = build_indicator_events(
                raw_indicator, calculate_periods
            )
            depreciation_events = build_depreciation_events(
                cashflow_events, indicator_events
            )
            add_depreciation_ttm_pit_column(connection, depreciation_events)
            print("API summary:")
            for endpoint in sorted(counters):
                counter = counters[endpoint]
                print(
                    f"  {endpoint}: calls={counter.calls:,}, retries={counter.retries:,}, "
                    f"fallback_calls={counter.fallback_calls:,}, "
                    f"capped_responses={counter.capped_responses:,}"
                )
            report_coverage(connection)
            print(
                "Done. financial now includes depreciation_ttm_pit; "
                "existing source columns were preserved."
            )
            return

        incremental = not args.full_rebuild and financial_schema_matches(connection)
        financial_max = validate_key_state(connection) if incremental else None
        fetch_periods = all_periods[-9:] if incremental else all_periods
        calculate_periods = set(all_periods[-5:] if incremental else all_periods)
        print(
            f"Mode: {'incremental' if incremental else 'full'}; market={market_rows:,} rows, "
            f"{len(universe):,} stocks, {market_min}-{market_max}; "
            f"fetching {len(fetch_periods)} quarters ({fetch_periods[0]}-{fetch_periods[-1]})"
        )

        pro = ts.pro_api(TUSHARE_API_KEY.strip())
        raw_income = fetch_source(pro, "income", fetch_periods, universe, market_max, counters)
        raw_balance = fetch_source(pro, "balancesheet", fetch_periods, universe, market_max, counters)
        raw_cashflow = fetch_source(pro, "cashflow", fetch_periods, universe, market_max, counters)
        raw_indicator = fetch_source(pro, "fina_indicator", fetch_periods, universe, market_max, counters)

        activation = build_activation_events(raw_income, raw_balance, calculate_periods)
        income_events = build_ttm_events(raw_income, INCOME_FIELDS, INCOME_OUTPUTS, calculate_periods)
        balance_events = build_balance_events(raw_balance, calculate_periods)
        cashflow_events = build_ttm_events(
            raw_cashflow, CASHFLOW_FIELDS, CASHFLOW_OUTPUTS, calculate_periods
        )
        indicator_events = build_indicator_events(raw_indicator, calculate_periods)
        depreciation_events = build_depreciation_events(
            cashflow_events, indicator_events
        )

        if incremental:
            earliest_event = activation["effective_date"].min() if not activation.empty else None
            repo_start = min(earliest_event, financial_max) if earliest_event else financial_max
        else:
            earliest_event = None
            repo_start = market_min
        repo = fetch_repo_daily(pro, repo_start, market_max, counters)

        register_events(
            connection,
            activation,
            income_events,
            balance_events,
            cashflow_events,
            indicator_events,
            depreciation_events,
            repo,
        )
        try:
            build_table(connection, incremental, earliest_event)
            validate_table(connection, "financial__build")
        finally:
            unregister_events(connection)

        connection.execute("BEGIN TRANSACTION")
        try:
            connection.execute("DROP TABLE IF EXISTS financial")
            connection.execute("ALTER TABLE financial__build RENAME TO financial")
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
        connection.execute("CHECKPOINT")

        print("API summary:")
        for endpoint in sorted(counters):
            counter = counters[endpoint]
            print(
                f"  {endpoint}: calls={counter.calls:,}, retries={counter.retries:,}, "
                f"fallback_calls={counter.fallback_calls:,}, capped_responses={counter.capped_responses:,}"
            )
        report_coverage(connection)
        print("Done. financial is ready and all temporary API data has been released.")
    finally:
        connection.close()


if __name__ == "__main__":
    main()
