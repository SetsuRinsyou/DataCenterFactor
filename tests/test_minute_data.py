"""分钟读取、日内统计和日频因子兼容性测试。"""

import gc
import io
import tempfile
import unittest
import weakref
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import duckdb
import numpy as np
import pandas as pd

from data_manager import DataManager, AllStockDataManager
from factor_base import FactorBase


class ClosingVolumeFactor(FactorBase):
    def __init__(self):
        super().__init__(window="2D", data_fields=[], minute_fields=["vol", "close"],
                         minute_window_days=2)

    def calculate_daily_factor(self, trade_date, symbols, market_data, history_start,
                               *, minute_data=None):
        times = minute_data.index.get_level_values("trade_time")
        dates = times.strftime("%Y%m%d")
        stock_codes = minute_data.index.get_level_values("ts_code")
        assert all(d <= trade_date for d in dates)
        total = minute_data.vol.groupby([dates, stock_codes]).sum()
        closing_mask = ((times.strftime("%H:%M:%S") > "14:30:00")
                        & (times.strftime("%H:%M:%S") <= "15:00:00"))
        closing = minute_data.vol.where(closing_mask, 0).groupby([dates, stock_codes]).sum()
        ratio = closing / total.where(total > 0)
        if ratio.empty:
            values = pd.DataFrame(index=minute_data.attrs["trade_dates"], columns=symbols, dtype=float)
        else:
            values = ratio.unstack().reindex(index=minute_data.attrs["trade_dates"], columns=symbols)
        return values.mean().where(values.count() == self.minute_window_days)

class DailyFactor(FactorBase):
    def __init__(self):
        super().__init__(window="2D", data_fields=["close"])

    def calculate_daily_factor(self, trade_date, symbols, market_data, history_start):
        values = market_data["close"].loc[history_start:trade_date]
        return values.iloc[-1] / values.iloc[0] - 1


class MinuteTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.daily = self.root / "daily.duckdb"
        self.minute = self.root / "minute.duckdb"
        self.dates = pd.bdate_range("2022-01-04", periods=12).strftime("%Y%m%d").tolist()
        self.codes = [f"{i:06d}.SZ" for i in range(1, 7)]
        with duckdb.connect(str(self.daily)) as c:
            c.execute("CREATE TABLE calender(cal_date VARCHAR, is_open INTEGER)")
            c.executemany("INSERT INTO calender VALUES (?,1)", [(d,) for d in self.dates])
            c.execute("CREATE TABLE market(trade_date VARCHAR,ts_code VARCHAR,close DOUBLE,adj_factor DOUBLE)")
            c.executemany("INSERT INTO market VALUES (?,?,?,1)", [
                (d, code, 10 + j + i * (j + 1) / 10)
                for i, d in enumerate(self.dates) for j, code in enumerate(self.codes)
            ])
            c.execute("CREATE TABLE financial(trade_date VARCHAR,ts_code VARCHAR)")
            c.execute("CREATE TABLE anomaly(trade_date VARCHAR,ts_code VARCHAR,value VARCHAR)")
            c.execute("CREATE TABLE zz500_constituents(trade_date VARCHAR," +
                      ",".join(f"s{i} VARCHAR" for i in range(6)) + ")")
            c.executemany("INSERT INTO zz500_constituents VALUES (?,?,?,?,?,?,?)", [
                [self.dates[0], *self.codes[:5], None], [self.dates[6], *self.codes]
            ])
        with duckdb.connect(str(self.minute)) as c:
            c.execute("CREATE TABLE market_1min(ts_code VARCHAR,trade_time TIMESTAMP,"
                      "open DOUBLE,high DOUBLE,low DOUBLE,close DOUBLE,vol DOUBLE,amount DOUBLE)")
            c.executemany("INSERT INTO market_1min VALUES (?,?,10,10,10,10,?,?)", [
                (code, str(pd.Timestamp(d).date()) + " " + t, vol, vol * 10)
                for i, d in enumerate(self.dates) for j, code in enumerate(self.codes)
                for t, vol in [("09:31:00", 100), ("15:00:00", (i + 1) * (j + 1))]
            ])
        self.manager = DataManager(str(self.daily), "zz500", self.dates[4], self.dates[8], str(self.minute))

    def tearDown(self):
        self.manager.close()
        self.temp.cleanup()

    def compute(self, factor=None):
        with redirect_stdout(io.StringIO()):
            return (factor or ClosingVolumeFactor()).compute_eval(self.manager, 1)

    def test_read_boundaries_warmup_filter_and_initialized_close(self):
        m = self.manager
        self.assertIsNotNone(m.minute_connection)
        self.assertIn("vol", m.minute_columns)
        self.assertNotIn("adj_factor", m.minute_columns)
        result = m.get_minute_data(self.dates[2], ["vol"], [self.codes[1]])
        self.assertEqual(len(result), 2)
        self.assertEqual(result.columns.tolist(), ["vol"])
        self.assertEqual(result.index.names, ["trade_time", "ts_code"])
        self.assertTrue(result.index.is_monotonic_increasing)
        self.assertEqual(result.vol.tolist(), [100, 6])
        self.assertTrue(m.get_minute_data(self.dates[2], ["vol"], []).empty)
        self.assertTrue(m.get_minute_data(self.dates[2], ["vol"], ["absent"]).empty)
        for date, fields in [("20220108", ["vol"]), (self.dates[2], ["bad"])]:
            with self.assertRaises(ValueError):
                m.get_minute_data(date, fields, self.codes)
        m.close()
        m.close()

    def test_initialization_failure_and_context_cleanup(self):
        broken = self.root / "broken.duckdb"
        with duckdb.connect(str(broken)) as c:
            c.execute("CREATE TABLE unrelated(x INT)")
        connect = duckdb.connect
        for minute in (self.root / "absent", broken):
            opened = []
            def tracked(*args, **kwargs):
                c = connect(*args, **kwargs)
                opened.append(c)
                return c
            with patch("data_manager.duckdb.connect", side_effect=tracked):
                with self.assertRaises(duckdb.Error):
                    DataManager(str(self.daily), "zz500", self.dates[4], self.dates[8], str(minute))
            for c in opened:
                with self.assertRaises(duckdb.Error):
                    c.execute("SELECT 1")
        for source in ("connection", "minute_connection"):
            m = DataManager(str(self.daily), "zz500", self.dates[4], self.dates[8], str(self.minute))
            connections = (m.connection, m.minute_connection)
            with self.assertRaises(duckdb.Error):
                with m as entered:
                    self.assertIs(entered, m)
                    getattr(m, source).execute("SELECT * FROM absent_table")
            for c in connections:
                with self.assertRaises(duckdb.Error):
                    c.execute("SELECT 1")
        self.manager.close()
        daily, minute = Mock(), Mock()
        self.manager.connection, self.manager.minute_connection = daily, minute
        minute.close.side_effect = RuntimeError("cleanup failed")
        try:
            with self.assertRaisesRegex(ValueError, "original"):
                with self.manager:
                    raise ValueError("original")
            daily.close.assert_called_once()
            minute.close.assert_called_once()
            with self.assertRaisesRegex(RuntimeError, "cleanup failed"):
                with self.manager:
                    pass
        finally:
            self.manager.connection = self.manager.minute_connection = None

    def test_rolling_switch_coverage_and_release(self):
        reads, refs = [], []
        original = self.manager.get_minute_data
        def read(date, fields, symbols, window_days=1):
            gc.collect()
            self.assertLessEqual(sum(ref() is not None for ref in refs), 1)
            result = original(date, fields, symbols, window_days=window_days)
            reads.append((date, window_days, tuple(symbols)))
            refs.append(weakref.ref(result))
            return result
        with patch.object(self.manager, "get_minute_data", side_effect=read), \
             patch.object(self.manager, "get_market_data", side_effect=AssertionError("unneeded daily read")):
            signals, ic, groups = self.compute()
        self.assertEqual(len(reads), 5)  # First window of each period is fetched in one query.
        self.assertEqual([(date, days) for date, days, _ in reads], [
            (self.dates[4], 2), (self.dates[5], 1),
            (self.dates[6], 2), (self.dates[7], 1),
            (self.dates[8], 1),
        ])
        for i in range(4, 9):
            for j, code in enumerate(self.codes[:5] if i < 6 else self.codes):
                expected = np.mean([k * (j + 1) / (100 + k * (j + 1)) for k in (i, i + 1)])
                self.assertAlmostEqual(signals.loc[self.dates[i], code], expected)
        self.assertEqual(ic.shape, (5,))
        self.assertEqual(groups.shape, (5, 5))
        self.assertTrue(pd.isna(signals.loc[self.dates[4], self.codes[5]]))

    def test_missing_day_zero_volume_and_no_future_leakage(self):
        self.manager.close()
        with duckdb.connect(str(self.minute)) as c:
            c.execute("DELETE FROM market_1min WHERE CAST(trade_time AS DATE)=?", [pd.Timestamp(self.dates[3]).date()])
            c.execute("UPDATE market_1min SET vol=0,amount=0 WHERE ts_code=? AND CAST(trade_time AS DATE)=?",
                      [self.codes[0], pd.Timestamp(self.dates[5]).date()])
        self.manager = DataManager(str(self.daily), "zz500", self.dates[4], self.dates[8], str(self.minute))
        first = self.compute()[0]
        self.assertTrue(first.loc[self.dates[4]].isna().all())
        self.assertTrue(pd.isna(first.loc[self.dates[5], self.codes[0]]))
        zero = self.manager.get_minute_data(self.dates[5], ["vol"], [self.codes[0]])
        self.assertEqual(zero.vol.tolist(), [0, 0])
        self.manager.close()
        with duckdb.connect(str(self.minute)) as c:
            c.execute("UPDATE market_1min SET vol=10000 WHERE CAST(trade_time AS DATE)=?", [pd.Timestamp(self.dates[8]).date()])
        self.manager = DataManager(str(self.daily), "zz500", self.dates[4], self.dates[8], str(self.minute))
        second = self.compute()[0]
        pd.testing.assert_frame_equal(first.loc[:self.dates[7]], second.loc[:self.dates[7]])


    def test_daily_factor_without_minute_database_and_all_stock(self):
        self.manager.close()
        self.manager = DataManager(str(self.daily), "zz500", self.dates[4], self.dates[8])
        result = self.compute(DailyFactor())
        self.assertEqual(result[0].shape, (5, 6))
        self.assertIsNone(self.manager.minute_connection)
        m = AllStockDataManager(str(self.daily), "all", self.dates[4], self.dates[8], str(self.minute))
        try:
            self.assertEqual(m.get_symbols(self.dates[4]), self.codes)
            self.assertEqual(len(m.get_minute_data(self.dates[2], ["vol"], self.codes)), 12)
        finally:
            m.close()


    def test_range_read_and_window_validation(self):
        data = self.manager.get_minute_data(
            self.dates[4], ["vol"], self.codes, window_days=3
        )
        self.assertEqual(len(data), 36)
        with self.assertRaises(ValueError):
            self.manager.get_minute_data(
                self.dates[4], ["vol"], self.codes, window_days=0
            )
        for value in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                FactorBase.__init__(ClosingVolumeFactor(), "2D", [], minute_window_days=value)
        factor = ClosingVolumeFactor()
        factor.minute_window_days = 20
        with self.assertRaisesRegex(ValueError, "calendar history"):
            self.compute(factor)
        for fields in (["unknown"], ["vol", "vol"]):
            factor = ClosingVolumeFactor()
            factor.minute_fields = fields
            with patch.object(self.manager, "get_market_data", side_effect=AssertionError("early validation")):
                with self.assertRaises(ValueError):
                    self.compute(factor)

    def test_mixed_daily_financial_minutes_and_single_day(self):
        self.manager.close()
        with duckdb.connect(str(self.daily)) as c:
            c.execute("ALTER TABLE financial ADD COLUMN book_value DOUBLE")
            c.execute("INSERT INTO financial SELECT trade_date,ts_code,2.0 FROM market")
        self.manager = DataManager(str(self.daily), "zz500", self.dates[4], self.dates[8], str(self.minute))
        class Mixed(ClosingVolumeFactor):
            def __init__(self):
                super().__init__()
                self.data_fields = ["close", "book_value"]
                self.minute_window_days = 1
            def calculate_daily_factor(self, date, symbols, market_data, history_start, *, minute_data=None):
                self.assert_inputs = (list(market_data), minute_data.attrs["trade_dates"])
                assert all(values.index.max() <= date for values in market_data.values())
                assert minute_data.attrs["trade_dates"] == (date,)
                ratio = super().calculate_daily_factor(date, symbols, market_data, history_start,
                                                       minute_data=minute_data)
                return ratio * market_data["close"].loc[date] / market_data["book_value"].loc[date]
        factor = Mixed()
        reads = []
        original = self.manager.get_minute_data
        def read(date, fields, symbols, window_days=1):
            reads.append(date)
            return original(date, fields, symbols, window_days=window_days)
        with patch.object(self.manager, "get_minute_data", side_effect=read):
            result = self.compute(factor)[0]
        self.assertEqual(reads, self.dates[4:9])
        for i in range(4, 9):
            for j, code in enumerate(self.codes[:5] if i < 6 else self.codes):
                volume = (i + 1) * (j + 1)
                expected = volume / (100 + volume) * (10 + j + i * (j + 1) / 10) / 2
                self.assertAlmostEqual(result.loc[self.dates[i], code], expected)

    def test_runner_paths_and_cleanup(self):
        import run_pipeline
        args = SimpleNamespace(
            forward_days=5, db_path=str(self.daily), output_db=str(self.root / "result.duckdb"),
            minute_db_path=str(self.minute), factor_module="", factor_class="", factor_params={},
            pool_name="zz500", start_date=self.dates[4], end_date=self.dates[8],
        )
        for factor in (DailyFactor(), ClosingVolumeFactor()):
            factor.compute_eval = Mock(side_effect=RuntimeError("calculation failed"))
            with patch.object(run_pipeline, "parse_args", return_value=args), \
                 patch.object(run_pipeline, "load_factor", return_value=factor), \
                 patch.object(run_pipeline, "DataManager") as manager, redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "calculation failed"):
                    run_pipeline.main()
                manager.return_value.close.assert_called_once()
                self.assertEqual(manager.call_args.kwargs["minute_db_path"],
                                 str(self.minute) if factor.minute_fields else None)
        args.output_db = str(self.minute)
        with patch.object(run_pipeline, "parse_args", return_value=args), \
             patch.object(run_pipeline, "load_factor", return_value=ClosingVolumeFactor()), \
             patch.object(run_pipeline, "DataManager") as manager:
            with self.assertRaisesRegex(ValueError, "minute database"):
                run_pipeline.main()
            manager.assert_not_called()


if __name__ == "__main__":
    unittest.main()
