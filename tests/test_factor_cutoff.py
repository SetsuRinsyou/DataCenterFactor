"""日频和分钟因子均只能看到信号日及之前的日频输入。"""

import io
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace

import numpy as np
import pandas as pd

from factor_base import FactorBase


class FactorCutoffTests(unittest.TestCase):
    def test_daily_and_minute_inputs_stop_at_signal_date(self):
        dates = ["20220104", "20220105", "20220106"]
        symbols = ["a", "b", "c", "d", "e"]
        index = pd.MultiIndex.from_product(
            [dates, symbols], names=["trade_date", "ts_code"]
        )
        prices = pd.DataFrame({"close": np.arange(15) + 1.0}, index=index)
        financial = prices.rename(columns={"close": "book_value"})
        index_prices = pd.DataFrame(
            {"close": [1., 2., 999.]},
            index=pd.MultiIndex.from_product(
                [dates, ["index"]], names=["trade_date", "ts_code"]
            ),
        )
        ff3 = pd.DataFrame({"mkt": [1., 2., 999.]}, index=dates)
        manager = SimpleNamespace(
            market_columns={"close"}, financial_columns={"book_value"},
            minute_columns={"trade_time", "ts_code", "vol"},
            calender=pd.DataFrame({"cal_date": dates[1:], "is_open": 1}),
            open_dates=dates, open_date_positions=dict(zip(dates, range(3))),
            get_constituent_periods=lambda: [(dates[1:], symbols)],
            get_symbols=lambda date: symbols,
            get_pre_window_start_date=lambda date, window: dates[0],
            get_market_data=lambda *args: prices.copy(),
            get_financial_data=lambda *args: financial.copy(),
            get_index_data=lambda *args: index_prices.copy(),
            get_ff3_factor_data=lambda *args: ff3.copy(),
            get_eval_data=lambda *args: prices.rename(columns={"close": "return"}),
            get_minute_data=lambda date, *args, **kwargs: pd.DataFrame(
                {"vol": 1.}, index=pd.MultiIndex.from_product(
                    [[pd.Timestamp(date) + pd.Timedelta(hours=15)], symbols],
                    names=["trade_time", "ts_code"],
                ),
            ),
        )
        test = self

        class LastRowFactor(FactorBase):
            def __init__(self, minutes):
                super().__init__("1D", ["close", "book_value"],
                                 index_code="index", requires_ff3=True,
                                 minute_fields=["vol"] if minutes else [])

            def calculate_daily_factor(self, date, symbols, market_data,
                                       history_start, *, minute_data=None):
                test.assertEqual(set(market_data), {"close", "book_value", "index_close", "ff3"})
                for values in market_data.values():
                    test.assertEqual(values.index.tolist(), dates[:dates.index(date) + 1])
                if self.minute_fields:
                    test.assertEqual(minute_data.attrs["trade_dates"], (date,))
                return market_data["close"].tail(1).iloc[-1]

        for minutes in (False, True):
            with self.subTest(minutes=minutes), redirect_stdout(io.StringIO()):
                signals, _, _ = LastRowFactor(minutes).compute_eval(manager, 1)
                pd.testing.assert_frame_equal(
                    signals, prices["close"].unstack("ts_code").loc[dates[1:]],
                )


if __name__ == "__main__":
    unittest.main()
