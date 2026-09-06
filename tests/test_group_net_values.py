"""每日持仓估值与收益图的行为回归测试。"""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import duckdb
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from factor_base import FactorBase
from eval_plot import plot_group_cumulative_returns


class DailyNetValueTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.bdate_range('2024-01-01', periods=9).strftime('%Y%m%d').tolist()
        self.symbols = [f's{i}' for i in range(10)]
        self.prices = pd.DataFrame(100.0, index=self.dates, columns=self.symbols)
        self.signals = pd.DataFrame(
            [np.arange(10)] * len(self.dates), index=self.dates, columns=self.symbols
        )
        self.connection = duckdb.connect(':memory:')
        self.connection.execute('CREATE TABLE anomaly (trade_date VARCHAR, ts_code VARCHAR, value VARCHAR)')
        self.reads = []

        def get_market_data(dates, window, fields, symbols):
            self.reads.append((list(dates), list(symbols)))
            frame = self.prices.reindex(index=dates, columns=symbols)
            frame.index.name = 'trade_date'
            frame.columns.name = 'ts_code'
            return frame.stack(future_stack=True).rename('close').to_frame()

        self.manager = SimpleNamespace(
            calender=pd.DataFrame({'cal_date': self.dates, 'is_open': 1}),
            connection=self.connection,
            get_market_data=get_market_data,
        )

    def tearDown(self):
        self.connection.close()

    def calculate(self, days=5):
        return FactorBase.calculate_group_net_values(None, self.signals, self.manager, days)

    def suspend(self, date, symbol):
        self.connection.execute('INSERT INTO anomaly VALUES (?, ?, ?)', [date, symbol, 'SUSPENDED'])

    def test_drawdown_recovery_and_plot_metrics(self):
        self.prices.loc[self.dates[2], ['s0', 's1']] = 70
        nav = self.calculate()
        self.assertAlmostEqual(nav.group_1.iloc[-1], 1)
        self.assertAlmostEqual((nav.group_1 / nav.group_1.cummax() - 1).min(), -.3)
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(Figure, 'savefig', autospec=True) as save:
                plot_group_cumulative_returns(nav, Path(directory) / 'curve.png')
                axis = save.call_args.args[0].axes[0]
                self.assertIn('max drawdown=-30.00%', axis.lines[0].get_label())
                self.assertEqual(len(axis.lines[0].get_xdata()), len(self.dates) - 1)
            path = plot_group_cumulative_returns(nav, Path(directory) / 'curve.png')
            self.assertGreater(path.stat().st_size, 0)

    def test_fixed_shares_and_old_period_endpoint(self):
        self.prices.loc[self.dates[2]:, 's0'] = 200
        self.prices.loc[self.dates[3]:, 's1'] = 200
        nav = self.calculate()
        self.assertAlmostEqual(nav.group_1.iloc[1], 1.5)
        self.assertAlmostEqual(nav.group_1.iloc[2], 2)  # daily equal weighting would give 2.25
        endpoint_returns = self.prices.loc[self.dates[6]] / self.prices.loc[self.dates[1]] - 1
        old = FactorBase.calculate_group_returns(self.signals.iloc[0], endpoint_returns)
        np.testing.assert_allclose(nav.loc[self.dates[6]], 1 + old)

    def test_daily_coverage_and_partial_last_period(self):
        for days in (1, 5, 20):
            nav = self.calculate(days)
            self.assertEqual(list(nav.index), self.dates[1:])
            np.testing.assert_allclose(nav, 1)

    def test_missing_signal_does_not_shift_schedule_or_sell(self):
        self.signals.loc[self.dates[3]] = np.nan
        self.prices.loc[self.dates[5]:, 's0'] = 200
        nav = self.calculate(3)
        self.assertAlmostEqual(nav.group_1.loc[self.dates[5]], 1.5)
        self.assertEqual([dates[0] for dates, _ in self.reads], [self.dates[i] for i in (1, 4, 7)])

    def test_rebalance_day_values_old_holdings_before_trading(self):
        self.signals.loc[self.dates[3]:] = np.arange(10)[::-1]
        self.prices.loc[self.dates[4]:, ['s0', 's1']] = 200
        self.prices.loc[self.dates[5]:, ['s8', 's9']] = 300
        nav = self.calculate(3)
        self.assertAlmostEqual(nav.group_1.loc[self.dates[4]], 2)
        self.assertAlmostEqual(nav.group_1.loc[self.dates[5]], 6)

    def test_old_holdings_survive_signal_universe_change(self):
        self.signals.loc[self.dates[3]:, ['s0', 's1']] = np.nan
        self.prices.loc[self.dates[4]:, ['s0', 's1']] = 200
        nav = self.calculate(3)
        self.assertAlmostEqual(nav.group_1.loc[self.dates[4]], 2)
        self.assertTrue({'s0', 's1'}.issubset(self.reads[1][1]))

    def test_suspended_holding_carries_mark_and_recovers(self):
        date = self.dates[2]
        self.suspend(date, 's0')
        self.prices.loc[date, 's0'] = np.nan
        self.prices.loc[self.dates[3]:, 's0'] = 60
        nav = self.calculate()
        self.assertAlmostEqual(nav.group_1.loc[date], 1)
        self.assertAlmostEqual(nav.group_1.loc[self.dates[3]], .8)

    def test_suspended_position_is_not_sold_at_rebalance(self):
        self.signals.loc[self.dates[3]:] = np.arange(10)[::-1]
        self.suspend(self.dates[4], 's0')
        self.prices.loc[self.dates[4], 's0'] = np.nan
        self.prices.loc[self.dates[5]:, 's0'] = 200
        nav = self.calculate(3)
        self.assertAlmostEqual(nav.group_1.loc[self.dates[5]], 1.5)

    def test_suspended_entry_retains_cash(self):
        self.suspend(self.dates[1], 's0')
        self.prices.loc[self.dates[1], 's0'] = np.nan
        self.prices.loc[self.dates[2]:, 's0'] = 200
        self.prices.loc[self.dates[2]:, 's1'] = 200
        nav = self.calculate()
        self.assertAlmostEqual(nav.group_1.loc[self.dates[2]], 1.5)

    def test_unexplained_missing_prices_fail(self):
        for date in (self.dates[1], self.dates[2]):
            with self.subTest(date=date):
                self.prices.loc[date, 's0'] = np.nan
                with self.assertRaisesRegex(ValueError, 'Missing .*price'):
                    self.calculate()
                self.prices.loc[date, 's0'] = 100

    def test_plot_rejects_missing_valuation_instead_of_dropping_day(self):
        nav = self.calculate()
        nav.iloc[2, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'finite positive'):
            plot_group_cumulative_returns(nav)

    def test_invalid_rebalance_period(self):
        for value in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                self.calculate(value)


if __name__ == '__main__':
    unittest.main()
