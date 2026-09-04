"""面向全市场股票池的数据管理器。"""

from itertools import groupby

from data_manager import DataManager


class AllStockDataManager(DataManager):
    """保持 ``DataManager`` 接口不变，按交易日返回全部股票。"""

    def __init__(
        self,
        db_path: str,
        pool_name: str,
        start_date: str,
        end_date: str,
    ):
        if pool_name != "all":
            raise ValueError("pool_name must be 'all'")

        # 父类仍加载中证500快照，供 get_ff3_factor_data 构造既有口径的
        # SMB/HML；因子评价所用股票池由本类下面两个重写方法决定。
        super().__init__(db_path, "zz500", start_date, end_date)
        self.pool_name = pool_name
        self.all_symbols_by_date = {
            trade_date: list(symbols)
            for trade_date, symbols in self.connection.execute(
                """
                SELECT trade_date, list(ts_code ORDER BY ts_code) AS symbols
                FROM market
                WHERE trade_date BETWEEN ? AND ?
                GROUP BY trade_date
                ORDER BY trade_date
                """,
                [start_date, end_date],
            ).fetchall()
        }

    def get_symbols(self, trade_date: str) -> list[str]:
        """返回当日有行情且未被标记为ST或停牌的全部股票。"""
        excluded = self.excluded_symbols.get(trade_date, set())
        return [
            symbol
            for symbol in self.all_symbols_by_date.get(trade_date, [])
            if symbol not in excluded
        ]

    def get_constituent_periods(self):
        """按季度分块返回交易日及该季度出现过的股票并集。"""
        trading_dates = self.calender.loc[
            self.calender["is_open"] == 1, "cal_date"
        ].tolist()
        for _, dates in groupby(
            trading_dates,
            key=lambda date: (date[:4], (int(date[4:6]) - 1) // 3),
        ):
            period_dates = list(dates)
            period_symbols = sorted(
                {
                    symbol
                    for trade_date in period_dates
                    for symbol in self.all_symbols_by_date.get(trade_date, [])
                }
            )
            yield period_dates, period_symbols
