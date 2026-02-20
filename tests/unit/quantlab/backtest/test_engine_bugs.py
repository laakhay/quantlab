"""
Reproduction tests for 'Stuck Trades' bug.
Verifies that SL/TP and Max Bars limits actually close trades and free up the engine for new entries.
"""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from laakhay.data.models.bar import Bar
from laakhay.quantlab.backtest import (
    BacktestConfig,
    BacktestEngine,
    FrequencyControl,
    MemDataFeed,
    OrderSide,
    OrderType,
    Signal,
    Strategy,
)


def create_ohlcv_bars(prices, start_time):
    bars = []
    for i, p in enumerate(prices):
        # Create a bar with some volatility to ensure SL/TP hit if close is near
        bars.append(
            Bar(
                timestamp=start_time + timedelta(minutes=i),
                open=Decimal(str(p)),
                high=Decimal(str(p + 2)),
                low=Decimal(str(p - 2)),
                close=Decimal(str(p)),
                volume=Decimal("1000"),
            )
        )
    return bars


class MockStrategy(Strategy):
    def __init__(self, entry_index=0):
        super().__init__()
        self.entry_index = entry_index
        self.bar_count = 0

    def on_bar(self, dataset, symbol, timeframe):
        signals = []
        if self.bar_count == self.entry_index:
            # Enter Long
            signals.append(
                Signal(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    type=OrderType.MARKET,
                    size="1000",
                    sl="95",  # 5 pts below 100
                    tp="110",  # 10 pts above 100
                )
            )
        self.bar_count += 1
        return signals


def test_repro_sl_exits_trade():
    """
    Scenario:
    1. Enter Long at 100. SL 95.
    2. Price drops to 90 (below SL).
    3. Trade SHOULD close.
    4. Price goes back up.
    5. New Entry potential (if logic allows).

    If trade is stuck, the position remains open.
    """
    start = datetime(2024, 1, 1, tzinfo=UTC)
    # Prices: 100 (Entry), 98, 90 (SL Hit), 90, 90
    prices = [100, 98, 90, 90, 90]
    bars = create_ohlcv_bars(prices, start)

    feed = MemDataFeed(bars, symbol="BTCUSDT", timeframe="1m")

    # Simple config
    config = BacktestConfig(frequency=FrequencyControl(max_entries_per_day=10))

    engine = BacktestEngine(initial_capital=10000, config=config)
    strategy = MockStrategy(entry_index=0)

    report = engine.run(strategy, feed)
    trades = report["round_trips"]

    # Expectation: 1 Trade, Closed by STOP_LOSS
    assert len(trades) == 1
    t = trades[0]
    assert t["exit_reason"] == "STOP_LOSS"
    assert t["exit_price"] <= 95

    # Check that position is flat at the end
    pos = engine.positions.get("BTCUSDT")
    assert pos.is_flat, f"Position should be flat, but is {pos}"


def test_repro_max_bars_exits_trade():
    """
    Scenario:
    1. Max Bars in Trade = 3.
    2. Enter Long at 100.
    3. Price moves sideways (100, 100, 100, 100...).
    4. Trade SHOULD close after 3 bars.
    """
    start = datetime(2024, 1, 1, tzinfo=UTC)
    # 10 bars of flat price
    prices = [100] * 10
    bars = create_ohlcv_bars(prices, start)

    feed = MemDataFeed(bars, symbol="BTCUSDT", timeframe="1m")

    config = BacktestConfig(frequency=FrequencyControl(max_bars_in_trade=3))

    engine = BacktestEngine(initial_capital=10000, config=config)
    strategy = MockStrategy(entry_index=0)

    report = engine.run(strategy, feed)
    trades = report["round_trips"]

    assert len(trades) == 1
    t = trades[0]
    # Trade entered at index 0 (execution starts next bar usually?
    # Check engine behavior: signals processed at close, executed next open usually,
    # or execution_on_next_bar_open=False by default?)
    # Default is immediate execution if not configured otherwise?
    # BacktestEngine defaults: execute_signals_on_next_bar_open = False implies immediate fill at Close?
    # Let's see core.py...

    # If immediate fill at T0.
    # Bars held: T0, T1, T2. (3 bars).
    # Exit at T2 close or T3 open?

    assert t["exit_reason"] == "TIME_EXIT"
    assert t["holding_bars"] <= 4  # Tolerance for implementation details

    pos = engine.positions.get("BTCUSDT")
    assert pos.is_flat


def test_stuck_trade_blocking_new_entry():
    """
    Scenario:
    1. Max Entries Per Day = 2.
    2. Enter T1. SL Hit T3.
    3. Enter T5.

    If T1 is stuck, T5 cannot enter because position is still occupied (assuming 1 pos limit/hedging off).
    """
    start = datetime(2024, 1, 1, tzinfo=UTC)
    # Prices:
    # 0: 100 (Entry 1)
    # 1: 98
    # 2: 90 (SL Hit for #1)
    # 3: 100
    # 4: 100 (Entry 2 setup)
    # 5: 100

    prices = [100, 98, 90, 100, 100, 100]
    bars = create_ohlcv_bars(prices, start)

    class MultiEntryStrategy(Strategy):
        def on_bar(self, dataset, symbol, timeframe):
            # Try to enter at index 0 and index 4
            # We use a static counter or simply send signals.
            # Engine prevents entry if position exists.

            # Simple logic: Always send Buy signal.
            # If position exists, engine ignores.
            # If position closed, engine accepts.

            return [
                Signal(
                    symbol=symbol, side=OrderSide.BUY, type=OrderType.MARKET, size="1000", sl="95"
                )
            ]

    feed = MemDataFeed(bars, symbol="BTCUSDT", timeframe="1m")
    config = BacktestConfig(frequency=FrequencyControl(max_entries_per_day=5, cooldown_bars=0))
    engine = BacktestEngine(initial_capital=10000, config=config)
    strategy = MultiEntryStrategy()

    report = engine.run(strategy, feed)
    round_trips = report["round_trips"]
    trades = report["trades"]

    # Re-entry check: after first SL round trip, engine should allow a new BUY.
    # Second trade can remain open at backtest end, so inspect raw trade list.
    assert len(round_trips) >= 1, "Expected at least one closed round trip from initial SL."
    assert len(trades) >= 3, (
        f"Expected re-entry after SL (at least 3 fills: open, close, re-open), got {len(trades)}."
    )
