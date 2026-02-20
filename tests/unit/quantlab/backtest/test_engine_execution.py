"""Tests for backtest engine execution logic (Signals, Gaps, EoD)."""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal

from laakhay.data.models.bar import Bar
from laakhay.quantlab.backtest import (
    BacktestConfig,
    BacktestEngine,
    MemDataFeed,
    FrequencyControl,
)
from laakhay.quantlab.backtest.models import OrderSide, OrderType, Signal

def _bars(closes: list[str | int], start_time: datetime | None = None) -> list[Bar]:
    if start_time is None:
        start_time = datetime(2024, 1, 1, 0, 0, 0)
    bars: list[Bar] = []
    for idx, close in enumerate(closes):
        c = Decimal(str(close))
        bars.append(
            Bar(
                timestamp=start_time + timedelta(hours=idx),
                open=c,
                high=c,
                low=c,
                close=c,
                volume=Decimal("100"),
            )
        )
    return bars

class SequenceSignalsStrategy:
    """Emit a fixed sequence of signal lists."""
    def __init__(self, sequence: list[list[Signal]]):
        self._sequence = sequence
        self._idx = 0

    def prepare(self, symbol, timeframe): pass
    def required_lookback(self): return 1

    def on_bar(self, dataset, symbol, timeframe):
        if self._idx >= len(self._sequence):
            return []
        signals = self._sequence[self._idx]
        self._idx += 1
        return signals

class TestEngineExecution:
    """Tests for signal execution, priority, and data edge cases."""

    def test_end_of_data_signals_dropped(self):
        # Bars: T0, T1, T2, T3 (Pad).
        # Need 4 bars because:
        # T0 (Skip). T1 (Buy->T2). T2 (Sell->T3). 
        bars = _bars(["100"] * 4) 
        feed = MemDataFeed(bars, symbol="BTCUSDT", timeframe="1h")
        
        # T0: Skip.
        # T1: Emit Buy. Exec T2. (Entry exists).
        # T2: Emit Sell. Exec T3. (Exit exists).
        # T3: Emit Buy. Exec T4? (Missing).
        strategy = SequenceSignalsStrategy([
            [Signal(symbol="BTCUSDT", side=OrderSide.BUY, type=OrderType.MARKET, size="100")],
            [Signal(symbol="BTCUSDT", side=OrderSide.SELL, type=OrderType.MARKET)],
            [Signal(symbol="BTCUSDT", side=OrderSide.BUY, type=OrderType.MARKET, size="100")],
        ])
        
        engine = BacktestEngine(
            initial_capital=10000,
            config=BacktestConfig(execute_signals_on_next_bar_open=True)
        )
        results = engine.run(strategy, feed)
        
        # Expect: Entry (T1), Exit (T2). 
        # Last Buy signal (at T2) dropped due to EoD.
        assert results["total_trades"] == 2
        assert len(results["round_trips"]) == 1
        assert len(results["open_positions"]) == 0

    def test_signal_priority_exit_before_entry(self):
        # T0, T1, T2, T3 (Pad).
        bars = _bars(["100", "100", "100", "100"])
        feed = MemDataFeed(bars, symbol="BTCUSDT", timeframe="1h")
        
        # T0: Buy -> Exec T1.
        # T1: [Sell, Buy] -> Exec T2.
        # T2: Empty.
        
        strategy = SequenceSignalsStrategy([
            [Signal(symbol="BTCUSDT", side=OrderSide.BUY, type=OrderType.MARKET, size="100")],
            [   # Priority Check:
                Signal(symbol="BTCUSDT", side=OrderSide.SELL, type=OrderType.MARKET),
                Signal(symbol="BTCUSDT", side=OrderSide.BUY, type=OrderType.MARKET, size="100")
            ],
            []
        ])
        
        engine = BacktestEngine(
            initial_capital=10000,
            config=BacktestConfig(
                allow_entry_same_bar_as_exit=True,
                frequency=FrequencyControl(min_hold_bars=0)
            )
        )
        results = engine.run(strategy, feed)
        
        # Expect: 
        # T1: Buy Exec.
        # T2: Sell Exec (Close). Buy Exec (Open).
        # Total 3 trades.
        assert results["total_trades"] == 3
        assert len(results["round_trips"]) == 1
        assert len(results["open_positions"]) == 1

    def test_gap_excludes_time_constraint(self):
        start = datetime(2024, 1, 1, 0, 0)
        # Bar 0: 00:00
        # Bar 1: 01:00
        # Bar 2: 05:00 (Gap).
        # Bar 3: 06:00
        # Bar 4: 07:00 (Pad execution for T3 signal)
        
        bars = [
            Bar(timestamp=start, open=100, high=100, low=100, close=100, volume=100, symbol="BTC", interval="1h"),
            Bar(timestamp=start+timedelta(hours=1), open=100, high=100, low=100, close=100, volume=100, symbol="BTC", interval="1h"),
            Bar(timestamp=start+timedelta(hours=5), open=100, high=100, low=100, close=100, volume=100, symbol="BTC", interval="1h"),
            Bar(timestamp=start+timedelta(hours=6), open=100, high=100, low=100, close=100, volume=100, symbol="BTC", interval="1h"),
            Bar(timestamp=start+timedelta(hours=7), open=100, high=100, low=100, close=100, volume=100, symbol="BTC", interval="1h"),
        ]
        
        feed = MemDataFeed(bars, symbol="BTC", timeframe="1h")
        
        # Index 0 (00h): Buy. Exec Index 1.
        # Index 1 (01h): Sell. Exec Index 2. (Last Exit Index = 2).
        # Index 2 (05h): Buy. Exec Index 3?
        # Entry Check at Index 3 (06h).
        # Last Exit at Index 2 (05h).
        # Delta = 3 - 2 = 1.
        # Cooldown = 2.
        # 1 <= 2. True. Blocked.
        
        strategy = SequenceSignalsStrategy([
            [Signal(symbol="BTC", side=OrderSide.BUY, type=OrderType.MARKET, size="100")],
            [Signal(symbol="BTC", side=OrderSide.SELL, type=OrderType.MARKET)],
            [Signal(symbol="BTC", side=OrderSide.BUY, type=OrderType.MARKET, size="100")],
            [],
            [],
        ])
        
        engine = BacktestEngine(
            initial_capital=10000,
            config=BacktestConfig(
                frequency=FrequencyControl(cooldown_bars=2, min_hold_bars=0)
            )
        )
        results = engine.run(strategy, feed)
        
        # Expect: Entry 1, Exit 1. Entry 2 Blocked.
        assert results["total_trades"] == 2
        assert len(results["round_trips"]) == 1
        assert len(results["open_positions"]) == 0
