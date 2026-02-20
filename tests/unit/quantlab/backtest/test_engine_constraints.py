"""Tests for backtest engine constraints (Frequency, Risk, etc.)."""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal

from laakhay.data.models.bar import Bar
from laakhay.quantlab.backtest import (
    BacktestConfig,
    BacktestEngine,
    FrequencyControl,
    MemDataFeed,
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
    """Emit a fixed sequence of signal lists over successive on_bar calls."""

    def __init__(self, sequence: list[list[Signal]]):
        self._sequence = sequence
        self._idx = 0

    def prepare(self, symbol: str, timeframe: str) -> None:
        pass

    def required_lookback(self) -> int:
        return 2

    def on_bar(self, dataset, symbol: str, timeframe: str):
        if self._idx >= len(self._sequence):
            return []
        signals = self._sequence[self._idx]
        self._idx += 1
        return signals


class TestEngineConstraints:
    """Tests for FrequencyControl and Risk Controls."""

    def test_max_entries_per_day_resets_next_day(self):
        start_d1 = datetime(2024, 1, 1, 0, 0, 0)
        start_d2 = datetime(2024, 1, 2, 0, 0, 0)
        
        # Warmup (2) + D1 (4) + D2 (2) + EndPadding (1) = 9 bars
        # Indices: 0,1 (Warmup). 2,3,4,5 (D1). 6,7 (D2). 8 (Padding).
        
        bars = _bars(["100"] * 9, start_time=start_d1)
        
        # Adjust timestamps for D2 (Indices 6, 7)
        # Note: _bars auto-increments hours. 
        # Index 0=00h. Index 6=06h. 
        # We force Index 6 to D2-00h.
        bars[6] = bars[6].model_copy(update={"timestamp": start_d2})
        bars[7] = bars[7].model_copy(update={"timestamp": start_d2 + timedelta(hours=1)})
        bars[8] = bars[8].model_copy(update={"timestamp": start_d2 + timedelta(hours=2)})

        strategy = SequenceSignalsStrategy([
            # Warmup (Indices 0, 1) -> Skipped by Strategy logic (lookback=2).
            # Index 2 (D1-00h equivalent for signal logic): Emit Buy.
            # Exec at Index 3 Open.
            [Signal(symbol="BTCUSDT", side=OrderSide.BUY, type=OrderType.MARKET, size="100")],
            
            # Index 3: Emit Sell. Exec at Index 4 Open.
            [Signal(symbol="BTCUSDT", side=OrderSide.SELL, type=OrderType.MARKET)],
            
            # Index 4: Emit Buy. Exec at Index 5 Open.
            [Signal(symbol="BTCUSDT", side=OrderSide.BUY, type=OrderType.MARKET, size="100")],
            
            # Index 5: Emit Sell. Exec at Index 6 Open (D2-00h).
            [Signal(symbol="BTCUSDT", side=OrderSide.SELL, type=OrderType.MARKET)],
            
            # Index 6 (D2-00h): Emit Buy. Exec at Index 7 Open.
            [Signal(symbol="BTCUSDT", side=OrderSide.BUY, type=OrderType.MARKET, size="100")],
            
            # Index 7: Empty.
            [],
        ])
        
        feed = MemDataFeed(bars, symbol="BTCUSDT", timeframe="1h")

        engine = BacktestEngine(
            initial_capital=10000,
            config=BacktestConfig(frequency=FrequencyControl(max_entries_per_day=2))
        )
        results = engine.run(strategy, feed)

        # Expected:
        # Trade 1: Entry T3, Exit T4.
        # Trade 2: Entry T5, Exit T6.
        # Trade 3: Entry T7.
        # Total Trades = 5.
        
        assert results["total_trades"] == 5
        assert len(results["round_trips"]) == 2
        assert len(results["open_positions"]) == 1

    def test_daily_loss_limit_resets_next_day(self):
        start_d1 = datetime(2024, 1, 1, 0, 0, 0)
        start_d2 = datetime(2024, 1, 2, 0, 0, 0)
        
        # Warmup(2) + D1(3) + D2(2) + Pad(1) = 8 bars
        # Values: 100, 100 | 100, 90, 100 | 100, 110 | 110
        prices = ["100", "100", "100", "90", "100", "100", "110", "110"]
        bars = _bars(prices, start_time=start_d1)
        
        # D1 Indices: 2, 3, 4.
        # D2 Indices: 5, 6.
        bars[5] = bars[5].model_copy(update={"timestamp": start_d2})
        bars[6] = bars[6].model_copy(update={"timestamp": start_d2 + timedelta(hours=1)})
        bars[7] = bars[7].model_copy(update={"timestamp": start_d2 + timedelta(hours=2)})
        
        feed = MemDataFeed(bars, symbol="BTCUSDT", timeframe="1h")
        
        strategy = SequenceSignalsStrategy([
            # Index 2: Buy. Exec Index 3 (100 -> 90).
            [Signal(symbol="BTCUSDT", side=OrderSide.BUY, type=OrderType.MARKET, size="1000")],
            # Index 3: Sell. Exec Index 4 (90). Loss.
            [Signal(symbol="BTCUSDT", side=OrderSide.SELL, type=OrderType.MARKET)],
            # Index 4: Empty.
            [],
            # Index 5 (D2): Buy. Exec Index 6.
            [Signal(symbol="BTCUSDT", side=OrderSide.BUY, type=OrderType.MARKET, size="1000")],
            # Index 6: Sell. Exec Index 7.
            [Signal(symbol="BTCUSDT", side=OrderSide.SELL, type=OrderType.MARKET)],
        ])
        
        engine = BacktestEngine(
            initial_capital=1000,
            config=BacktestConfig(daily_loss_limit_pct=5.0)
        )
        
        results = engine.run(strategy, feed)
        
        assert results["total_trades"] == 4
        # Trip 1 (Loss). Trip 2 (Profit).
        assert len(results["round_trips"]) == 2
        assert len(results["open_positions"]) == 0

    def test_cooldown_bars_exact_boundary(self):
        # Warmup(2) + 5 Bars + Pad(1) = 8
        bars = _bars(["100"] * 8)
        feed = MemDataFeed(bars, symbol="BTCUSDT", timeframe="1h")
        
        strategy = SequenceSignalsStrategy([
            # Index 2: Buy. Exec 3.
            [Signal(symbol="BTCUSDT", side=OrderSide.BUY, type=OrderType.MARKET, size="100")],
            # Index 3: Sell. Exec 4.
            [Signal(symbol="BTCUSDT", side=OrderSide.SELL, type=OrderType.MARKET)],
            # Index 4: Buy. Exec 5?
            # Cooldown check at 5. Last Exit at 4. (5-4)=1. Cooldown=1. Block.
            [Signal(symbol="BTCUSDT", side=OrderSide.BUY, type=OrderType.MARKET, size="100")],
            # Index 5: Buy. Exec 6.
            # (6-4)=2. Allow.
            [Signal(symbol="BTCUSDT", side=OrderSide.BUY, type=OrderType.MARKET, size="100")],
            [],
        ])
        
        engine = BacktestEngine(
            initial_capital=10000,
            config=BacktestConfig(
                frequency=FrequencyControl(cooldown_bars=1, min_hold_bars=0)
            )
        )
        
        results = engine.run(strategy, feed)
        
        # Trade 1: Entry, Exit. (2 trades)
        # Trade 2: Entry (Open). (1 trade)
        assert results["total_trades"] == 3
        assert len(results["round_trips"]) == 1
        assert len(results["open_positions"]) == 1

    def test_allow_entry_same_bar_as_exit_disabled(self):
        # Warmup(2) + 3 Bars = 5
        bars = _bars(["100"] * 5)
        feed = MemDataFeed(bars, symbol="BTCUSDT", timeframe="1h")
        
        # Index 2: Buy. Exec 3.
        # Index 3: Sell, Buy. Exec 4.
        # 4: Sell executes. Entry executes?
        
        strategy = SequenceSignalsStrategy([
            [Signal(symbol="BTCUSDT", side=OrderSide.BUY, type=OrderType.MARKET, size="100")],
            [
                Signal(symbol="BTCUSDT", side=OrderSide.SELL, type=OrderType.MARKET),
                Signal(symbol="BTCUSDT", side=OrderSide.BUY, type=OrderType.MARKET, size="100"),
            ]
        ])
        
        engine = BacktestEngine(
            initial_capital=10000,
            config=BacktestConfig(
                allow_entry_same_bar_as_exit=False,
                frequency=FrequencyControl(min_hold_bars=0)
            )
        )
        
        results = engine.run(strategy, feed)
        
        # Entry 1, Exit 1. Entry 2 Blocked.
        assert results["total_trades"] == 2
        assert len(results["round_trips"]) == 1
        assert len(results["open_positions"]) == 0

