"""Robustness tests for the backtest engine."""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal

from laakhay.data.models.bar import Bar
from laakhay.quantlab.backtest import (
    BacktestConfig,
    BacktestEngine,
    BreakevenControl,
    FrequencyControl,
    MemDataFeed,
    Strategy,
    TradeDirectionMode,
)
from laakhay.quantlab.backtest.engine.oms import OrderManager
from laakhay.quantlab.backtest.models import OrderSide, OrderType, Signal


def _bars(closes: list[str | int]) -> list[Bar]:
    start = datetime(2024, 1, 1, 0, 0, 0)
    bars: list[Bar] = []
    for idx, close in enumerate(closes):
        c = Decimal(str(close))
        bars.append(
            Bar(
                timestamp=start + timedelta(hours=idx),
                open=c,
                high=c,
                low=c,
                close=c,
                volume=Decimal("100"),
            )
        )
    return bars


class OneShotEntryStrategy:
    """Emit one buy signal and then stay flat."""

    def __init__(
        self, *, size: str = "500", sl: str | Decimal | None = None, tp: str | Decimal | None = None
    ):
        self._fired = False
        self._size = size
        self._sl = sl
        self._tp = tp

    def prepare(self, symbol: str, timeframe: str) -> None:
        pass

    def required_lookback(self) -> int:
        return 2

    def on_bar(self, dataset, symbol: str, timeframe: str):  # noqa: ANN001
        if self._fired:
            return []
        self._fired = True
        return [
            Signal(
                symbol=symbol,
                side=OrderSide.BUY,
                type=OrderType.MARKET,
                size=self._size,
                sl=self._sl,
                tp=self._tp,
            )
        ]


class EnterThenExitStrategy:
    """Emit a buy once, then an explicit sell once."""

    def __init__(self, *, size: str = "900"):
        self._step = 0
        self._size = size

    def prepare(self, symbol: str, timeframe: str) -> None:
        pass

    def required_lookback(self) -> int:
        return 2

    def on_bar(self, dataset, symbol: str, timeframe: str):  # noqa: ANN001
        self._step += 1
        if self._step == 1:
            return [
                Signal(symbol=symbol, side=OrderSide.BUY, type=OrderType.MARKET, size=self._size)
            ]
        if self._step == 2:
            return [Signal(symbol=symbol, side=OrderSide.SELL, type=OrderType.MARKET)]
        return []


class SequenceSignalsStrategy:
    """Emit a fixed sequence of signal lists over successive on_bar calls."""

    def __init__(self, sequence: list[list[Signal]]):
        self._sequence = sequence
        self._idx = 0

    def prepare(self, symbol: str, timeframe: str) -> None:
        pass

    def required_lookback(self) -> int:
        return 2

    def on_bar(self, dataset, symbol: str, timeframe: str):  # noqa: ANN001
        if self._idx >= len(self._sequence):
            return []
        signals = self._sequence[self._idx]
        self._idx += 1
        return signals


class TestBacktestEngineRobustness:
    """Backtest behavior checks for safety and accounting."""

    def test_report_uses_equity_for_pnl_with_open_position(self):
        feed = MemDataFeed(_bars(["100", "110", "120"]), symbol="BTCUSDT", timeframe="1h")
        strategy = OneShotEntryStrategy(size="500")

        engine = BacktestEngine(initial_capital=1000)
        results = engine.run(strategy, feed)

        assert results["final_equity"] > results["final_capital"]
        assert results["pnl"] >= 0
        assert len(results["open_positions"]) == 1

    def test_sl_tp_double_trigger_fills_only_once(self):
        bars = _bars(["100", "100", "100"])
        # Third bar triggers both SL and TP after position is opened on bar 3 open.
        bars[2] = bars[2].model_copy(
            update={
                "high": Decimal("106"),
                "low": Decimal("94"),
                "close": Decimal("100"),
            }
        )

        feed = MemDataFeed(bars, symbol="BTCUSDT", timeframe="1h")
        strategy = OneShotEntryStrategy(size="1000", sl=Decimal("95"), tp=Decimal("105"))

        engine = BacktestEngine(initial_capital=5000)
        results = engine.run(strategy, feed)

        assert results["total_trades"] == 2
        assert len(results["open_positions"]) == 0
        exit_trade = engine.trades[-1]
        assert exit_trade.side == OrderSide.SELL
        assert exit_trade.price == Decimal("95")

    def test_next_open_mode_blocks_entry_and_discretionary_exit_on_same_bar(self):
        feed = MemDataFeed(_bars(["100", "100", "100"]), symbol="BTCUSDT", timeframe="1h")
        strategy = SequenceSignalsStrategy(
            [
                [
                    Signal(symbol="BTCUSDT", side=OrderSide.BUY, type=OrderType.MARKET, size="1000"),
                    Signal(symbol="BTCUSDT", side=OrderSide.SELL, type=OrderType.MARKET),
                ],
            ]
        )
        engine = BacktestEngine(
            initial_capital=5000,
            config=BacktestConfig(frequency=FrequencyControl(min_hold_bars=0)),
        )

        results = engine.run(strategy, feed)

        assert results["total_trades"] == 1
        assert len(results["open_positions"]) == 1

    def test_stop_loss_does_not_fill_without_stop_touch(self):
        bars = _bars(["100", "100", "100"])
        bars[1] = bars[1].model_copy(
            update={
                "high": Decimal("104"),
                "low": Decimal("99"),
                "close": Decimal("103"),
            }
        )

        feed = MemDataFeed(bars, symbol="BTCUSDT", timeframe="1h")
        strategy = OneShotEntryStrategy(size="1000", sl=Decimal("95"), tp=Decimal("105"))

        engine = BacktestEngine(initial_capital=5000)
        results = engine.run(strategy, feed)

        assert results["total_trades"] == 1
        assert len(results["open_positions"]) == 1

    def test_commission_and_slippage_reduce_capital(self):
        feed = MemDataFeed(_bars(["100", "100", "100", "100"]), symbol="BTCUSDT", timeframe="1h")
        strategy = EnterThenExitStrategy(size="900")
        engine = BacktestEngine(
            initial_capital=1000,
            commission_bps=10,
            slippage_bps=10,
        )

        results = engine.run(strategy, feed)

        assert results["total_trades"] == 2
        assert results["final_capital"] < Decimal("1000")
        assert results["pnl"] < 0

    def test_percent_sizing_with_commission_is_clipped_not_rejected(self):
        feed = MemDataFeed(_bars(["100", "100", "100"]), symbol="BTCUSDT", timeframe="1h")
        strategy = OneShotEntryStrategy(size="100%")
        engine = BacktestEngine(
            initial_capital=1000,
            commission_bps=10,
        )

        results = engine.run(strategy, feed)

        # 100% sizing with fees should still enter by clipping to max affordable qty.
        assert results["total_trades"] == 1
        assert len(results["open_positions"]) == 1

    def test_run_honors_start_datetime_for_execution_window(self):
        bars = _bars(["100", "100", "100", "100", "100"])
        start_dt = bars[3].timestamp
        feed = MemDataFeed(bars, symbol="BTCUSDT", timeframe="1h")
        strategy = OneShotEntryStrategy(size="100")
        engine = BacktestEngine(initial_capital=1000)

        results = engine.run(strategy, feed, start_dt=start_dt)

        assert results["total_trades"] == 1
        assert engine.trades[0].timestamp >= start_dt

    def test_ta_strategy_exposes_required_lookback(self):
        strategy = Strategy(entry_signal="sma(20) > sma(50)")

        assert strategy.required_lookback() >= 50

    def test_long_only_blocks_short_entries(self):
        feed = MemDataFeed(_bars(["100", "100", "100", "100"]), symbol="BTCUSDT", timeframe="1h")
        strategy = SequenceSignalsStrategy(
            [
                [Signal(symbol="BTCUSDT", side=OrderSide.SELL, type=OrderType.MARKET, size="500")],
                [],
            ]
        )
        engine = BacktestEngine(
            initial_capital=5000,
            config=BacktestConfig(direction_mode=TradeDirectionMode.LONG_ONLY),
        )

        results = engine.run(strategy, feed)

        assert results["total_trades"] == 0
        assert len(results["open_positions"]) == 0

    def test_short_only_allows_short_round_trip(self):
        feed = MemDataFeed(_bars(["100", "100", "95", "95"]), symbol="BTCUSDT", timeframe="1h")
        strategy = SequenceSignalsStrategy(
            [
                [Signal(symbol="BTCUSDT", side=OrderSide.SELL, type=OrderType.MARKET, size="1000")],
                [Signal(symbol="BTCUSDT", side=OrderSide.BUY, type=OrderType.MARKET)],
            ]
        )
        engine = BacktestEngine(
            initial_capital=10000,
            config=BacktestConfig(direction_mode=TradeDirectionMode.SHORT_ONLY),
        )

        results = engine.run(strategy, feed)

        assert results["total_trades"] == 2
        assert len(results["round_trips"]) == 1
        assert results["round_trips"][0]["direction"] == "short"

    def test_min_hold_bars_prevents_early_discretionary_exit(self):
        feed = MemDataFeed(_bars(["100", "100", "100", "101"]), symbol="BTCUSDT", timeframe="1h")
        strategy = SequenceSignalsStrategy(
            [
                [Signal(symbol="BTCUSDT", side=OrderSide.BUY, type=OrderType.MARKET, size="1000")],
                [Signal(symbol="BTCUSDT", side=OrderSide.SELL, type=OrderType.MARKET)],
            ]
        )
        engine = BacktestEngine(
            initial_capital=5000,
            config=BacktestConfig(frequency=FrequencyControl(min_hold_bars=2)),
        )

        results = engine.run(strategy, feed)

        assert results["total_trades"] == 1
        assert len(results["open_positions"]) == 1

    def test_cooldown_bars_blocks_immediate_reentry(self):
        feed = MemDataFeed(
            _bars(["100", "100", "101", "102", "103", "104"]), symbol="BTCUSDT", timeframe="1h"
        )
        strategy = SequenceSignalsStrategy(
            [
                [Signal(symbol="BTCUSDT", side=OrderSide.BUY, type=OrderType.MARKET, size="1000")],
                [Signal(symbol="BTCUSDT", side=OrderSide.SELL, type=OrderType.MARKET)],
                [Signal(symbol="BTCUSDT", side=OrderSide.BUY, type=OrderType.MARKET, size="1000")],
            ]
        )
        engine = BacktestEngine(
            initial_capital=5000,
            config=BacktestConfig(frequency=FrequencyControl(cooldown_bars=2)),
        )

        results = engine.run(strategy, feed)

        assert results["total_trades"] == 2
        assert len(results["open_positions"]) == 0

    def test_time_exit_closes_position_after_max_bars(self):
        feed = MemDataFeed(
            _bars(["100", "100", "100", "100", "100"]), symbol="BTCUSDT", timeframe="1h"
        )
        strategy = SequenceSignalsStrategy(
            [
                [Signal(symbol="BTCUSDT", side=OrderSide.BUY, type=OrderType.MARKET, size="1000")],
            ]
        )
        engine = BacktestEngine(
            initial_capital=5000,
            config=BacktestConfig(frequency=FrequencyControl(max_bars_in_trade=2)),
        )

        results = engine.run(strategy, feed)

        assert results["total_trades"] == 2
        assert len(results["round_trips"]) == 1
        assert len(results["open_positions"]) == 0
        assert results["round_trips"][0]["exit_reason"] == "TIME_EXIT"

    def test_breakeven_moves_stop_and_limits_loss(self):
        bars = _bars(["100", "100", "106", "100"])
        bars[3] = bars[3].model_copy(
            update={
                "high": Decimal("102"),
                "low": Decimal("99"),
                "close": Decimal("100"),
            }
        )
        feed = MemDataFeed(bars, symbol="BTCUSDT", timeframe="1h")
        strategy = SequenceSignalsStrategy(
            [
                [
                    Signal(
                        symbol="BTCUSDT",
                        side=OrderSide.BUY,
                        type=OrderType.MARKET,
                        size="1000",
                        sl="5%",
                    )
                ],
            ]
        )
        engine = BacktestEngine(
            initial_capital=5000,
            config=BacktestConfig(
                breakeven=BreakevenControl(enabled=True, trigger_rr=1.0, offset_bps=0)
            ),
        )

        results = engine.run(strategy, feed)

        assert results["total_trades"] == 2
        exit_trade = engine.trades[-1]
        assert exit_trade.side == OrderSide.SELL
        assert exit_trade.price == Decimal("100.70")

    def test_signal_execution_default_is_next_bar_open(self):
        bars = _bars(["100", "100", "110"])
        feed = MemDataFeed(bars, symbol="BTCUSDT", timeframe="1h")
        strategy = OneShotEntryStrategy(size="1000")
        engine = BacktestEngine(initial_capital=5000)

        engine.run(strategy, feed)

        assert len(engine.trades) == 1
        assert engine.trades[0].price == Decimal("110")

    def test_signal_execution_can_use_same_bar_close_mode(self):
        bars = _bars(["100", "100", "110"])
        feed = MemDataFeed(bars, symbol="BTCUSDT", timeframe="1h")
        strategy = OneShotEntryStrategy(size="1000")
        engine = BacktestEngine(
            initial_capital=5000,
            config=BacktestConfig(execute_signals_on_next_bar_open=False),
        )

        engine.run(strategy, feed)

        assert len(engine.trades) == 1
        assert engine.trades[0].price == Decimal("100")

    def test_report_contains_performance_and_trade_metrics(self):
        feed = MemDataFeed(_bars(["100", "100", "110", "110"]), symbol="BTCUSDT", timeframe="1h")
        strategy = SequenceSignalsStrategy(
            [
                [Signal(symbol="BTCUSDT", side=OrderSide.BUY, type=OrderType.MARKET, size="1000")],
                [Signal(symbol="BTCUSDT", side=OrderSide.SELL, type=OrderType.MARKET)],
            ]
        )
        engine = BacktestEngine(initial_capital=5000)

        results = engine.run(strategy, feed)

        assert "performance" in results
        assert "trade_metrics" in results
        assert results["performance"]["max_drawdown"] <= 0
        assert results["trade_metrics"]["total_round_trips"] == 1
        assert results["round_trips"][0]["entry_time"] is not None
        assert results["round_trips"][0]["exit_time"] is not None
        assert results["round_trips"][0]["exit_reason"] == "EXIT_SIGNAL"

    def test_report_contains_bar_diagnostics_payload(self):
        feed = MemDataFeed(_bars(["100", "100", "100"]), symbol="BTCUSDT", timeframe="1h")
        strategy = OneShotEntryStrategy(size="1000", sl=Decimal("95"), tp=Decimal("110"))
        engine = BacktestEngine(initial_capital=5000)

        results = engine.run(strategy, feed)

        diagnostics = results.get("bar_diagnostics", [])
        assert len(diagnostics) == 3
        required_keys = {
            "timestamp",
            "position_open",
            "entry_signal_eval",
            "exit_on_expression_eval",
            "exit_on_sl_eval",
            "exit_on_tp_eval",
            "exit_on_max_time_eval",
            "exit_eval",
        }
        assert required_keys.issubset(set(diagnostics[0].keys()))
        assert any(bool(point["entry_signal_eval"]) for point in diagnostics)

    def test_bar_diagnostics_flags_time_exit_evaluation(self):
        feed = MemDataFeed(_bars(["100", "100", "100", "100"]), symbol="BTCUSDT", timeframe="1h")
        strategy = SequenceSignalsStrategy(
            [
                [Signal(symbol="BTCUSDT", side=OrderSide.BUY, type=OrderType.MARKET, size="1000")],
                [],
            ]
        )
        engine = BacktestEngine(
            initial_capital=5000,
            config=BacktestConfig(frequency=FrequencyControl(max_bars_in_trade=1)),
        )

        results = engine.run(strategy, feed)

        diagnostics = results.get("bar_diagnostics", [])
        assert any(point["exit_on_max_time_eval"] for point in diagnostics)
        assert any(point["exit_eval"] for point in diagnostics)

    def test_stop_loss_exit_reason_is_tagged(self):
        bars = _bars(["100", "100", "100"])
        bars[2] = bars[2].model_copy(
            update={
                "high": Decimal("101"),
                "low": Decimal("94"),
                "close": Decimal("95"),
            }
        )
        feed = MemDataFeed(bars, symbol="BTCUSDT", timeframe="1h")
        strategy = OneShotEntryStrategy(size="1000", sl=Decimal("95"), tp=Decimal("110"))
        engine = BacktestEngine(initial_capital=5000)

        results = engine.run(strategy, feed)

        assert len(results["round_trips"]) == 1
        assert results["round_trips"][0]["exit_reason"] == "STOP_LOSS"
        assert results["round_trips"][0]["sl_price"] == 95.0
        assert results["round_trips"][0]["tp_price"] == 110.0
        assert len(results["trades"]) >= 1
        assert str(results["trades"][0]["sl_price"]) == "95"
        assert str(results["trades"][0]["tp_price"]) == "110"

    def test_short_stop_loss_exit_reason_is_tagged(self):
        bars = _bars(["100", "100", "100"])
        bars[2] = bars[2].model_copy(
            update={
                "high": Decimal("106"),
                "low": Decimal("99"),
                "close": Decimal("104"),
            }
        )
        feed = MemDataFeed(bars, symbol="BTCUSDT", timeframe="1h")
        strategy = SequenceSignalsStrategy(
            [
                [
                    Signal(
                        symbol="BTCUSDT",
                        side=OrderSide.SELL,
                        type=OrderType.MARKET,
                        size="1000",
                        sl=Decimal("105"),
                        tp=Decimal("95"),
                    )
                ],
            ]
        )
        engine = BacktestEngine(initial_capital=5000)

        results = engine.run(strategy, feed)

        assert len(results["round_trips"]) == 1
        assert results["round_trips"][0]["direction"] == "short"
        assert results["round_trips"][0]["exit_reason"] == "STOP_LOSS"
        assert results["round_trips"][0]["sl_price"] == 105.0
        assert results["round_trips"][0]["tp_price"] == 95.0

    def test_protective_exit_fallback_closes_when_matcher_misses(self, monkeypatch):
        bars = _bars(["100", "100", "100"])
        bars[2] = bars[2].model_copy(
            update={
                "high": Decimal("101"),
                "low": Decimal("94"),
                "close": Decimal("95"),
            }
        )
        feed = MemDataFeed(bars, symbol="BTCUSDT", timeframe="1h")
        strategy = OneShotEntryStrategy(size="1000", sl=Decimal("95"))
        monkeypatch.setattr(OrderManager, "match_orders", lambda self, bar, symbol: [])
        engine = BacktestEngine(initial_capital=5000)

        results = engine.run(strategy, feed)

        assert len(results["round_trips"]) == 1
        assert results["round_trips"][0]["exit_reason"] == "STOP_LOSS"
        assert len(results["open_positions"]) == 0

    def test_time_exit_closes_short_position_after_max_bars(self):
        feed = MemDataFeed(
            _bars(["100", "100", "100", "100", "100"]),
            symbol="BTCUSDT",
            timeframe="1h",
        )
        strategy = SequenceSignalsStrategy(
            [
                [Signal(symbol="BTCUSDT", side=OrderSide.SELL, type=OrderType.MARKET, size="1000")],
            ]
        )
        engine = BacktestEngine(
            initial_capital=5000,
            config=BacktestConfig(frequency=FrequencyControl(max_bars_in_trade=2)),
        )

        results = engine.run(strategy, feed)

        assert results["total_trades"] == 2
        assert len(results["round_trips"]) == 1
        assert len(results["open_positions"]) == 0
        assert results["round_trips"][0]["direction"] == "short"
        assert results["round_trips"][0]["exit_reason"] == "TIME_EXIT"

    def test_max_entries_per_day_blocks_excess_reentries(self):
        feed = MemDataFeed(
            _bars(["100", "100", "100", "100", "100", "100", "100"]),
            symbol="BTCUSDT",
            timeframe="1h",
        )
        strategy = SequenceSignalsStrategy(
            [
                [Signal(symbol="BTCUSDT", side=OrderSide.BUY, type=OrderType.MARKET, size="1000")],
                [Signal(symbol="BTCUSDT", side=OrderSide.SELL, type=OrderType.MARKET)],
                [Signal(symbol="BTCUSDT", side=OrderSide.BUY, type=OrderType.MARKET, size="1000")],
                [Signal(symbol="BTCUSDT", side=OrderSide.SELL, type=OrderType.MARKET)],
                [Signal(symbol="BTCUSDT", side=OrderSide.BUY, type=OrderType.MARKET, size="1000")],
            ]
        )
        engine = BacktestEngine(
            initial_capital=5000,
            config=BacktestConfig(frequency=FrequencyControl(max_entries_per_day=2)),
        )

        results = engine.run(strategy, feed)

        assert results["total_trades"] == 4
        assert len(results["round_trips"]) == 2
        assert len(results["open_positions"]) == 0

    def test_daily_loss_limit_blocks_new_entries_until_day_roll(self):
        bars = _bars(["100", "95", "95", "95", "95", "95"])
        bars[4] = bars[4].model_copy(update={"timestamp": datetime(2024, 1, 2, 0, 0, 0)})
        bars[5] = bars[5].model_copy(update={"timestamp": datetime(2024, 1, 2, 1, 0, 0)})
        feed = MemDataFeed(bars, symbol="BTCUSDT", timeframe="1h")
        strategy = SequenceSignalsStrategy(
            [
                [Signal(symbol="BTCUSDT", side=OrderSide.BUY, type=OrderType.MARKET, size="1000")],
                [Signal(symbol="BTCUSDT", side=OrderSide.SELL, type=OrderType.MARKET)],
                [Signal(symbol="BTCUSDT", side=OrderSide.BUY, type=OrderType.MARKET, size="1000")],
                [],
                [Signal(symbol="BTCUSDT", side=OrderSide.BUY, type=OrderType.MARKET, size="1000")],
            ]
        )
        engine = BacktestEngine(
            initial_capital=5000,
            config=BacktestConfig(daily_loss_limit_pct=1),
        )

        results = engine.run(strategy, feed)

        assert results["total_trades"] == 3
        assert len(results["open_positions"]) == 1
