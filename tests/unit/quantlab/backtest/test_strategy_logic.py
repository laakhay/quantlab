"""Tests for quantlab Strategy behavior."""

from __future__ import annotations

from laakhay.quantlab.backtest.models import OrderSide, Signal
from laakhay.quantlab.backtest.strategy.base import Strategy


class TestStrategyLogic:
    def test_on_bar_generates_entry_signal_when_entry_rule_true(self):
        strategy = Strategy(side="long", entry_rule=lambda _d, _s, _t: True)

        signals = strategy.on_bar(dataset=[], symbol="BTC", timeframe="1h")

        assert len(signals) == 1
        assert isinstance(signals[0], Signal)
        assert signals[0].side == OrderSide.BUY

    def test_on_bar_generates_exit_signal_when_exit_rule_true(self):
        strategy = Strategy(side="long", exit_rule=lambda _d, _s, _t: True)

        signals = strategy.on_bar(dataset=[], symbol="BTC", timeframe="1h")

        assert len(signals) == 1
        assert signals[0].side == OrderSide.SELL

    def test_on_bar_catches_rule_exceptions(self):
        strategy = Strategy(entry_rule=lambda _d, _s, _t: 1 / 0)
        signals = strategy.on_bar(dataset=[], symbol="BTC", timeframe="1h")
        assert signals == []

    def test_required_lookback_calculation(self):
        strategy = Strategy(required_lookback=10)
        assert strategy.required_lookback() == 10

    def test_expression_strings_infer_lookback_without_ta_dependency(self):
        strategy = Strategy(entry_signal="sma(20) > sma(50)")
        assert strategy.required_lookback() >= 50
