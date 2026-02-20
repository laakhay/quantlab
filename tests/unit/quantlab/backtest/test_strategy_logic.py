"""Tests for Strategy base class logic (Warmup, Error Handling)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
from laakhay.quantlab.backtest.strategy.base import Strategy
from laakhay.quantlab.backtest.models import Signal, OrderSide

class TestStrategyLogic:
    """Tests for Strategy.on_bar behavior."""

    def test_on_bar_catches_value_error_during_warmup(self):
        # Strategy init with no signals to avoid parsing
        strategy = Strategy()
        
        # Inject mock plan and evaluator
        strategy.entry_plan = MagicMock()
        
        # Mock Evaluator
        mock_evaluator = MagicMock()
        mock_evaluator.evaluate.side_effect = ValueError("Not enough data")
        strategy.evaluator = mock_evaluator
        
        # Call on_bar
        signals = strategy.on_bar(dataset=MagicMock(), symbol="BTC", timeframe="1h")
        
        # Should return empty list (no signals) and NOT raise
        assert signals == []

    def test_on_bar_handles_general_exception(self):
        strategy = Strategy()
        strategy.entry_plan = MagicMock()
        
        mock_evaluator = MagicMock()
        mock_evaluator.evaluate.side_effect = Exception("General Error")
        strategy.evaluator = mock_evaluator
        
        signals = strategy.on_bar(dataset=MagicMock(), symbol="BTC", timeframe="1h")
        
        assert signals == []

    def test_on_bar_generates_signal_on_true_eval(self):
        strategy = Strategy(side="long")
        # Inject plan so it checks entry
        strategy.entry_plan = MagicMock()
        
        mock_evaluator = MagicMock()
        mock_series = MagicMock()
        # Series values used: values[-1] -> True
        mock_series.values = [False, True]
        mock_series.__len__.return_value = 2
        mock_evaluator.evaluate.return_value = mock_series
        
        strategy.evaluator = mock_evaluator
        
        signals = strategy.on_bar(dataset=MagicMock(), symbol="BTC", timeframe="1h")
        
        assert len(signals) == 1
        assert isinstance(signals[0], Signal)
        assert signals[0].side == OrderSide.BUY

    def test_required_lookback_calculation(self):
        # Test that required_lookback property works
        strategy = Strategy()
        
        # If no plans, default is 2?
        # base.py: required_lookback() -> max(2, self._required_lookback)
        assert strategy.required_lookback() >= 2
        
        # If we manually set _required_lookback
        strategy._required_lookback = 10
        assert strategy.required_lookback() == 10
