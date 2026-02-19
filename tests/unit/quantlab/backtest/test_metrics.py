"""Tests for backtest metrics helpers."""

from __future__ import annotations

import pytest

from laakhay.quantlab.backtest.metrics import (
    RoundTripTrade,
    compute_max_drawdown,
    compute_performance_metrics,
    compute_trade_metrics,
    infer_periods_per_year,
)


class TestPerformanceMetrics:
    def test_max_drawdown(self):
        equity = [100.0, 120.0, 110.0, 130.0, 100.0]

        max_dd = compute_max_drawdown(equity)

        assert max_dd == pytest.approx((100.0 / 130.0) - 1.0)

    def test_compute_performance_metrics(self):
        equity = [100.0, 110.0, 105.0, 120.0]

        metrics = compute_performance_metrics(equity, periods_per_year=252, risk_free_rate=0.0)

        assert metrics.total_return == pytest.approx(0.2)
        assert metrics.max_drawdown < 0
        assert metrics.periods == 3

    def test_infer_periods_per_year(self):
        assert infer_periods_per_year("1h") == 365 * 24
        assert infer_periods_per_year("1d") == 365
        assert infer_periods_per_year("invalid", fallback=252) == 252


class TestTradeMetrics:
    def test_trade_metrics(self):
        trades = [
            RoundTripTrade(
                symbol="BTCUSDT",
                direction="long",
                qty=1.0,
                entry_price=100.0,
                exit_price=110.0,
                pnl=10.0,
                pnl_pct=0.1,
                holding_seconds=3600.0,
                holding_bars=1,
            ),
            RoundTripTrade(
                symbol="BTCUSDT",
                direction="short",
                qty=1.0,
                entry_price=110.0,
                exit_price=120.0,
                pnl=-10.0,
                pnl_pct=-0.090909,
                holding_seconds=7200.0,
                holding_bars=2,
            ),
        ]

        metrics = compute_trade_metrics(trades)

        assert metrics.total_round_trips == 2
        assert metrics.win_rate == pytest.approx(0.5)
        assert metrics.profit_factor == pytest.approx(1.0)
        assert metrics.expectancy == pytest.approx(0.0)
        assert metrics.avg_holding_bars == pytest.approx(1.5)
