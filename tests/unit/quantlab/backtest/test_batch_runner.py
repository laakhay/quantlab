from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal

from laakhay.quantlab.backtest import BacktestEngine, BatchBacktestRunner, MemDataFeed, Strategy
from laakhay.quantlab.backtest.models import Bar


def _bars(start_price: str) -> list[Bar]:
    start = datetime(2024, 1, 1, 0, 0, 0)
    p0 = Decimal(start_price)
    return [
        Bar(timestamp=start, open=p0, high=p0, low=p0, close=p0, volume=Decimal("100")),
        Bar(
            timestamp=start + timedelta(hours=1),
            open=p0,
            high=p0 + Decimal("10"),
            low=p0,
            close=p0 + Decimal("10"),
            volume=Decimal("100"),
        ),
        Bar(
            timestamp=start + timedelta(hours=2),
            open=p0 + Decimal("10"),
            high=p0 + Decimal("20"),
            low=p0 + Decimal("10"),
            close=p0 + Decimal("20"),
            volume=Decimal("100"),
        ),
    ]


def test_batch_runner_independent_reports_and_aggregate():
    feeds = {
        "BTCUSDT": MemDataFeed(_bars("100"), symbol="BTCUSDT", timeframe="1h"),
        "ETHUSDT": MemDataFeed(_bars("200"), symbol="ETHUSDT", timeframe="1h"),
    }

    result = BatchBacktestRunner.run_batch_independent(
        feeds_by_symbol=feeds,
        strategy_factory=lambda _symbol: Strategy(entry_rule=lambda *_: True, required_lookback=1),
        engine_factory=lambda _symbol: BacktestEngine(initial_capital=1000),
    )

    assert result.run_mode.value == "batch_independent"
    assert len(result.symbol_reports) == 2
    assert result.aggregate["symbols"] == 2
    assert result.errors_by_symbol == {}
