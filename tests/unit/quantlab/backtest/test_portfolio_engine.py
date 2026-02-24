from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from laakhay.quantlab.backtest import (
    BacktestConfig,
    BatchBacktestRunner,
    LeverageControl,
    MultiAssetMemFeed,
    PortfolioBacktestEngine,
    PortfolioControl,
    RunMode,
)
from laakhay.quantlab.backtest.execution.risk import PortfolioExposure, PortfolioRiskChecker
from laakhay.quantlab.backtest.models import Bar, OrderSide, OrderType, Signal


def _bar(ts: datetime, close: str, symbol: str) -> Bar:
    c = Decimal(close)
    return Bar(
        timestamp=ts,
        open=c,
        high=c,
        low=c,
        close=c,
        volume=Decimal("100"),
        symbol=symbol,
    )


class EntryThenExitStrategy:
    def __init__(self) -> None:
        self._step = 0

    def prepare(self, symbol: str, timeframe: str) -> None:
        return

    def required_lookback(self) -> int:
        return 1

    def on_bar(self, dataset, symbol: str, timeframe: str):
        self._step += 1
        if self._step == 1:
            return [Signal(symbol=symbol, side=OrderSide.BUY, type=OrderType.MARKET, size="100")]
        if self._step == 2:
            return [Signal(symbol=symbol, side=OrderSide.SELL, type=OrderType.MARKET)]
        return []


class EnterOnceStrategy:
    def __init__(self) -> None:
        self._entered = False

    def prepare(self, symbol: str, timeframe: str) -> None:
        return

    def required_lookback(self) -> int:
        return 1

    def on_bar(self, dataset, symbol: str, timeframe: str):
        if self._entered:
            return []
        self._entered = True
        return [Signal(symbol=symbol, side=OrderSide.BUY, type=OrderType.MARKET, size="100")]


def test_portfolio_engine_shared_capital_basic_roundtrip():
    t0 = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
    feed = MultiAssetMemFeed(
        {
            "BTCUSDT": [
                _bar(t0, "100", "BTCUSDT"),
                _bar(t0 + timedelta(hours=1), "110", "BTCUSDT"),
            ]
        },
        timeframe="1h",
    )
    engine = PortfolioBacktestEngine(
        initial_capital=1000,
        config=BacktestConfig(run_mode=RunMode.PORTFOLIO_SHARED),
    )

    report = BatchBacktestRunner.run_portfolio_shared(
        feed=feed,
        strategy_factory=lambda _symbol: EntryThenExitStrategy(),
        engine=engine,
    )

    assert report["run_mode"] == "portfolio_shared"
    assert report["total_trades"] >= 2
    assert report["portfolio"]["open_positions"] == 0
    assert report["final_equity"] > Decimal("1000")


def test_portfolio_engine_respects_max_open_positions():
    t0 = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
    feed = MultiAssetMemFeed(
        {
            "BTCUSDT": [
                _bar(t0, "100", "BTCUSDT"),
                _bar(t0 + timedelta(hours=1), "100", "BTCUSDT"),
            ],
            "ETHUSDT": [
                _bar(t0, "200", "ETHUSDT"),
                _bar(t0 + timedelta(hours=1), "200", "ETHUSDT"),
            ],
        },
        timeframe="1h",
    )
    config = BacktestConfig(
        run_mode=RunMode.PORTFOLIO_SHARED,
        portfolio=PortfolioControl(max_open_positions=1),
    )
    engine = PortfolioBacktestEngine(initial_capital=1000, config=config)

    report = BatchBacktestRunner.run_portfolio_shared(
        feed=feed,
        strategy_factory=lambda _symbol: EntryThenExitStrategy(),
        engine=engine,
    )

    reasons = {item["reason"] for item in report["portfolio"]["rejections"]}
    assert "max_open_positions" in reasons


def test_portfolio_engine_rejects_max_symbol_weight():
    checker = PortfolioRiskChecker()
    config = BacktestConfig(
        run_mode=RunMode.PORTFOLIO_SHARED,
        portfolio=PortfolioControl(max_symbol_weight_pct=50),
    )
    reason = checker.check_open(
        config=config,
        equity=Decimal("1000"),
        current=PortfolioExposure(
            gross_notional=Decimal("0"),
            net_notional=Decimal("0"),
            by_symbol_notional={},
        ),
        symbol="BTCUSDT",
        side_sign=1,
        notional=Decimal("600"),
    )
    assert reason == "max_symbol_weight"


def test_portfolio_engine_rejects_max_gross_exposure():
    t0 = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
    feed = MultiAssetMemFeed(
        {
            "BTCUSDT": [_bar(t0, "100", "BTCUSDT")],
        },
        timeframe="1h",
    )
    config = BacktestConfig(
        run_mode=RunMode.PORTFOLIO_SHARED,
        portfolio=PortfolioControl(max_gross_exposure_pct=80),
    )
    engine = PortfolioBacktestEngine(initial_capital=1000, config=config)

    report = BatchBacktestRunner.run_portfolio_shared(
        feed=feed,
        strategy_factory=lambda _symbol: EnterOnceStrategy(),
        engine=engine,
    )

    reasons = [item["reason"] for item in report["portfolio"]["rejections"]]
    assert "max_gross_exposure" in reasons
    assert report["total_trades"] == 0


def test_portfolio_engine_rejects_max_net_exposure():
    t0 = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
    feed = MultiAssetMemFeed(
        {
            "BTCUSDT": [_bar(t0, "100", "BTCUSDT")],
        },
        timeframe="1h",
    )
    config = BacktestConfig(
        run_mode=RunMode.PORTFOLIO_SHARED,
        portfolio=PortfolioControl(max_net_exposure_pct=70),
    )
    engine = PortfolioBacktestEngine(initial_capital=1000, config=config)

    report = BatchBacktestRunner.run_portfolio_shared(
        feed=feed,
        strategy_factory=lambda _symbol: EnterOnceStrategy(),
        engine=engine,
    )

    reasons = [item["reason"] for item in report["portfolio"]["rejections"]]
    assert "max_net_exposure" in reasons
    assert report["total_trades"] == 0


def test_portfolio_engine_rejects_maintenance_margin_breach():
    t0 = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
    feed = MultiAssetMemFeed(
        {
            "BTCUSDT": [_bar(t0, "100", "BTCUSDT")],
            "ETHUSDT": [_bar(t0, "200", "ETHUSDT")],
        },
        timeframe="1h",
    )
    config = BacktestConfig(
        run_mode=RunMode.PORTFOLIO_SHARED,
        portfolio=PortfolioControl(
            max_open_positions=2,
            max_symbol_weight_pct=200,
            max_gross_exposure_pct=200,
            max_net_exposure_pct=200,
        ),
        leverage=LeverageControl(enabled=True, leverage=2, maintenance_margin_ratio=0.9),
    )
    engine = PortfolioBacktestEngine(initial_capital=1000, config=config)

    report = BatchBacktestRunner.run_portfolio_shared(
        feed=feed,
        strategy_factory=lambda _symbol: EnterOnceStrategy(),
        engine=engine,
    )

    reasons = [item["reason"] for item in report["portfolio"]["rejections"]]
    assert "maintenance_margin_breach" in reasons
    assert report["total_trades"] == 1
