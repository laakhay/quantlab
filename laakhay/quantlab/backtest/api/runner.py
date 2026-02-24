from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..execution import BacktestEngine, PortfolioBacktestEngine
from ..feed import MemDataFeed, MultiAssetMemFeed
from ..report import BatchBacktestReport, SymbolBacktestReport, build_batch_independent_report


class BatchBacktestRunner:
    """Run single or independent multi-symbol backtests with shared semantics."""

    @staticmethod
    def run_single(
        strategy: Any,
        feed: MemDataFeed,
        engine: BacktestEngine,
        *,
        start_dt: Any | None = None,
        end_dt: Any | None = None,
    ) -> dict[str, Any]:
        return engine.run(strategy, feed, start_dt=start_dt, end_dt=end_dt)

    @staticmethod
    def run_batch_independent(
        *,
        feeds_by_symbol: dict[str, MemDataFeed],
        strategy_factory: Callable[[str], Any],
        engine_factory: Callable[[str], BacktestEngine],
        start_dt: Any | None = None,
        end_dt: Any | None = None,
    ) -> BatchBacktestReport:
        symbol_reports: list[SymbolBacktestReport] = []
        errors_by_symbol: dict[str, str] = {}
        for symbol in sorted(feeds_by_symbol.keys()):
            feed = feeds_by_symbol[symbol]
            try:
                strategy = strategy_factory(symbol)
                engine = engine_factory(symbol)
                report = engine.run(strategy, feed, start_dt=start_dt, end_dt=end_dt)
                symbol_reports.append(
                    SymbolBacktestReport(symbol=symbol, timeframe=feed.timeframe, report=report)
                )
            except Exception as exc:  # pragma: no cover - defensive batch isolation
                errors_by_symbol[symbol] = str(exc)
        return build_batch_independent_report(
            symbol_reports=symbol_reports,
            errors_by_symbol=errors_by_symbol,
        )

    @staticmethod
    def run_portfolio_shared(
        *,
        feed: MultiAssetMemFeed,
        strategy_factory: Callable[[str], Any],
        engine: PortfolioBacktestEngine,
        start_dt: Any | None = None,
        end_dt: Any | None = None,
    ) -> dict[str, Any]:
        return engine.run(
            strategy_factory=strategy_factory,
            feed=feed,
            start_dt=start_dt,
            end_dt=end_dt,
        )
