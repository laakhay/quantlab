from __future__ import annotations

from typing import Any

from ..analytics import aggregate_batch_reports
from ..domain.enums import RunMode
from .schema import BatchBacktestReport, SingleBacktestReport, SymbolBacktestReport


def build_single_report(report: dict[str, Any]) -> SingleBacktestReport:
    """Wrap a legacy single-symbol report in the v2 report envelope."""
    return SingleBacktestReport(run_mode=RunMode.SINGLE, report=report)


def build_batch_independent_report(
    *,
    symbol_reports: list[SymbolBacktestReport],
    errors_by_symbol: dict[str, str] | None = None,
) -> BatchBacktestReport:
    """Build a normalized batch report with aggregate metrics."""
    errors = errors_by_symbol or {}
    aggregate = aggregate_batch_reports({item.symbol: item.report for item in symbol_reports})
    if errors:
        aggregate["completed_symbols"] = len(symbol_reports)
        aggregate["failed_symbols"] = len(errors)
    return BatchBacktestReport(
        run_mode=RunMode.BATCH_INDEPENDENT,
        symbol_reports=symbol_reports,
        aggregate=aggregate,
        errors_by_symbol=errors,
    )
