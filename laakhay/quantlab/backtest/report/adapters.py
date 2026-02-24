from __future__ import annotations

from typing import Any

from .schema import BatchBacktestReport, SingleBacktestReport


def single_report_to_dict(data: SingleBacktestReport) -> dict[str, Any]:
    """Serialize single report envelope for external consumers."""
    return {
        "run_mode": data.run_mode.value,
        "report": data.report,
    }


def batch_report_to_dict(data: BatchBacktestReport) -> dict[str, Any]:
    """Serialize batch report envelope for external consumers."""
    return {
        "run_mode": data.run_mode.value,
        "aggregate": data.aggregate,
        "symbol_reports": [
            {"symbol": item.symbol, "timeframe": item.timeframe, "report": item.report}
            for item in data.symbol_reports
        ],
        "errors_by_symbol": data.errors_by_symbol,
    }


def batch_report_to_legacy_dict(data: BatchBacktestReport) -> dict[str, Any]:
    """Legacy-compatible aggregate payload for consumers that do not handle envelopes yet."""
    return {
        "aggregate": data.aggregate,
        "reports_by_symbol": {item.symbol: item.report for item in data.symbol_reports},
        "errors_by_symbol": data.errors_by_symbol,
    }
