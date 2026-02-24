from __future__ import annotations

from statistics import fmean
from typing import Any


def aggregate_batch_reports(reports_by_symbol: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Aggregate high-level metrics from independent per-symbol reports."""
    if not reports_by_symbol:
        return {
            "symbols": 0,
            "completed_symbols": 0,
            "avg_total_return": 0.0,
            "median_like_total_return": 0.0,
            "avg_max_drawdown": 0.0,
            "avg_sharpe_ratio": None,
            "total_trades": 0,
        }

    total_returns: list[float] = []
    max_drawdowns: list[float] = []
    sharpes: list[float] = []
    total_trades = 0
    for report in reports_by_symbol.values():
        perf = report.get("performance", {})
        total_returns.append(float(perf.get("total_return", 0.0)))
        max_drawdowns.append(float(perf.get("max_drawdown", 0.0)))
        sharpe = perf.get("sharpe_ratio")
        if sharpe is not None:
            sharpes.append(float(sharpe))
        total_trades += int(report.get("total_trades", 0) or 0)

    sorted_returns = sorted(total_returns)
    mid = len(sorted_returns) // 2
    if len(sorted_returns) % 2 == 1:
        median_like = sorted_returns[mid]
    else:
        median_like = (sorted_returns[mid - 1] + sorted_returns[mid]) / 2

    return {
        "symbols": len(reports_by_symbol),
        "completed_symbols": len(reports_by_symbol),
        "avg_total_return": fmean(total_returns) if total_returns else 0.0,
        "median_like_total_return": median_like,
        "avg_max_drawdown": fmean(max_drawdowns) if max_drawdowns else 0.0,
        "avg_sharpe_ratio": fmean(sharpes) if sharpes else None,
        "total_trades": total_trades,
    }
