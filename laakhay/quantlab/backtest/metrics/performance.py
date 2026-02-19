from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal

Number = int | float | Decimal


def _to_float_series(values: Sequence[Number]) -> list[float]:
    return [float(v) for v in values]


def compute_period_returns(equity_values: Sequence[Number]) -> list[float]:
    """Compute simple period-over-period returns from equity values."""
    values = _to_float_series(equity_values)
    returns: list[float] = []
    for prev, current in zip(values, values[1:], strict=False):
        if prev <= 0:
            continue
        returns.append((current / prev) - 1.0)
    return returns


def compute_drawdown_series(equity_values: Sequence[Number]) -> list[float]:
    """Compute drawdown series (0 to negative values)."""
    values = _to_float_series(equity_values)
    if not values:
        return []

    peak = values[0]
    drawdowns: list[float] = []
    for value in values:
        if value > peak:
            peak = value
        if peak <= 0:
            drawdowns.append(0.0)
            continue
        drawdowns.append((value / peak) - 1.0)
    return drawdowns


def compute_max_drawdown(equity_values: Sequence[Number]) -> float:
    """Compute maximum drawdown as a negative fraction."""
    drawdowns = compute_drawdown_series(equity_values)
    if not drawdowns:
        return 0.0
    return min(drawdowns)


def compute_annualized_return(
    equity_values: Sequence[Number], periods_per_year: int
) -> float | None:
    """Compute annualized return from start/end equity and number of periods."""
    values = _to_float_series(equity_values)
    if len(values) < 2:
        return None
    start = values[0]
    end = values[-1]
    if start <= 0:
        return None
    periods = len(values) - 1
    exponent = periods_per_year / periods
    return (end / start) ** exponent - 1.0


def compute_annualized_volatility(
    period_returns: Sequence[float], periods_per_year: int
) -> float | None:
    """Compute annualized volatility using sample standard deviation."""
    if len(period_returns) < 2:
        return None
    stdev = statistics.stdev(period_returns)
    return stdev * math.sqrt(periods_per_year)


def compute_sharpe_ratio(
    period_returns: Sequence[float],
    periods_per_year: int,
    risk_free_rate: float = 0.0,
) -> float | None:
    """Compute annualized Sharpe ratio from period returns."""
    if len(period_returns) < 2:
        return None

    rf_per_period = (1.0 + risk_free_rate) ** (1.0 / periods_per_year) - 1.0
    excess = [r - rf_per_period for r in period_returns]
    stdev = statistics.stdev(excess)
    if stdev == 0:
        return None
    mean_excess = statistics.fmean(excess)
    return (mean_excess / stdev) * math.sqrt(periods_per_year)


@dataclass(frozen=True)
class PerformanceMetrics:
    """Backtest performance summary metrics."""

    total_return: float
    annualized_return: float | None
    annualized_volatility: float | None
    sharpe_ratio: float | None
    max_drawdown: float
    periods: int

    def to_dict(self) -> dict[str, float | int | None]:
        return {
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "annualized_volatility": self.annualized_volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "periods": self.periods,
        }


def compute_performance_metrics(
    equity_values: Sequence[Number],
    *,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> PerformanceMetrics:
    """Compute standard performance metrics from an equity curve."""
    values = _to_float_series(equity_values)
    if not values:
        values = [0.0]

    start = values[0]
    end = values[-1]
    total_return = ((end / start) - 1.0) if start > 0 else 0.0
    period_returns = compute_period_returns(values)

    return PerformanceMetrics(
        total_return=total_return,
        annualized_return=compute_annualized_return(values, periods_per_year),
        annualized_volatility=compute_annualized_volatility(period_returns, periods_per_year),
        sharpe_ratio=compute_sharpe_ratio(
            period_returns, periods_per_year=periods_per_year, risk_free_rate=risk_free_rate
        ),
        max_drawdown=compute_max_drawdown(values),
        periods=max(0, len(values) - 1),
    )
