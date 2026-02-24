from __future__ import annotations

from enum import StrEnum


class PositionMode(StrEnum):
    """Position accounting mode."""

    ONE_WAY = "ONE_WAY"


class TradeDirectionMode(StrEnum):
    """Allowed directional exposure."""

    LONG_ONLY = "LONG_ONLY"
    SHORT_ONLY = "SHORT_ONLY"
    LONG_SHORT = "LONG_SHORT"


class RunMode(StrEnum):
    """Execution topology for a backtest run."""

    SINGLE = "single"
    BATCH_INDEPENDENT = "batch_independent"
    PORTFOLIO_SHARED = "portfolio_shared"


class PortfolioMode(StrEnum):
    """Capital accounting model across multiple symbols."""

    INDEPENDENT = "independent"
    SHARED_CAPITAL = "shared_capital"


class MarginMode(StrEnum):
    """Margin accounting mode for leveraged runs."""

    CROSS = "cross"
    ISOLATED = "isolated"


class ProtectiveExitPolicy(StrEnum):
    """How to resolve SL/TP dual-hit within the same bar."""

    CONSERVATIVE = "conservative"
    OPTIMISTIC = "optimistic"
    NEAREST_TO_OPEN = "nearest_to_open"
