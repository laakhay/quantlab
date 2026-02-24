from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from ..domain import BacktestConfig, PortfolioAccount


@dataclass(frozen=True)
class AllocationDecision:
    notional: Decimal
    reason: str | None = None


class EqualNotionalAllocator:
    """Allocate notional evenly across available slots in shared-capital mode."""

    def allocate(
        self,
        *,
        account: PortfolioAccount,
        config: BacktestConfig,
        open_positions_count: int,
        current_equity: Decimal,
    ) -> AllocationDecision:
        max_open_positions = config.portfolio.max_open_positions or 1
        remaining_slots = max(1, max_open_positions - open_positions_count)
        tentative_notional = account.cash / Decimal(str(remaining_slots))
        if tentative_notional <= 0:
            return AllocationDecision(notional=Decimal("0"), reason="no_available_cash")

        symbol_weight_pct = config.portfolio.max_symbol_weight_pct
        if symbol_weight_pct is not None:
            cap_notional = current_equity * (Decimal(str(symbol_weight_pct)) / Decimal("100"))
            tentative_notional = min(tentative_notional, cap_notional)

        if tentative_notional <= 0:
            return AllocationDecision(notional=Decimal("0"), reason="allocation_cap_zero")
        return AllocationDecision(notional=tentative_notional)
