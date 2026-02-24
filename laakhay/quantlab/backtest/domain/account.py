from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


@dataclass
class PortfolioAccount:
    """Shared account state for portfolio-level backtesting."""

    initial_cash: Decimal
    cash: Decimal
    realized_pnl: Decimal = Decimal("0")

    @classmethod
    def from_initial_capital(cls, initial_capital: float | Decimal) -> PortfolioAccount:
        capital = Decimal(str(initial_capital))
        return cls(initial_cash=capital, cash=capital)

    def can_reserve(self, amount: Decimal) -> bool:
        return amount >= 0 and self.cash >= amount

    def reserve(self, amount: Decimal) -> bool:
        if not self.can_reserve(amount):
            return False
        self.cash -= amount
        return True

    def release(self, amount: Decimal) -> None:
        if amount <= 0:
            return
        self.cash += amount

    def apply_realized_pnl(self, pnl: Decimal) -> None:
        self.realized_pnl += pnl
        self.cash += pnl
