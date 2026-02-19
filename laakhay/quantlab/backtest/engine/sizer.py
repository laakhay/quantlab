from __future__ import annotations

from decimal import Decimal
from typing import Any


class PositionSizer:
    """Calculates position size based on risk parameters."""

    def __init__(self, capital_provider: callable):
        """
        Args:
            capital_provider: Function that returns current capital.
        """
        self._get_capital = capital_provider

    def calculate_size(self, signal: Any, price: Any) -> Decimal:
        """Calculate quantity based on risk parameters."""
        px = Decimal(str(price))
        if px <= 0:
            return Decimal("0")

        # 1. Check Signal override
        if signal.size:
            return self._parse_size(signal.size, px)

        # 2. Default Strategy Sizing (Assuming mostly fixed for now or hardcoded default)
        # TODO: Get defaults from Strategy proper
        return self._parse_size("10%", px)  # Default 10% equity

    def _parse_size(self, size: str | Decimal | float, price: Decimal) -> Decimal:
        """Parse size string ("1%", "100") or number."""
        capital = self._get_capital()
        if capital <= 0 or price <= 0:
            return Decimal("0")

        # Always cap by what can be bought with available cash.
        max_affordable_qty = capital / price

        if isinstance(size, int | float | Decimal):
            # If plain number, assume Quantity? Or Fixed Cash?
            # Ambiguous. Let's assume Fixed Cash for float/int/decimal unless small?
            # Creating a convention: < 1.0 is %, > 1.0 is Cash.
            # This is risky. Better to stick to string for %
            val = Decimal(str(size))
            if val <= 0:
                return Decimal("0")
            qty = (capital * val) / price if val < 1 else min(val, capital) / price
            return min(qty, max_affordable_qty)

        if isinstance(size, str):
            if size.endswith("%"):
                pct = Decimal(size[:-1]) / 100
                if pct <= 0:
                    return Decimal("0")
                qty = (capital * pct) / price
                return min(qty, max_affordable_qty)
            else:
                cash = Decimal(size)
                if cash <= 0:
                    return Decimal("0")
                return min(cash, capital) / price

        return Decimal("0")
