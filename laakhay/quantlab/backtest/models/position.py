from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from .order import OrderSide
from .trade import Trade


@dataclass
class Position:
    """Current holding of an asset."""

    symbol: str
    qty: Decimal = Decimal("0")
    avg_entry_price: Decimal = Decimal("0")

    @property
    def market_value(self) -> Decimal:
        """Signed market value at average entry price."""
        return self.qty * self.avg_entry_price

    @property
    def abs_qty(self) -> Decimal:
        return abs(self.qty)

    @property
    def is_flat(self) -> bool:
        return self.qty == 0

    @property
    def is_long(self) -> bool:
        return self.qty > 0

    @property
    def is_short(self) -> bool:
        return self.qty < 0

    def update(self, trade: Trade) -> None:
        """Update position with a new trade in one-way netting mode."""
        if trade.symbol != self.symbol:
            raise ValueError(
                f"Trade symbol {trade.symbol} does not match position symbol {self.symbol}"
            )
        if trade.qty <= 0:
            raise ValueError("Trade quantity must be positive")

        if trade.side == OrderSide.BUY:
            self._apply_buy(trade)
            return
        self._apply_sell(trade)

    def _apply_buy(self, trade: Trade) -> None:
        """Apply BUY against current net position."""
        if self.is_flat:
            self.qty = trade.qty
            self.avg_entry_price = trade.price
            return

        if self.is_long:
            total_cost = (self.qty * self.avg_entry_price) + (trade.qty * trade.price)
            self.qty += trade.qty
            self.avg_entry_price = total_cost / self.qty
            return

        # Covering short position.
        short_qty = abs(self.qty)
        if trade.qty < short_qty:
            self.qty += trade.qty
            return
        if trade.qty == short_qty:
            self.qty = Decimal("0")
            self.avg_entry_price = Decimal("0")
            return

        # Reverse short -> long.
        remainder = trade.qty - short_qty
        self.qty = remainder
        self.avg_entry_price = trade.price

    def _apply_sell(self, trade: Trade) -> None:
        """Apply SELL against current net position."""
        if self.is_flat:
            self.qty = -trade.qty
            self.avg_entry_price = trade.price
            return

        if self.is_short:
            total_qty = abs(self.qty) + trade.qty
            total_notional = (abs(self.qty) * self.avg_entry_price) + (trade.qty * trade.price)
            self.qty = -total_qty
            self.avg_entry_price = total_notional / total_qty
            return

        # Reducing long position.
        if trade.qty < self.qty:
            self.qty -= trade.qty
            return
        if trade.qty == self.qty:
            self.qty = Decimal("0")
            self.avg_entry_price = Decimal("0")
            return

        # Reverse long -> short.
        remainder = trade.qty - self.qty
        self.qty = -remainder
        self.avg_entry_price = trade.price
