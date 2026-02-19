from __future__ import annotations

from decimal import Decimal
from typing import Any

from ..models import Order, OrderSide, OrderType, Trade


class OrderManager:
    """Manages active orders and matches them against market data."""

    def __init__(self):
        self.active_orders: list[Order] = []
        self.orders_history: list[Order] = []
        self.trades: list[Trade] = []

    def create_order(
        self,
        symbol: str,
        side: OrderSide,
        qty: Decimal,
        type: OrderType,
        price: Decimal | None = None,
        label: str = "",
        parent_order_id: str | None = None,
        oco_group_id: str | None = None,
        timestamp: Any = None,
    ) -> Order:
        """Create and register a new order."""
        order_id = (
            f"{label}-{len(self.orders_history) + 1}"
            if label
            else f"O-{len(self.orders_history) + 1}"
        )

        order = Order(
            id=order_id,
            symbol=symbol,
            side=side,
            qty=qty,
            type=type,
            price=price,
            label=label,
            parent_order_id=parent_order_id,
            oco_group_id=oco_group_id,
            timestamp=timestamp,
        )
        self.active_orders.append(order)
        self.orders_history.append(order)
        return order

    def match_orders(self, bar: Any, symbol: str) -> list[tuple[Order, Decimal]]:
        """Check for order fills against current bar.

        Returns:
            List of (Order, FillPrice) tuples.
        """
        triggered: list[tuple[Order, Decimal]] = []

        for order in self.active_orders:
            if order.symbol != symbol:
                continue

            is_filled = False
            fill_price = order.price

            # LIMIT matching:
            # - LIMIT BUY fills when market trades at or below limit.
            # - LIMIT SELL fills when market trades at or above limit.
            if order.type == OrderType.LIMIT:
                if (order.side == OrderSide.BUY and bar.low <= order.price) or (
                    order.side == OrderSide.SELL and bar.high >= order.price
                ):
                    is_filled = True
                    fill_price = order.price

            # STOP matching:
            # - STOP BUY triggers when market trades at or above stop.
            # - STOP SELL triggers when market trades at or below stop.
            elif (
                order.type == OrderType.STOP
                and order.price
                and (
                    (order.side == OrderSide.SELL and bar.low <= order.price)
                    or (order.side == OrderSide.BUY and bar.high >= order.price)
                )
            ):
                is_filled = True
                fill_price = order.price

            if is_filled:
                triggered.append((order, fill_price))

        # If multiple exit orders trigger inside the same bar (e.g., SL and TP),
        # use a conservative single fill to avoid unrealistic double execution.
        filled: list[tuple[Order, Decimal]] = []
        if triggered:
            if len(triggered) == 1:
                filled = triggered
            else:
                sell_triggers = [item for item in triggered if item[0].side == OrderSide.SELL]
                if sell_triggers:
                    # Conservative for long exits: assume the lowest executable exit price.
                    filled = [min(sell_triggers, key=lambda item: item[1])]
                else:
                    # Conservative for short exits: assume the highest executable exit price.
                    filled = [max(triggered, key=lambda item: item[1])]

        # Remove filled from active
        for order, _ in filled:
            if order in self.active_orders:
                self.active_orders.remove(order)

        return filled

    def cancel_symbol_orders(self, symbol: str) -> None:
        """Cancel all active orders for a symbol."""
        self.active_orders = [o for o in self.active_orders if o.symbol != symbol]

    def find_active_order(self, symbol: str, label: str) -> Order | None:
        """Find an active order by symbol and label."""
        for order in self.active_orders:
            if order.symbol == symbol and order.label == label:
                return order
        return None

    def update_order_price(self, order_id: str, new_price: Decimal) -> bool:
        """Update order price in-place for an active order."""
        if new_price <= 0:
            return False
        for order in self.active_orders:
            if order.id == order_id:
                order.price = new_price
                return True
        return False
