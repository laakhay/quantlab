from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal

from .order import OrderSide


@dataclass
class Trade:
    """Execution of an order."""

    id: str
    order_id: str
    symbol: str
    side: OrderSide
    qty: Decimal
    price: Decimal
    commission: Decimal = Decimal("0")
    sl_price: Decimal | None = None
    tp_price: Decimal | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
