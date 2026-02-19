from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum


class OrderSide(StrEnum):
    """Side of the order."""

    BUY = "BUY"
    SELL = "SELL"


class OrderType(StrEnum):
    """Type of the order."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"


class OrderStatus(StrEnum):
    """Status of the order."""

    PENDING = "PENDING"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass
class Order:
    """Order to buy or sell an asset."""

    id: str
    symbol: str
    side: OrderSide
    qty: Decimal
    type: OrderType
    price: Decimal | None = None  # Structurally optional, required for LIMIT
    label: str = ""
    parent_order_id: str | None = None
    oco_group_id: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    status: OrderStatus = OrderStatus.PENDING

    def __post_init__(self) -> None:
        """Validate order parameters."""
        if self.qty <= 0:
            raise ValueError("Order quantity must be positive")
        if self.type in {OrderType.LIMIT, OrderType.STOP} and (
            self.price is None or self.price <= 0
        ):
            raise ValueError(f"{self.type.value} order must have a positive price")
