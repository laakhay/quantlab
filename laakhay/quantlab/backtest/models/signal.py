from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from .order import OrderSide, OrderType

# Type Definitions
RiskType = float | Decimal | str  # e.g. 0.01, Decimal("100"), "1%"
SLTPType = Decimal | str  # e.g. Decimal("50000"), "2%"


@dataclass
class Signal:
    """Trading signal with advanced parameters."""

    symbol: str
    side: OrderSide
    type: OrderType
    price: Decimal | None = None  # Required for LIMIT/STOP

    # Risk Management (Overrides Strategy defaults)
    sl: SLTPType | None = None
    tp: SLTPType | None = None

    # Position Sizing (Overrides Strategy defaults)
    size: RiskType | None = None
