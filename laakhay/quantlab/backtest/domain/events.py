from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any


@dataclass(frozen=True)
class ExecutionRejection:
    symbol: str
    timestamp: Any
    reason: str


@dataclass(frozen=True)
class LiquidationEvent:
    symbol: str
    timestamp: Any
    mark_price: Decimal
    reason: str
