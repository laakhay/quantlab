from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any


@dataclass
class Bar:
    """Quantlab-native OHLCV bar model."""

    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    symbol: str | None = None
    interval: str | None = None
    is_closed: bool = True

    def __post_init__(self) -> None:
        self.open = Decimal(str(self.open))
        self.high = Decimal(str(self.high))
        self.low = Decimal(str(self.low))
        self.close = Decimal(str(self.close))
        self.volume = Decimal(str(self.volume))
        if self.timestamp.tzinfo is None:
            self.timestamp = self.timestamp.replace(tzinfo=UTC)

    def model_copy(self, *, update: dict[str, Any] | None = None) -> Bar:
        """Pydantic-like compatibility helper used in tests."""
        return replace(self, **(update or {}))
