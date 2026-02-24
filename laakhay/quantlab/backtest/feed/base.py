from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from datetime import datetime

from ..domain import Bar


class DataFeed(ABC):
    """Abstract base class for data feeds."""

    @abstractmethod
    def stream(
        self, start_dt: datetime | None = None, end_dt: datetime | None = None
    ) -> Iterator[Bar]:
        """Yield bars one by one."""
        raise NotImplementedError

    @abstractmethod
    def get_history(self, symbol: str, lookback: int) -> list[Bar]:
        """Get historical bars up to the current point in stream."""
        raise NotImplementedError
