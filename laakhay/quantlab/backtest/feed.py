from abc import ABC, abstractmethod
from collections.abc import Iterator
from datetime import datetime

from .models import Bar


class DataFeed(ABC):
    """Abstract base class for data feeds."""

    @abstractmethod
    def stream(
        self, start_dt: datetime | None = None, end_dt: datetime | None = None
    ) -> Iterator[Bar]:
        """Yield bars one by one."""
        pass

    @abstractmethod
    def get_history(self, symbol: str, lookback: int) -> list[Bar]:
        """Get historical bars up to the current point in stream."""
        pass


class MemDataFeed(DataFeed):
    """In-memory grid of bars."""

    def __init__(self, bars: list[Bar], symbol: str, timeframe: str = "1h"):
        """Initialize with a list of bars."""
        # Sort bars by timestamp to be safe
        self.bars = sorted(bars, key=lambda b: b.timestamp)
        self.symbol = symbol
        self.timeframe = timeframe
        self._current_idx = -1

    def stream(
        self, start_dt: datetime | None = None, end_dt: datetime | None = None
    ) -> Iterator[Bar]:
        """Stream bars."""
        self._current_idx = -1
        for i, bar in enumerate(self.bars):
            if start_dt and bar.timestamp < start_dt:
                continue
            if end_dt and bar.timestamp > end_dt:
                break

            self._current_idx = i
            yield bar

    def get_history(self, symbol: str, lookback: int) -> list[Bar]:
        """Get history up to current_idx."""
        if symbol != self.symbol:
            # In a multi-asset feed, we'd handle this differently
            return []

        if self._current_idx < 0:
            return []

        # simple slice: max(0, current - lookback + 1) : current + 1
        start = max(0, self._current_idx - lookback + 1)
        end = self._current_idx + 1

        sliced_bars = self.bars[start:end]
        return sliced_bars
