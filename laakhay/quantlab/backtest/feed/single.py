from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime

from ..domain import Bar
from .base import DataFeed


class MemDataFeed(DataFeed):
    """In-memory single-symbol bar feed."""

    def __init__(self, bars: list[Bar], symbol: str, timeframe: str = "1h"):
        self.bars = sorted(bars, key=lambda b: b.timestamp)
        self.symbol = symbol
        self.timeframe = timeframe
        self._current_idx = -1

    def stream(
        self, start_dt: datetime | None = None, end_dt: datetime | None = None
    ) -> Iterator[Bar]:
        self._current_idx = -1
        for i, bar in enumerate(self.bars):
            if start_dt and bar.timestamp < start_dt:
                continue
            if end_dt and bar.timestamp > end_dt:
                break
            self._current_idx = i
            yield bar

    def get_history(self, symbol: str, lookback: int) -> list[Bar]:
        if symbol != self.symbol or self._current_idx < 0:
            return []
        start = max(0, self._current_idx - lookback + 1)
        end = self._current_idx + 1
        return self.bars[start:end]
