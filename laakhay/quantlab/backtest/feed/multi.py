from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime

from ..domain import Bar


class MultiAssetMemFeed:
    """In-memory multi-symbol feed with deterministic timestamp ordering."""

    def __init__(self, bars_by_symbol: dict[str, list[Bar]], timeframe: str = "1h"):
        self._bars_by_symbol: dict[str, list[Bar]] = {
            symbol: sorted(bars, key=lambda b: b.timestamp)
            for symbol, bars in bars_by_symbol.items()
        }
        self.timeframe = timeframe
        self.symbols = sorted(self._bars_by_symbol.keys())
        self._current_idx: dict[str, int] = dict.fromkeys(self.symbols, -1)
        events: list[tuple[str, int, datetime]] = []
        for symbol, bars in self._bars_by_symbol.items():
            for idx, bar in enumerate(bars):
                events.append((symbol, idx, bar.timestamp))
        self._events = sorted(events, key=lambda x: (x[2], x[0]))

    def stream(
        self, start_dt: datetime | None = None, end_dt: datetime | None = None
    ) -> Iterator[tuple[str, Bar]]:
        self._current_idx = dict.fromkeys(self.symbols, -1)
        for symbol, idx, ts in self._events:
            if start_dt and ts < start_dt:
                continue
            if end_dt and ts > end_dt:
                break
            self._current_idx[symbol] = idx
            yield symbol, self._bars_by_symbol[symbol][idx]

    def get_history(self, symbol: str, lookback: int) -> list[Bar]:
        idx = self._current_idx.get(symbol, -1)
        if idx < 0:
            return []
        bars = self._bars_by_symbol[symbol]
        start = max(0, idx - lookback + 1)
        return bars[start : idx + 1]

    def bars_for_symbol(self, symbol: str) -> list[Bar]:
        return list(self._bars_by_symbol.get(symbol, ()))
