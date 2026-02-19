from abc import ABC, abstractmethod
from collections.abc import Iterator
from datetime import datetime

from laakhay.data.models.bar import Bar
from laakhay.ta.core.dataset import Dataset
from laakhay.ta.core.ohlcv import OHLCV


class DataFeed(ABC):
    """Abstract base class for data feeds."""

    @abstractmethod
    def stream(
        self, start_dt: datetime | None = None, end_dt: datetime | None = None
    ) -> Iterator[Bar]:
        """Yield bars one by one."""
        pass

    @abstractmethod
    def get_history(self, symbol: str, lookback: int) -> Dataset:
        """Get historical data for a symbol up to the current point in stream.

        This is crucial for feeding the Strategy Engine without lookahead bias.
        """
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

    def get_history(self, symbol: str, lookback: int) -> Dataset:
        """Get history up to current_idx."""
        if symbol != self.symbol:
            # In a multi-asset feed, we'd handle this differently
            return Dataset()

        if self._current_idx < 0:
            return Dataset()

        # simple slice: max(0, current - lookback + 1) : current + 1
        start = max(0, self._current_idx - lookback + 1)
        end = self._current_idx + 1

        sliced_bars = self.bars[start:end]
        if not sliced_bars:
            return Dataset()

        # Convert to TA OHLCV
        # Direct construction to handle attribute mismatch (Bar.timestamp vs Bar.ts)
        ohlcv = OHLCV(
            timestamps=tuple(b.timestamp for b in sliced_bars),
            opens=tuple(b.open for b in sliced_bars),
            highs=tuple(b.high for b in sliced_bars),
            lows=tuple(b.low for b in sliced_bars),
            closes=tuple(b.close for b in sliced_bars),
            volumes=tuple(b.volume for b in sliced_bars),
            is_closed=tuple(b.is_closed for b in sliced_bars),
            symbol=self.symbol,
            timeframe=self.timeframe,
        )

        # Wrap in Dataset
        # Note: We create a fresh dataset every step.
        # Optimization: In Phase 2, we can use a rolling buffer or pre-calculated columns.
        ds = Dataset()
        ds.add_series(self.symbol, self.timeframe, ohlcv, source="ohlcv")
        return ds
