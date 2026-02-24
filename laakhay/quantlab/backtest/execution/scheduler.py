from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime

from ..domain import Bar
from ..feed import MultiAssetMemFeed


class MultiAssetScheduler:
    """Deterministic event scheduler over a multi-asset feed."""

    def __init__(self, feed: MultiAssetMemFeed):
        self._feed = feed

    def iter_events(
        self, start_dt: datetime | None = None, end_dt: datetime | None = None
    ) -> Iterator[tuple[str, Bar]]:
        yield from self._feed.stream(start_dt=start_dt, end_dt=end_dt)
