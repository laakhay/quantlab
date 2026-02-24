from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from laakhay.quantlab.backtest.feed import MultiAssetMemFeed
from laakhay.quantlab.backtest.models import Bar


def _bar(ts: datetime, close: str, symbol: str) -> Bar:
    c = Decimal(close)
    return Bar(
        timestamp=ts,
        open=c,
        high=c,
        low=c,
        close=c,
        volume=Decimal("100"),
        symbol=symbol,
    )


def test_multi_feed_stream_is_time_then_symbol_ordered():
    t0 = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
    feed = MultiAssetMemFeed(
        {
            "ETHUSDT": [
                _bar(t0, "2000", "ETHUSDT"),
                _bar(t0 + timedelta(hours=1), "2010", "ETHUSDT"),
            ],
            "BTCUSDT": [
                _bar(t0, "40000", "BTCUSDT"),
                _bar(t0 + timedelta(hours=1), "40100", "BTCUSDT"),
            ],
        }
    )

    events = list(feed.stream())
    assert [(sym, bar.timestamp) for sym, bar in events] == [
        ("BTCUSDT", t0),
        ("ETHUSDT", t0),
        ("BTCUSDT", t0 + timedelta(hours=1)),
        ("ETHUSDT", t0 + timedelta(hours=1)),
    ]


def test_multi_feed_history_scoped_by_symbol():
    t0 = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
    feed = MultiAssetMemFeed(
        {
            "BTCUSDT": [
                _bar(t0, "40000", "BTCUSDT"),
                _bar(t0 + timedelta(hours=1), "40100", "BTCUSDT"),
                _bar(t0 + timedelta(hours=2), "40200", "BTCUSDT"),
            ]
        }
    )
    for _ in feed.stream():
        pass

    hist = feed.get_history("BTCUSDT", lookback=2)
    assert len(hist) == 2
    assert [bar.close for bar in hist] == [Decimal("40100"), Decimal("40200")]
