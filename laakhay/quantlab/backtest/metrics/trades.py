from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class RoundTripTrade:
    """Closed round-trip trade."""

    symbol: str
    direction: str
    qty: float
    entry_price: float
    exit_price: float
    pnl: float
    pnl_pct: float
    holding_seconds: float
    holding_bars: int
    entry_time: datetime | None = None
    exit_time: datetime | None = None
    exit_reason: str | None = None
    sl_price: float | None = None
    tp_price: float | None = None


@dataclass(frozen=True)
class TradeMetrics:
    total_round_trips: int
    win_rate: float | None
    profit_factor: float | None
    expectancy: float | None
    avg_holding_seconds: float | None
    avg_holding_bars: float | None
    avg_win: float | None
    avg_loss: float | None

    def to_dict(self) -> dict[str, float | int | None]:
        return {
            "total_round_trips": self.total_round_trips,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "expectancy": self.expectancy,
            "avg_holding_seconds": self.avg_holding_seconds,
            "avg_holding_bars": self.avg_holding_bars,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
        }


def compute_trade_metrics(round_trips: Sequence[RoundTripTrade]) -> TradeMetrics:
    """Compute trade-level metrics from closed round trips."""
    total = len(round_trips)
    if total == 0:
        return TradeMetrics(
            total_round_trips=0,
            win_rate=None,
            profit_factor=None,
            expectancy=None,
            avg_holding_seconds=None,
            avg_holding_bars=None,
            avg_win=None,
            avg_loss=None,
        )

    pnls = [trade.pnl for trade in round_trips]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))

    profit_factor = None if gross_loss == 0 else gross_profit / gross_loss

    avg_win = (sum(wins) / len(wins)) if wins else None
    avg_loss = (sum(losses) / len(losses)) if losses else None

    holding_seconds = [trade.holding_seconds for trade in round_trips]
    holding_bars = [trade.holding_bars for trade in round_trips]

    return TradeMetrics(
        total_round_trips=total,
        win_rate=len(wins) / total,
        profit_factor=profit_factor,
        expectancy=sum(pnls) / total,
        avg_holding_seconds=sum(holding_seconds) / total if holding_seconds else None,
        avg_holding_bars=sum(holding_bars) / total if holding_bars else None,
        avg_win=avg_win,
        avg_loss=avg_loss,
    )
