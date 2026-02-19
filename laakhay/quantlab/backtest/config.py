from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum


class PositionMode(StrEnum):
    """Position accounting mode."""

    ONE_WAY = "ONE_WAY"


class TradeDirectionMode(StrEnum):
    """Allowed directional exposure."""

    LONG_ONLY = "LONG_ONLY"
    SHORT_ONLY = "SHORT_ONLY"
    LONG_SHORT = "LONG_SHORT"


@dataclass(frozen=True)
class FrequencyControl:
    """Controls trade frequency and holding constraints."""

    cooldown_bars: int = 0
    min_hold_bars: int = 1
    max_bars_in_trade: int | None = None
    max_entries_per_day: int | None = None

    def __post_init__(self) -> None:
        if self.cooldown_bars < 0:
            raise ValueError("cooldown_bars must be >= 0")
        if self.min_hold_bars < 0:
            raise ValueError("min_hold_bars must be >= 0")
        if self.max_bars_in_trade is not None and self.max_bars_in_trade <= 0:
            raise ValueError("max_bars_in_trade must be > 0 when provided")
        if self.max_entries_per_day is not None and self.max_entries_per_day <= 0:
            raise ValueError("max_entries_per_day must be > 0 when provided")


@dataclass(frozen=True)
class BreakevenControl:
    """Configuration for moving stop to breakeven after favorable movement."""

    enabled: bool = False
    trigger_rr: float = 1.0
    offset_bps: float | Decimal = 0

    def __post_init__(self) -> None:
        if self.trigger_rr <= 0:
            raise ValueError("trigger_rr must be > 0")
        offset = Decimal(str(self.offset_bps))
        if offset < 0:
            raise ValueError("offset_bps must be >= 0")

    @property
    def offset_fraction(self) -> Decimal:
        return Decimal(str(self.offset_bps)) / Decimal("10000")


@dataclass(frozen=True)
class BacktestConfig:
    """Core backtesting controls."""

    position_mode: PositionMode = PositionMode.ONE_WAY
    direction_mode: TradeDirectionMode = TradeDirectionMode.LONG_SHORT
    frequency: FrequencyControl = field(default_factory=FrequencyControl)
    breakeven: BreakevenControl = field(default_factory=BreakevenControl)
    risk_free_rate: float = 0.0
    allow_entry_same_bar_as_exit: bool = False
    daily_loss_limit_pct: float | Decimal | None = None

    def __post_init__(self) -> None:
        if self.daily_loss_limit_pct is None:
            return
        pct = Decimal(str(self.daily_loss_limit_pct))
        if pct <= 0 or pct > 100:
            raise ValueError("daily_loss_limit_pct must be in (0, 100]")

    def allows_long_entry(self) -> bool:
        return self.direction_mode in {TradeDirectionMode.LONG_ONLY, TradeDirectionMode.LONG_SHORT}

    def allows_short_entry(self) -> bool:
        return self.direction_mode in {TradeDirectionMode.SHORT_ONLY, TradeDirectionMode.LONG_SHORT}

    @property
    def daily_loss_limit_fraction(self) -> Decimal | None:
        if self.daily_loss_limit_pct is None:
            return None
        return Decimal(str(self.daily_loss_limit_pct)) / Decimal("100")
