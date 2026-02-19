"""Barrier option contracts."""

from dataclasses import dataclass
from enum import Enum, auto
from .base import PathDependentOption, Side, PayoffType
from ..utils import infer_backend


class BarrierDirection(Enum):
    """Direction of barrier."""

    UP = auto()
    DOWN = auto()


class KnockType(Enum):
    """Type of barrier knock."""

    OUT = auto()
    IN = auto()


@dataclass(frozen=True)
class BarrierOption(PathDependentOption):
    """Base class for barrier options."""

    barrier: float = None
    barrier_direction: BarrierDirection = BarrierDirection.UP
    knock_type: KnockType = KnockType.OUT

    def __post_init__(self):
        super().__post_init__()
        if self.barrier is None or self.barrier <= 0:
            raise ValueError("barrier must be > 0")

    @property
    def payoff_type(self) -> PayoffType:
        return PayoffType.BARRIER

    @infer_backend
    def _barrier_breached(self, price_paths, backend=None):
        """Check if barrier is breached in any path."""
        price_paths = backend.convert(price_paths)
        barrier = backend.convert(self.barrier)
        if self.barrier_direction == BarrierDirection.UP:
            return backend.any(backend.greater_equal(price_paths, barrier), axis=1)
        else:
            return backend.any(backend.less_equal(price_paths, barrier), axis=1)

    @infer_backend
    def _option_active(self, price_paths, backend=None):
        """Check if option is active based on barrier and knock type."""
        breached = self._barrier_breached(price_paths, backend=backend)
        if self.knock_type == KnockType.OUT:
            return backend.logical_not(breached)
        else:
            return breached

    @infer_backend
    def _path_payoff(self, price_paths, backend=None):
        """Compute barrier option payoff."""
        price_paths = backend.convert(price_paths)
        terminal_prices = price_paths[:, -1]
        active = self._option_active(price_paths, backend=backend)

        strike = backend.convert(self.strike)
        zero = backend.convert(0.0)

        if self.side == Side.CALL:
            vanilla_payoff = backend.maximum(
                backend.add(terminal_prices, backend.mul(-1, strike)), zero
            )
        else:
            vanilla_payoff = backend.maximum(
                backend.add(strike, backend.mul(-1, terminal_prices)), zero
            )
        return backend.where(active, vanilla_payoff, zero)


@dataclass(frozen=True)
class UpAndOutCall(BarrierOption):
    barrier_direction: BarrierDirection = BarrierDirection.UP
    knock_type: KnockType = KnockType.OUT

    @property
    def side(self) -> Side:
        return Side.CALL


@dataclass(frozen=True)
class UpAndOutPut(BarrierOption):
    barrier_direction: BarrierDirection = BarrierDirection.UP
    knock_type: KnockType = KnockType.OUT

    @property
    def side(self) -> Side:
        return Side.PUT


@dataclass(frozen=True)
class DownAndOutCall(BarrierOption):
    barrier_direction: BarrierDirection = BarrierDirection.DOWN
    knock_type: KnockType = KnockType.OUT

    @property
    def side(self) -> Side:
        return Side.CALL


@dataclass(frozen=True)
class DownAndOutPut(BarrierOption):
    barrier_direction: BarrierDirection = BarrierDirection.DOWN
    knock_type: KnockType = KnockType.OUT

    @property
    def side(self) -> Side:
        return Side.PUT


@dataclass(frozen=True)
class UpAndInCall(BarrierOption):
    barrier_direction: BarrierDirection = BarrierDirection.UP
    knock_type: KnockType = KnockType.IN

    @property
    def side(self) -> Side:
        return Side.CALL


@dataclass(frozen=True)
class UpAndInPut(BarrierOption):
    barrier_direction: BarrierDirection = BarrierDirection.UP
    knock_type: KnockType = KnockType.IN

    @property
    def side(self) -> Side:
        return Side.PUT


@dataclass(frozen=True)
class DownAndInCall(BarrierOption):
    barrier_direction: BarrierDirection = BarrierDirection.DOWN
    knock_type: KnockType = KnockType.IN

    @property
    def side(self) -> Side:
        return Side.CALL


@dataclass(frozen=True)
class DownAndInPut(BarrierOption):
    barrier_direction: BarrierDirection = BarrierDirection.DOWN
    knock_type: KnockType = KnockType.IN

    @property
    def side(self) -> Side:
        return Side.PUT
