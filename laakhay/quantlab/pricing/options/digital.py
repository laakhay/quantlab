"""Digital option contracts."""

from dataclasses import dataclass

from ..utils import infer_backend
from .base import PathIndependentOption, PayoffType, Side


@dataclass(frozen=True)
class DigitalCall(PathIndependentOption):
    """Digital call option contract."""

    payout: float = 1.0

    @property
    def payoff_type(self) -> PayoffType:
        return PayoffType.DIGITAL

    @property
    def side(self) -> Side:
        return Side.CALL

    @infer_backend
    def payoff(self, spot_prices, backend=None):
        spot_prices = backend.array(spot_prices)
        strike = backend.array(self.strike)
        cond = backend.greater(spot_prices, strike)
        payout = backend.array(self.payout)
        zero = backend.array(0.0)
        return backend.where(cond, payout, zero)

    def intrinsic_value(self, spot: float) -> float:
        return self.payout if spot > self.strike else 0.0


@dataclass(frozen=True)
class DigitalPut(PathIndependentOption):
    """Digital put option contract."""

    payout: float = 1.0

    @property
    def payoff_type(self) -> PayoffType:
        return PayoffType.DIGITAL

    @property
    def side(self) -> Side:
        return Side.PUT

    @infer_backend
    def payoff(self, spot_prices, backend=None):
        spot_prices = backend.array(spot_prices)
        strike = backend.array(self.strike)
        cond = backend.less(spot_prices, strike)
        payout = backend.array(self.payout)
        zero = backend.array(0.0)
        return backend.where(cond, payout, zero)

    def intrinsic_value(self, spot: float) -> float:
        return self.payout if spot < self.strike else 0.0
