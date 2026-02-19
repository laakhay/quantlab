"""European option contracts."""

from dataclasses import dataclass

from laakhay.quantlab.backend import Backend

from ..utils import infer_backend
from .base import PathIndependentOption, PayoffType, Side


@dataclass(frozen=True)
class EuropeanCall(PathIndependentOption):
    """European call option contract."""

    @property
    def payoff_type(self) -> PayoffType:
        return PayoffType.EUROPEAN

    @property
    def side(self) -> Side:
        return Side.CALL

    @infer_backend
    def payoff(self, spot_prices, backend: Backend = None):
        spot_prices = backend.array(spot_prices)
        strike = backend.array(self.strike)
        zero = backend.array(0.0)
        return backend.maximum(backend.sub(spot_prices, strike), zero)

    def intrinsic_value(self, spot: float, backend: Backend = None) -> float:
        return max(spot - self.strike, 0.0)


@dataclass(frozen=True)
class EuropeanPut(PathIndependentOption):
    """European put option contract."""

    @property
    def payoff_type(self) -> PayoffType:
        return PayoffType.EUROPEAN

    @property
    def side(self) -> Side:
        return Side.PUT

    @infer_backend
    def payoff(self, spot_prices, backend: Backend = None):
        spot_prices = backend.array(spot_prices)
        strike = backend.array(self.strike)
        zero = backend.array(0.0)
        return backend.maximum(backend.sub(strike, spot_prices), zero)

    def intrinsic_value(self, spot: float, backend: Backend = None) -> float:
        return max(self.strike - spot, 0.0)
