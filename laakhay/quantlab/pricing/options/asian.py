"""Asian option contracts."""

from dataclasses import dataclass
from .base import PathDependentOption, Side, PayoffType
from ..utils import infer_backend


@dataclass(frozen=True)
class AsianCall(PathDependentOption):
    """Asian call option contract."""

    @property
    def payoff_type(self) -> PayoffType:
        return PayoffType.ASIAN

    @property
    def side(self) -> Side:
        return Side.CALL

    @infer_backend
    def _path_payoff(self, price_paths, backend=None):
        """Compute Asian call payoff: max(Avg(S) - K, 0)."""
        price_paths = backend.convert(price_paths)
        average_price = backend.mean(price_paths, axis=1)
        strike = backend.convert(self.strike)
        zero = backend.convert(0.0)
        return backend.maximum(backend.add(average_price, backend.mul(-1, strike)), zero)


@dataclass(frozen=True)
class AsianPut(PathDependentOption):
    """Asian put option contract."""

    @property
    def payoff_type(self) -> PayoffType:
        return PayoffType.ASIAN

    @property
    def side(self) -> Side:
        return Side.PUT

    @infer_backend
    def _path_payoff(self, price_paths, backend=None):
        """Compute Asian put payoff: max(K - Avg(S), 0)."""
        price_paths = backend.convert(price_paths)
        average_price = backend.mean(price_paths, axis=1)
        strike = backend.convert(self.strike)
        zero = backend.convert(0.0)
        return backend.maximum(backend.add(strike, backend.mul(-1, average_price)), zero)


@dataclass(frozen=True)
class GeometricAsianCall(PathDependentOption):
    """Geometric Asian call option contract."""

    @property
    def payoff_type(self) -> PayoffType:
        return PayoffType.GEOMETRIC_ASIAN

    @property
    def side(self) -> Side:
        return Side.CALL

    @infer_backend
    def _path_payoff(self, price_paths, backend=None):
        """Compute geometric Asian call payoff: max(GeoAvg(S) - K, 0)."""
        price_paths = backend.convert(price_paths)
        geometric_average = backend.exp(backend.mean(backend.log(price_paths), axis=1))
        strike = backend.convert(self.strike)
        zero = backend.convert(0.0)
        return backend.maximum(backend.add(geometric_average, backend.mul(-1, strike)), zero)


@dataclass(frozen=True)
class GeometricAsianPut(PathDependentOption):
    """Geometric Asian put option contract."""

    @property
    def payoff_type(self) -> PayoffType:
        return PayoffType.GEOMETRIC_ASIAN

    @property
    def side(self) -> Side:
        return Side.PUT

    @infer_backend
    def _path_payoff(self, price_paths, backend=None):
        """Compute geometric Asian put payoff: max(K - GeoAvg(S), 0)."""
        price_paths = backend.convert(price_paths)
        geometric_average = backend.exp(backend.mean(backend.log(price_paths), axis=1))
        strike = backend.convert(self.strike)
        zero = backend.convert(0.0)
        return backend.maximum(backend.add(strike, backend.mul(-1, geometric_average)), zero)
