"""Core types and interfaces for Greeks calculation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from enum import Enum, auto

from ...greeks import Greeks
from ...market import MarketData
from ...options.base import BaseLeg, OptionContract
from ...options.strategies import CompositeOptionContract

# Type alias for user convenience
Priceable = (
    OptionContract
    | BaseLeg
    | CompositeOptionContract
    | Sequence[OptionContract | BaseLeg | CompositeOptionContract]
)


class GreeksMethod(Enum):
    """Available methods for Greeks calculation in order of preference."""

    ANALYTICAL = auto()  # Fastest, limited coverage
    AUTODIFF = auto()  # Fast, broad coverage, requires JAX/torch
    FINITE_DIFF = auto()  # Slowest, universal fallback
    PRICER = auto()  # Use pricer's built-in Greeks (if available)

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented


class GreeksCalculator(ABC):
    """Abstract base for Greeks calculators."""

    def __init__(
        self,
        market: MarketData | None = None,
        *,
        spot: float | None = None,
        rate: float | None = None,
        vol: float | None = None,
    ):
        """Initialize calculator with optional market data or scalar overrides."""
        self._base_market = market
        self._base_spot = spot
        self._base_rate = rate
        self._base_vol = vol
        self.market = None

    def _create_market(self, backend):
        """Create market data with the given backend."""
        if self._base_market is not None:
            self.market = self._base_market
        else:
            self.market = MarketData.create(
                spot=self._base_spot if self._base_spot is not None else 100.0,
                rate=self._base_rate if self._base_rate is not None else 0.05,
                vol=self._base_vol if self._base_vol is not None else 0.2,
                backend=backend,
            )

    @abstractmethod
    def supports(self, option: Priceable) -> bool:
        """Check if this calculator supports the given option type."""
        pass

    @abstractmethod
    def calculate(
        self,
        option: Priceable,
        *,
        market: MarketData | None = None,
        spot: float | None = None,
        rate: float | None = None,
        vol: float | None = None,
    ) -> Greeks:
        """Calculate Greeks for option or portfolio."""
        pass

    @abstractmethod
    def calculate_greeks(
        self,
        option: OptionContract | BaseLeg | CompositeOptionContract,
    ) -> Greeks:
        """Calculate Greeks for the given option."""
        pass

    def _resolve_market(
        self,
        provided: MarketData | None,
        spot: float | None,
        rate: float | None,
        vol: float | None,
    ) -> MarketData:
        """Resolve final MarketData from optional overrides."""
        m = provided or self.market
        if spot is not None:
            m = m.with_spot(spot)
        if rate is not None:
            m = m.with_rate(rate)
        if vol is not None:
            m = m.with_vol(vol)
        return m

    def _zero_greeks(
        self,
        vol: object | None = None,
        spot: object | None = None,
        rate: object | None = None,
    ) -> Greeks:
        """Return zero Greeks with appropriate shape."""
        params = [p for p in (vol, spot, rate) if p is not None]
        arrays = [p for p in params if hasattr(p, "shape")]

        if arrays:
            ref = arrays[0]
            zero = ref * 0
            return Greeks(delta=zero, gamma=zero, vega=zero, theta=zero, rho=zero)

        return Greeks()
