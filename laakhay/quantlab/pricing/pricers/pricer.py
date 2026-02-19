"""Unified pricer interface."""

from __future__ import annotations
from enum import Enum
from laakhay.quantlab.backend import get_backend
from ..market import MarketData
from ..greeks import Greeks
from .black_scholes import BlackScholesPricer
from .monte_carlo import MonteCarloPricer


class PricingMethod(Enum):
    """Available pricing methods."""

    BLACK_SCHOLES = "black_scholes"
    MONTE_CARLO = "monte_carlo"
    AUTO = "auto"


class Pricer:
    """Unified interface for option pricing."""

    def __init__(self, method: PricingMethod = PricingMethod.AUTO, backend=None):
        self.backend = backend or get_backend()
        self.method = method
        self._bs_pricer = BlackScholesPricer(backend=self.backend)
        self._mc_pricer = None  # Lazy init

    @property
    def mc_pricer(self):
        if self._mc_pricer is None:
            self._mc_pricer = MonteCarloPricer(backend=self.backend)
        return self._mc_pricer

    def _select_pricer(self, option):
        """Select appropriate pricer based on method and option."""
        if self.method == PricingMethod.BLACK_SCHOLES:
            return self._bs_pricer
        if self.method == PricingMethod.MONTE_CARLO:
            return self.mc_pricer

        # AUTO logic: try BS first (analytical), then MC
        if self._bs_pricer.supports(option):
            return self._bs_pricer
        return self.mc_pricer

    def price(self, option, market: MarketData | None = None) -> object:
        pricer = self._select_pricer(option)
        return pricer.price(option, market)

    def price_with_greeks(self, option, market: MarketData | None = None) -> tuple[object, Greeks]:
        pricer = self._select_pricer(option)
        if hasattr(pricer, "price_with_greeks"):
            return pricer.price_with_greeks(option, market)

        # Fallback for pricers without analytical Greeks (like MC)
        price = pricer.price(option, market)
        backend = market.backend if market else self.backend
        zero = backend.zeros_like(price) if hasattr(backend, "zeros_like") else 0.0
        return price, Greeks(delta=zero, gamma=zero, vega=zero, theta=zero, rho=zero)
