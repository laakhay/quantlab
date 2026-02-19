"""Modular Black-Scholes pricer with registry pattern."""

from __future__ import annotations
from laakhay.quantlab.backend import get_backend
from ...market import MarketData
from ...greeks import Greeks
from .registry import pricing_registry
from .formulas.european import EuropeanCallFormula, EuropeanPutFormula
from .formulas.digital import DigitalCallFormula, DigitalPutFormula
from .formulas.asian import GeometricAsianCallFormula, GeometricAsianPutFormula
from .formulas.barrier import (
    UpAndOutCallFormula,
    UpAndOutPutFormula,
    DownAndOutCallFormula,
    DownAndOutPutFormula,
    UpAndInCallFormula,
    UpAndInPutFormula,
    DownAndInCallFormula,
    DownAndInPutFormula,
)


class BlackScholesPricer:
    """Modular Black-Scholes pricer."""

    def __init__(self, backend=None):
        self.backend = backend or get_backend()
        self._setup_registry()

    def _setup_registry(self):
        pricing_registry.clear()

        from ...options import (
            EuropeanCall,
            EuropeanPut,
            DigitalCall,
            DigitalPut,
            GeometricAsianCall,
            GeometricAsianPut,
            UpAndOutCall,
            UpAndOutPut,
            DownAndOutCall,
            DownAndOutPut,
            UpAndInCall,
            UpAndInPut,
            DownAndInCall,
            DownAndInPut,
        )

        # European
        pricing_registry.register(EuropeanCall, EuropeanCallFormula())
        pricing_registry.register(EuropeanPut, EuropeanPutFormula())

        # Digital
        pricing_registry.register(DigitalCall, DigitalCallFormula())
        pricing_registry.register(DigitalPut, DigitalPutFormula())

        # Asian
        pricing_registry.register(GeometricAsianCall, GeometricAsianCallFormula())
        pricing_registry.register(GeometricAsianPut, GeometricAsianPutFormula())

        # Barrier
        pricing_registry.register(UpAndOutCall, UpAndOutCallFormula)
        pricing_registry.register(UpAndOutPut, UpAndOutPutFormula)
        pricing_registry.register(DownAndOutCall, DownAndOutCallFormula)
        pricing_registry.register(DownAndOutPut, DownAndOutPutFormula)
        pricing_registry.register(UpAndInCall, UpAndInCallFormula)
        pricing_registry.register(UpAndInPut, UpAndInPutFormula)
        pricing_registry.register(DownAndInCall, DownAndInCallFormula)
        pricing_registry.register(DownAndInPut, DownAndInPutFormula)

    def price(self, option, market: MarketData | None = None) -> object:
        market = market or MarketData(backend=self.backend)
        return pricing_registry.price(option, market)

    def price_with_greeks(self, option, market: MarketData | None = None) -> tuple[object, Greeks]:
        market = market or MarketData(backend=self.backend)
        return pricing_registry.price_with_greeks(option, market)

    def supports(self, option) -> bool:
        return pricing_registry.supports(option)
