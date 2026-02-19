"""Modular Black-Scholes pricer with registry pattern."""

from __future__ import annotations

from laakhay.quantlab.backend import get_backend

from ...greeks import Greeks
from ...market import MarketData
from .formulas.asian import GeometricAsianCallFormula, GeometricAsianPutFormula
from .formulas.barrier import (
    DownAndInCallFormula,
    DownAndInPutFormula,
    DownAndOutCallFormula,
    DownAndOutPutFormula,
    UpAndInCallFormula,
    UpAndInPutFormula,
    UpAndOutCallFormula,
    UpAndOutPutFormula,
)
from .formulas.digital import DigitalCallFormula, DigitalPutFormula
from .formulas.european import EuropeanCallFormula, EuropeanPutFormula
from .registry import pricing_registry


class BlackScholesPricer:
    """Modular Black-Scholes pricer."""

    def __init__(self, backend=None, market: MarketData | None = None):
        self.backend = backend or get_backend()
        self.default_market = market or MarketData(backend=self.backend)
        self._setup_registry()

    def _setup_registry(self):
        pricing_registry.clear()

        from ...options import (
            DigitalCall,
            DigitalPut,
            DownAndInCall,
            DownAndInPut,
            DownAndOutCall,
            DownAndOutPut,
            EuropeanCall,
            EuropeanPut,
            GeometricAsianCall,
            GeometricAsianPut,
            UpAndInCall,
            UpAndInPut,
            UpAndOutCall,
            UpAndOutPut,
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
        market = market or self.default_market
        return pricing_registry.price(option, market)

    def price_with_greeks(self, option, market: MarketData | None = None) -> tuple[object, Greeks]:
        market = market or self.default_market
        return pricing_registry.price_with_greeks(option, market)

    def implied_vol(
        self,
        option,
        price: object,
        market: MarketData,
        vol_bounds: tuple = (1e-8, 5.0),
        tolerance: float = 1e-8,
    ) -> object:
        """Calculate implied volatility using bisection method."""

        def _bisection_solve(objective_func, low, high, tol, max_iter=100):
            """Simple bisection root finder."""
            try:
                obj_low = objective_func(low)
                obj_high = objective_func(high)

                # Check if root exists in interval
                if obj_low * obj_high > 0:
                    # Try expanding bounds
                    if obj_low > 0:  # Need lower vol
                        low = 1e-10
                        obj_low = objective_func(low)
                    else:  # Need higher vol
                        high = 10.0
                        obj_high = objective_func(high)

                    if obj_low * obj_high > 0:
                        return float("nan")

                # Bisection iteration
                for _ in range(max_iter):
                    mid = (low + high) / 2
                    obj_mid = objective_func(mid)

                    if abs(obj_mid) < tol:
                        return mid

                    if obj_low * obj_mid < 0:
                        high = mid
                        obj_high = obj_mid
                    else:
                        low = mid
                        obj_low = obj_mid

                    if high - low < tol:
                        return mid

                return (low + high) / 2

            except (ValueError, RuntimeError):
                return float("nan")

        # Handle scalar case
        if isinstance(price, (int, float)):
            if price <= 0:
                return float("nan")

            def objective(vol_guess: float) -> float:
                return float(self.price(option, market.with_vol(vol_guess))) - price

            return _bisection_solve(objective, vol_bounds[0], vol_bounds[1], tolerance)

        # Handle array case
        price = self.backend.convert(price)
        spot = market.spot
        rate = market.rate

        result = self.backend.zeros_like(price)
        result = self.backend.add(result, float("nan"))  # Initialize with NaN

        # Process each element
        price_flat = self.backend.to_numpy(price).flatten()

        # Handle scalar or array spot/rate
        if self.backend.ndim(spot) == 0:
            spot_flat = [float(spot)] * len(price_flat)
        else:
            spot_flat = self.backend.to_numpy(spot).flatten()

        if self.backend.ndim(rate) == 0:
            rate_flat = [float(rate)] * len(price_flat)
        else:
            rate_flat = self.backend.to_numpy(rate).flatten()

        for i, (p, s, r) in enumerate(zip(price_flat, spot_flat, rate_flat, strict=False)):
            if p <= 0:
                continue

            def objective(vol_guess: float, s=s, r=r, p=p) -> float:
                return float(
                    self.price(
                        option,
                        MarketData(spot=s, rate=r, vol=vol_guess, backend=self.backend),
                    )
                ) - float(p)

            iv = _bisection_solve(objective, vol_bounds[0], vol_bounds[1], tolerance)

            # Convert result to numpy for assignment, then back
            result_np = self.backend.to_numpy(result)
            result_np.flat[i] = iv
            result = self.backend.convert(result_np)

        return result

    def supports(self, option) -> bool:
        return pricing_registry.supports(option)
