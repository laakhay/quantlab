"""Unified Greeks calculation engine with automatic method selection."""

from __future__ import annotations

from laakhay.quantlab.backend import Backend, get_backend
from laakhay.quantlab.pricing.greeks import Greeks
from laakhay.quantlab.pricing.market import MarketData

from ...options.base import BaseLeg, OptionContract
from ...options.strategies import CompositeOptionContract
from ..black_scholes import BlackScholesPricer
from .base import GreeksCalculator, GreeksMethod, Priceable
from .utils import resolve_market_overrides


class GreeksEngine(GreeksCalculator):
    """Smart Greeks calculator that selects the best available method."""

    def __init__(
        self,
        backend: str | Backend | None = None,
        market: MarketData | None = None,
        preferred_methods: list[GreeksMethod] | None = None,
        *,
        spot: float | None = None,
        rate: float | None = None,
        vol: float | None = None,
    ) -> None:
        """Initialize with optional method preferences."""
        super().__init__(market=market, spot=spot, rate=rate, vol=vol)
        self.backend = get_backend(backend)
        self._create_market(self.backend)

        # Default method order
        self._methods = preferred_methods or [
            GreeksMethod.PRICER,
            GreeksMethod.AUTODIFF,
            GreeksMethod.FINITE_DIFF,
        ]

        # Initialize calculators lazily
        self._analytical = None
        self._autodiff = None
        self._finite_diff = None
        self._pricer = BlackScholesPricer(backend=self.backend)

        # Cache method selection
        self._method_cache: dict[type[OptionContract], GreeksMethod] = {}

    def supports(self, option: Priceable) -> bool:
        """Check if any method supports this option type."""
        return self._get_best_method(option) is not None

    def _get_best_method(self, option: Priceable) -> GreeksMethod | None:
        """Select best available method for option type."""
        if isinstance(option, (list, tuple, set)):
            methods = [self._get_best_method(opt) for opt in option]
            if None in methods:
                return None
            return max(methods)  # type: ignore

        if isinstance(option, OptionContract):
            cached = self._method_cache.get(type(option))
            if cached is not None:
                return cached

        for method in self._methods:
            if method == GreeksMethod.PRICER:
                if hasattr(self._pricer, "price_with_greeks") and self._pricer.supports(option):
                    if isinstance(option, OptionContract):
                        self._method_cache[type(option)] = method
                    return method
            elif method == GreeksMethod.AUTODIFF:
                ad = self._get_autodiff()
                if ad and ad.supports(option):
                    if isinstance(option, OptionContract):
                        self._method_cache[type(option)] = method
                    return method
            elif method == GreeksMethod.FINITE_DIFF:
                fd = self._get_finite_diff()
                if fd and fd.supports(option):
                    if isinstance(option, OptionContract):
                        self._method_cache[type(option)] = method
                    return method

        return None

    def _get_autodiff(self):
        """Get or create autodiff calculator."""
        if self._autodiff is None:
            try:
                from .autodiff import AutoDiffGreeksCalculator

                self._autodiff = AutoDiffGreeksCalculator(
                    backend=self.backend,
                    market=self.market,
                )
            except ImportError:
                return None
        return self._autodiff

    def _get_finite_diff(self):
        """Get or create finite difference calculator."""
        if self._finite_diff is None:
            try:
                from .finite_diff import FiniteDiffGreeksCalculator

                self._finite_diff = FiniteDiffGreeksCalculator(
                    backend=self.backend,
                    market=self.market,
                )
            except ImportError:
                return None
        return self._finite_diff

    def calculate_greeks(
        self,
        option: OptionContract | BaseLeg | CompositeOptionContract,
    ) -> Greeks:
        """Calculate Greeks for the given option."""
        return self.calculate(option)

    def calculate(
        self,
        option: Priceable,
        *,
        market: MarketData | None = None,
        spot: float | None = None,
        rate: float | None = None,
        vol: float | None = None,
        method: str | None = None,
    ) -> Greeks:
        """Calculate Greeks with automatic method selection."""
        final_market = resolve_market_overrides(self.market, market, spot, rate, vol)

        if isinstance(option, (list, tuple, set)):
            total_greeks = None
            for single_option in option:
                single_greeks = self.calculate(single_option, market=final_market, method=method)
                if total_greeks is None:
                    total_greeks = single_greeks
                else:
                    total_greeks = total_greeks + single_greeks
            return total_greeks or Greeks()

        if method is None or method.lower() in {"auto", "unified"}:
            best_method = self._get_best_method(option)
            if best_method is None:
                raise ValueError(f"No Greeks method supports {type(option)}")

            if best_method == GreeksMethod.PRICER:
                _, greeks = self._pricer.price_with_greeks(option, final_market)
                return greeks
            elif best_method == GreeksMethod.AUTODIFF:
                return self._get_autodiff().calculate(option, market=final_market)
            elif best_method == GreeksMethod.FINITE_DIFF:
                return self._get_finite_diff().calculate(option, market=final_market)

        # Explicit method
        method_norm = method.lower()
        if method_norm in {"analytical", "analytic", "pricer"}:
            _, greeks = self._pricer.price_with_greeks(option, final_market)
            return greeks
        elif method_norm in {"autodiff", "ad"}:
            calc = self._get_autodiff()
            if calc is None:
                raise RuntimeError("Autodiff not available")
            return calc.calculate(option, market=final_market)
        elif method_norm in {"finite_diff", "fd"}:
            calc = self._get_finite_diff()
            if calc is None:
                raise RuntimeError("Finite diff not available")
            return calc.calculate(option, market=final_market)
        else:
            raise ValueError(f"Unknown method '{method}'")
