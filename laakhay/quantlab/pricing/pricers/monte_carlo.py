"""Monte Carlo pricing engine."""

from __future__ import annotations
import itertools
from math import ceil
from ..market import MarketData
from ..simulations.samplers import NormalSampler
from ..simulations.models import GeometricBrownianMotionSimulation
from laakhay.quantlab.backend import get_backend


class MonteCarloPricer:
    """Monte Carlo engine for option pricing."""

    def __init__(
        self,
        market: MarketData | None = None,
        sampler=None,
        n_paths: int = 20_000,
        steps_per_year: int = 252,
        antithetic: bool = True,
        cache: bool = True,
        moment_match: bool = True,
        stratify: bool = False,
        backend=None,
    ):
        self.backend = backend or get_backend()
        self.default_market = market or MarketData(backend=self.backend)

        self.n_paths = n_paths
        self.steps_per_year = steps_per_year
        self.antithetic = antithetic
        self.cache = cache
        self.moment_match = moment_match
        self.stratify = stratify

        self.sampler = sampler or NormalSampler()
        self.simulator = GeometricBrownianMotionSimulation(
            sampler=self.sampler,
            antithetic=self.antithetic,
            moment_match=self.moment_match,
            stratify=self.stratify,
        )

        self._path_cache = {}

    def get_paths(self, expiry: float, market: MarketData | None = None):
        """Generate price paths."""
        market = market or self.default_market
        n_steps = max(1, int(ceil(self.steps_per_year * expiry)))

        # Simple caching based on expiry and n_steps
        cache_key = (expiry, n_steps, market.spot, market.rate, market.vol)
        if self.cache and cache_key in self._path_cache:
            return self._path_cache[cache_key]

        paths = self.simulator.generate_paths(
            n_paths=self.n_paths,
            n_steps=n_steps,
            expiry=expiry,
            spot=float(market.spot),
            rate=float(market.rate),
            vol=float(market.vol),
            backend=self.backend,
        )

        if self.cache:
            self._path_cache[cache_key] = paths

        return paths

    def price(self, option, market: MarketData | None = None) -> object:
        """Price option using Monte Carlo."""
        market = market or self.default_market

        paths = self.get_paths(option.expiry, market)

        # Payoff calculation
        payoffs = option(paths, backend=self.backend)
        discount = self.backend.exp(self.backend.mul(backend.mul(-1, market.rate), option.expiry))

        return self.backend.mul(discount, self.backend.mean(payoffs))

    def supports(self, option) -> bool:
        """Check if option is supported."""
        from ..options.base import OptionContract

        return isinstance(option, OptionContract)
