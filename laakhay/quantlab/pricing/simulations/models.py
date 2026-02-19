"""Geometric Brownian Motion simulation model."""

from typing import Optional
from .base import BaseSimulation, BaseSampler
from .samplers import NormalSampler
from laakhay.quantlab.backend import Backend
from ..utils import infer_backend


class GeometricBrownianMotionSimulation(BaseSimulation):
    """Geometric Brownian Motion simulation."""

    def __init__(
        self,
        sampler: Optional[NormalSampler] = None,
        antithetic: bool = False,
        moment_match: bool = False,
        stratify: bool = False,
    ):
        super().__init__(sampler or NormalSampler())
        self.antithetic = antithetic
        self.moment_match = moment_match
        self.stratify = stratify

    @infer_backend
    def generate_shocks(
        self,
        n_paths: int,
        n_steps: int,
        sampler: Optional[BaseSampler] = None,
        backend: Backend | None = None,
    ) -> object:
        """Draw standardized shocks Z ∈ ℝⁿᵖᵃᵗʰˢ×ⁿˢᵗᵉᵖˢ."""
        if sampler is None:
            sampler = self.sampler

        Z = sampler.sample(
            shape=(n_paths, n_steps), stratify=self.stratify, standardize=True, backend=backend
        )

        if self.antithetic and isinstance(sampler, NormalSampler):
            half = n_paths // 2
            Zp = Z[:half]
            Zn = backend.mul(-1, Zp)
            if n_paths % 2 == 0:
                Z = backend.concatenate([Zp, Zn], axis=0)
            else:
                extra = sampler.sample(
                    (1, n_steps), stratify=self.stratify, standardize=True, backend=backend
                )
                Z = backend.concatenate([Zp, Zn, extra], axis=0)

        if self.moment_match:
            m = backend.mean(Z)
            s = backend.std(Z)
            Z = backend.divide(backend.add(Z, backend.mul(-1, m)), backend.add(s, 1e-12))

        return Z

    @infer_backend
    def generate_paths(
        self,
        n_paths: int,
        n_steps: int,
        expiry: float,
        backend: Backend | None = None,
        spot: float = 100.0,
        rate: float = 0.0,
        vol: float = 0.2,
        **kwargs,
    ) -> object:
        """Generate complete GBM paths."""
        shocks = self.generate_shocks(n_paths, n_steps, backend=backend)

        dt = expiry / n_steps
        sqrt_dt = backend.sqrt(backend.convert(dt))

        # dlog(S) = (r - σ²/2)dt + σdW
        drift_adj = backend.mul(
            backend.add(
                backend.convert(rate), backend.mul(-0.5, backend.power(backend.convert(vol), 2))
            ),
            dt,
        )
        diffusion = backend.mul(backend.mul(backend.convert(vol), sqrt_dt), shocks)
        log_returns = backend.add(drift_adj, diffusion)

        log_prices = backend.cumsum(log_returns, axis=1)

        log_S0 = backend.log(backend.convert(spot))
        log_S0_col = backend.mul(backend.ones((n_paths, 1)), log_S0)
        log_prices_with_initial = backend.concatenate(
            [log_S0_col, backend.add(log_S0, log_prices)], axis=1
        )

        return backend.exp(log_prices_with_initial)
