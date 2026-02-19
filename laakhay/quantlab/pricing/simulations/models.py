"""Geometric Brownian Motion simulation model."""

from laakhay.quantlab.backend import Backend

from ..utils import infer_backend
from .base import BaseSampler, BaseSimulation
from .samplers import NormalSampler


class GeometricBrownianMotionSimulation(BaseSimulation):
    """Geometric Brownian Motion simulation."""

    def __init__(
        self,
        sampler: NormalSampler | None = None,
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
        sampler: BaseSampler | None = None,
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

        # Prepare parameters as arrays
        spot_arr = backend.array(spot)
        rate_arr = backend.array(rate)
        vol_arr = backend.array(vol)

        # Determine if we need vectorization across parameters
        is_vectorized = (
            backend.ndim(spot_arr) > 0 or backend.ndim(rate_arr) > 0 or backend.ndim(vol_arr) > 0
        )

        dt = expiry / n_steps
        sqrt_dt = backend.sqrt(backend.array(dt))

        # dlog(S) = (r - σ²/2)dt + σdW
        # drift_adj = (r - 0.5 * vol^2) * dt
        vol_sq = backend.power(vol_arr, 2)
        drift_adj = backend.mul(backend.subtract(rate_arr, backend.mul(0.5, vol_sq)), dt)

        if is_vectorized:
            # Expand shocks to (n_paths, n_steps, 1) to broadcast with (V,) params
            shocks_expanded = backend.expand_dims(shocks, axis=-1)
            diffusion = backend.mul(backend.mul(vol_arr, sqrt_dt), shocks_expanded)
            log_returns = backend.add(drift_adj, diffusion)  # (V,) + (P, S, V) -> (P, S, V)

            # Cumulative sums along time axis (axis 1)
            log_prices = backend.cumsum(log_returns, axis=1)  # (P, S, V)

            log_S0 = backend.log(spot_arr)  # (V,)
            # Reshape S0 for concatenation at t=0
            # S0 should be (n_paths, 1, V)
            log_S0_expanded = backend.expand_dims(backend.expand_dims(log_S0, axis=0), axis=0)
            log_S0_full = backend.mul(backend.ones((n_paths, 1, 1)), log_S0_expanded)

            log_prices_with_initial = backend.concatenate(
                [log_S0_full, backend.add(log_S0, log_prices)], axis=1
            )
        else:
            # Scalar case - standard 2D (n_paths, n_steps)
            diffusion = backend.mul(backend.mul(vol_arr, sqrt_dt), shocks)
            log_returns = backend.add(drift_adj, diffusion)
            log_prices = backend.cumsum(log_returns, axis=1)

            log_S0 = backend.log(spot_arr)
            log_S0_col = backend.mul(backend.ones((n_paths, 1)), log_S0)
            log_prices_with_initial = backend.concatenate(
                [log_S0_col, backend.add(log_S0, log_prices)], axis=1
            )

        return backend.exp(log_prices_with_initial)
