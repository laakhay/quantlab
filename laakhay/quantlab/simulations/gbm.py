"""Geometric Brownian Motion simulation."""

from __future__ import annotations

from typing import Any

from ..backend import ArrayBackend, active_backend
from .samplers import GaussianSampler


class GeometricBrownianMotion:
    """Geometric Brownian Motion simulator.

    Simulates paths following dS(t) = μ * S(t) * dt + σ * S(t) * dW(t).
    """

    def __init__(
        self,
        spot: float,
        drift: float,
        volatility: float,
        time_to_maturity: float,
        num_steps: int,
        sampler: GaussianSampler | None = None,
    ):
        """Initialize GBM simulator.

        Args:
            spot: Initial asset price S(0)
            drift: Drift rate μ
            volatility: Volatility σ
            time_to_maturity: Total simulation time T
            num_steps: Number of time steps
            sampler: Gaussian sampler (defaults to standard normal)

        Raises:
            ValueError: If parameters are invalid
        """
        if spot <= 0:
            raise ValueError(f"Spot price must be positive, got {spot}")
        if volatility < 0:
            raise ValueError(f"Volatility must be non-negative, got {volatility}")
        if time_to_maturity <= 0:
            raise ValueError(
                f"Time to maturity must be positive, got {time_to_maturity}"
            )
        if num_steps <= 0:
            raise ValueError(f"Number of steps must be positive, got {num_steps}")

        self.spot = float(spot)
        self.drift = float(drift)
        self.volatility = float(volatility)
        self.time_to_maturity = float(time_to_maturity)
        self.num_steps = int(num_steps)

        self.dt = self.time_to_maturity / self.num_steps
        self.sampler = sampler or GaussianSampler(mean=0.0, std=1.0)

    def generate_shocks(
        self,
        num_paths: int,
        backend: ArrayBackend | None = None,
        key: Any = None,
        antithetic: bool = False,
    ) -> Any:
        """Generate random shocks for all paths and time steps.

        Args:
            num_paths: Number of paths to simulate
            backend: Backend to use (defaults to active backend)
            key: Random key (for JAX backend)
            antithetic: If True, use antithetic variates

        Returns:
            Array of shape (num_paths, num_steps) with standard normal shocks
        """
        if backend is None:
            backend = active_backend()

        shape = (num_paths, self.num_steps)
        shocks = self.sampler.sample(shape, backend=backend, key=key, standardize=True)

        if antithetic:
            if num_paths % 2 != 0:
                raise ValueError("Number of paths must be even for antithetic variates")
            half = num_paths // 2
            pos_shocks = shocks[:half]
            neg_shocks = backend.mul(backend.array(-1.0), pos_shocks)
            shocks = backend.concat([pos_shocks, neg_shocks], axis=0)

        return shocks

    def build_unit_paths(
        self,
        shocks: Any,
        backend: ArrayBackend | None = None,
    ) -> Any:
        """Build standardized Brownian motion paths from shocks.

        Args:
            shocks: Random shocks array of shape (num_paths, num_steps)
            backend: Backend to use (defaults to active backend)

        Returns:
            Array of shape (num_paths, num_steps + 1) with cumulative sum
            including initial zero
        """
        if backend is None:
            backend = active_backend()

        sqrt_dt = backend.array(self.dt**0.5)
        scaled_shocks = backend.mul(shocks, sqrt_dt)

        num_paths = backend.shape(shocks)[0]
        zeros = backend.zeros((num_paths, 1))

        path_list = [zeros]
        cumsum = backend.zeros((num_paths,))
        for i in range(self.num_steps):
            cumsum = backend.add(cumsum, scaled_shocks[:, i])
            path_list.append(backend.expand_dims(cumsum, axis=1))

        paths = backend.concat(path_list, axis=1)

        return paths

    def build_log_paths(
        self,
        unit_paths: Any,
        backend: ArrayBackend | None = None,
    ) -> Any:
        """Build paths in log space from unit Brownian paths.

        Args:
            unit_paths: Standardized Brownian paths of shape (num_paths, num_steps + 1)
            backend: Backend to use (defaults to active backend)

        Returns:
            Array of shape (num_paths, num_steps + 1) with log prices
        """
        if backend is None:
            backend = active_backend()

        time_steps = backend.linspace(0, self.time_to_maturity, self.num_steps + 1)

        drift_adjustment = self.drift - 0.5 * self.volatility**2
        drift_term = backend.mul(backend.array(drift_adjustment), time_steps)
        diffusion_term = backend.mul(backend.array(self.volatility), unit_paths)

        log_spot = backend.log(backend.array(self.spot))
        log_paths = backend.add(backend.add(log_spot, drift_term), diffusion_term)

        return log_paths

    def add_drift(
        self,
        paths: Any,
        backend: ArrayBackend | None = None,
    ) -> Any:
        """Add drift component to paths.

        Args:
            paths: Price paths of shape (num_paths, num_steps + 1)
            backend: Backend to use (defaults to active backend)

        Returns:
            Paths with drift added
        """
        if backend is None:
            backend = active_backend()

        time_steps = backend.linspace(0, self.time_to_maturity, self.num_steps + 1)
        drift_factor = backend.exp(backend.mul(backend.array(self.drift), time_steps))
        return backend.mul(paths, drift_factor)

    def scale_to_spot(
        self,
        log_paths: Any,
        backend: ArrayBackend | None = None,
    ) -> Any:
        """Convert log paths to actual price paths.

        Args:
            log_paths: Log price paths of shape (num_paths, num_steps + 1)
            backend: Backend to use (defaults to active backend)

        Returns:
            Price paths of shape (num_paths, num_steps + 1)
        """
        if backend is None:
            backend = active_backend()

        return backend.exp(log_paths)

    def generate_paths(
        self,
        num_paths: int,
        backend: ArrayBackend | None = None,
        key: Any = None,
        antithetic: bool = False,
        return_log: bool = False,
    ) -> Any:
        """Generate GBM paths in one shot.

        Args:
            num_paths: Number of paths to simulate
            backend: Backend to use (defaults to active backend)
            key: Random key (for JAX backend)
            antithetic: If True, use antithetic variates
            return_log: If True, return log paths instead of price paths

        Returns:
            Array of shape (num_paths, num_steps + 1) with simulated paths
        """
        if backend is None:
            backend = active_backend()

        shocks = self.generate_shocks(
            num_paths, backend=backend, key=key, antithetic=antithetic
        )
        unit_paths = self.build_unit_paths(shocks, backend=backend)
        log_paths = self.build_log_paths(unit_paths, backend=backend)

        if return_log:
            return log_paths

        return self.scale_to_spot(log_paths, backend=backend)

    def terminal_values(
        self,
        num_paths: int,
        backend: ArrayBackend | None = None,
        key: Any = None,
        antithetic: bool = False,
    ) -> Any:
        """Generate only terminal values S(T).

        Args:
            num_paths: Number of terminal values to simulate
            backend: Backend to use (defaults to active backend)
            key: Random key (for JAX backend)
            antithetic: If True, use antithetic variates

        Returns:
            Array of shape (num_paths,) with terminal values
        """
        if backend is None:
            backend = active_backend()

        shape = (num_paths,)
        shocks = self.sampler.sample(shape, backend=backend, key=key, standardize=True)

        if antithetic:
            if num_paths % 2 != 0:
                raise ValueError("Number of paths must be even for antithetic variates")
            half = num_paths // 2
            pos_shocks = shocks[:half]
            neg_shocks = backend.mul(backend.array(-1.0), pos_shocks)
            shocks = backend.concat([pos_shocks, neg_shocks], axis=0)

        drift_term = (self.drift - 0.5 * self.volatility**2) * self.time_to_maturity
        sqrt_t = backend.sqrt(backend.array(self.time_to_maturity))
        diffusion_term = self.volatility * backend.to_numpy(sqrt_t)

        log_spot = backend.log(backend.array(self.spot))
        log_terminal = backend.add(
            backend.add(log_spot, backend.array(drift_term)),
            backend.mul(backend.array(diffusion_term), shocks),
        )

        return backend.exp(log_terminal)
