"""Gaussian/Normal distribution sampler."""

from __future__ import annotations

from typing import Any

from ...backend import ArrayBackend, active_backend
from ...types import Shape
from .base import BaseSampler


class GaussianSampler(BaseSampler):
    """Gaussian/Normal distribution sampler."""

    def _raw_sample(self, shape: Shape, backend: ArrayBackend, key: Any = None) -> Any:
        """Draw from standard normal N(0,1) using backend.

        Args:
            shape: Shape of samples to draw
            backend: Backend to use for sampling
            key: Random key (for JAX backend)

        Returns:
            Array of samples from N(0,1)
        """
        if not hasattr(backend, "random_normal"):
            raise NotImplementedError(f"Random sampling not implemented for {backend.name}")
        return backend.random_normal(key, shape)

    def sample(
        self,
        shape: Shape,
        backend: ArrayBackend | None = None,
        key: Any = None,
        standardize: bool = False,
    ) -> Any:
        """Sample from Gaussian distribution.

        Args:
            shape: Shape of samples to draw
            backend: Backend to use (defaults to active backend)
            key: Random key (for JAX backend)
            standardize: If True, return N(0,1) samples

        Returns:
            Array of samples from N(mean, std^2)
        """
        if backend is None:
            backend = active_backend()

        z = self._raw_sample(shape, backend=backend, key=key)

        if standardize:
            return z

        return backend.add(backend.mul(z, backend.array(self._std)), backend.array(self._mean))

    def pdf(
        self,
        x: Any,
        backend: ArrayBackend | None = None,
        standardize: bool = False,
    ) -> Any:
        """Compute Gaussian PDF at x.

        Args:
            x: Points at which to evaluate PDF
            backend: Backend to use (defaults to active backend)
            standardize: If True, compute standard normal PDF

        Returns:
            PDF values at x
        """
        if backend is None:
            backend = active_backend()

        x = backend.array(x) if not backend.is_array(x) else x

        pi = 3.141592653589793
        sqrt_2pi = (2.0 * pi) ** 0.5

        if standardize:
            norm_const = 1.0 / sqrt_2pi
            exponent = backend.mul(backend.array(-0.5), backend.pow(x, 2))
        else:
            norm_const = 1.0 / (self._std * sqrt_2pi)
            z = backend.div(backend.sub(x, backend.array(self._mean)), backend.array(self._std))
            exponent = backend.mul(backend.array(-0.5), backend.pow(z, 2))

        return backend.mul(backend.array(norm_const), backend.exp(exponent))

    def cdf(
        self,
        x: Any,
        backend: ArrayBackend | None = None,
        standardize: bool = False,
    ) -> Any:
        """Compute Gaussian CDF at x.

        Args:
            x: Points at which to evaluate CDF
            backend: Backend to use (defaults to active backend)
            standardize: If True, compute standard normal CDF

        Returns:
            CDF values at x
        """
        if backend is None:
            backend = active_backend()

        x = backend.array(x) if not backend.is_array(x) else x

        if standardize:
            z = x
        else:
            z = backend.div(backend.sub(x, backend.array(self._mean)), backend.array(self._std))

        return backend.norm_cdf(z)

    def ppf(
        self,
        q: Any,
        backend: ArrayBackend | None = None,
        standardize: bool = False,
    ) -> Any:
        """Compute inverse CDF (quantile function) at q.

        Args:
            q: Quantiles (must be in [0, 1])
            backend: Backend to use (defaults to active backend)
            standardize: If True, compute standard normal quantiles

        Returns:
            Quantile values
        """
        if backend is None:
            backend = active_backend()

        q = backend.array(q) if not backend.is_array(q) else q

        if backend.to_numpy(backend.min(q)) < 0 or backend.to_numpy(backend.max(q)) > 1:
            raise ValueError("Quantiles must be in [0, 1]")

        z = backend.norm_ppf(q)

        if standardize:
            return z

        return backend.add(backend.mul(z, backend.array(self._std)), backend.array(self._mean))

    def __repr__(self) -> str:
        """String representation of Gaussian sampler."""
        return f"GaussianSampler(mean={self._mean}, std={self._std})"
