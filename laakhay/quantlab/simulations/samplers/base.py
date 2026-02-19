"""Base sampler for distributions with abstract backend support."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ...backend import ArrayBackend, active_backend
from ...types import Shape


class BaseSampler(ABC):
    """Base sampler for distributions with abstract backend support."""

    def __init__(self, mean: float = 0.0, std: float = 1.0):
        """Initialize base sampler.

        Args:
            mean: Mean of the distribution
            std: Standard deviation of the distribution

        Raises:
            ValueError: If std <= 0
        """
        if std <= 0:
            raise ValueError(f"Standard deviation must be positive, got {std}")
        self._mean = float(mean)
        self._std = float(std)
        self._mgf_cache: dict[float, float] = {}
        self._skew: float | None = None
        self._ex_kurtosis: float | None = None

    @abstractmethod
    def _raw_sample(self, shape: Shape, backend: ArrayBackend, key: Any = None) -> Any:
        """Draw from standardized distribution using backend.

        Args:
            shape: Shape of samples to draw
            backend: Backend to use for sampling
            key: Random key (for JAX backend)

        Returns:
            Array of samples from standardized distribution
        """
        pass

    @abstractmethod
    def sample(
        self,
        shape: Shape,
        backend: ArrayBackend | None = None,
        key: Any = None,
        standardize: bool = False,
    ) -> Any:
        """Sample from distribution using backend.

        Args:
            shape: Shape of samples to draw
            backend: Backend to use (defaults to active backend)
            key: Random key (for JAX backend)
            standardize: If True, return standardized samples

        Returns:
            Array of samples
        """
        pass

    @abstractmethod
    def pdf(
        self,
        x: Any,
        backend: ArrayBackend | None = None,
        standardize: bool = False,
    ) -> Any:
        """Compute PDF at x using backend.

        Args:
            x: Points at which to evaluate PDF
            backend: Backend to use (defaults to active backend)
            standardize: If True, compute standardized PDF

        Returns:
            PDF values at x
        """
        pass

    @abstractmethod
    def cdf(
        self,
        x: Any,
        backend: ArrayBackend | None = None,
        standardize: bool = False,
    ) -> Any:
        """Compute CDF at x using backend.

        Args:
            x: Points at which to evaluate CDF
            backend: Backend to use (defaults to active backend)
            standardize: If True, compute standardized CDF

        Returns:
            CDF values at x
        """
        pass

    @abstractmethod
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
            standardize: If True, compute standardized quantiles

        Returns:
            Quantile values
        """
        pass

    def _compute_moments(self, backend: ArrayBackend | None = None) -> tuple[float, float]:
        """Compute skew and excess kurtosis using backend.

        Args:
            backend: Backend to use (defaults to active backend)

        Returns:
            Tuple of (skew, excess_kurtosis)
        """
        if backend is None:
            backend = active_backend()

        n = 100_000
        if backend.name == "jax":
            key = backend.random_key(42)
            z = self._raw_sample((n,), backend=backend, key=key)
        else:
            z = self._raw_sample((n,), backend=backend)

        z_std = (z - self._mean) / self._std
        m3 = backend.mean(backend.pow(z_std, 3))
        m4 = backend.mean(backend.pow(z_std, 4))

        skew = float(backend.to_numpy(m3))
        ex_kurt = float(backend.to_numpy(m4)) - 3.0

        return skew, ex_kurt

    def mgf(self, t: float, backend: ArrayBackend | None = None) -> float:
        """Estimate MGF using backend with numerical stability.

        Args:
            t: Point at which to evaluate MGF
            backend: Backend to use (defaults to active backend)

        Returns:
            MGF value at t

        Raises:
            ValueError: If MGF is unstable or invalid
        """
        if backend is None:
            backend = active_backend()

        key = round(t, 8)
        if key not in self._mgf_cache:
            n = 100_000

            if backend.name == "jax":
                rng_key = backend.random_key(42)
                samples = self.sample((n,), backend=backend, key=rng_key)
            else:
                samples = self.sample((n,), backend=backend)

            tx = backend.mul(backend.array(t), samples)
            max_tx = backend.max(tx)

            exp_normalized = backend.exp(backend.sub(tx, max_tx))
            mgf_val = backend.mul(backend.exp(max_tx), backend.mean(exp_normalized))
            mgf_float = float(backend.to_numpy(mgf_val))

            if not backend.to_numpy(backend.isfinite(mgf_val)) or mgf_float <= 0:
                raise ValueError(f"MGF unstable for t={t}: got {mgf_float}")

            self._mgf_cache[key] = mgf_float

        return self._mgf_cache[key]

    @property
    def mean(self) -> float:
        """Mean of the distribution."""
        return self._mean

    @property
    def std(self) -> float:
        """Standard deviation of the distribution."""
        return self._std

    @property
    def variance(self) -> float:
        """Variance of the distribution."""
        return self._std**2

    @property
    def skew(self) -> float:
        """Skewness of the distribution."""
        if self._skew is None:
            self._skew, self._ex_kurtosis = self._compute_moments()
        return self._skew

    @property
    def ex_kurtosis(self) -> float:
        """Excess kurtosis of the distribution."""
        if self._ex_kurtosis is None:
            self._skew, self._ex_kurtosis = self._compute_moments()
        return self._ex_kurtosis

    @abstractmethod
    def __repr__(self) -> str:
        """String representation of the sampler."""
        pass
