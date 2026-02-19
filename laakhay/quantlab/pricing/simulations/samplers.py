"""Distribution samplers for simulations."""

from laakhay.quantlab.backend import Backend

from ..utils import infer_backend
from .base import BaseSampler


class NormalSampler(BaseSampler):
    """Normal distribution sampler with backend support."""

    def _raw_sample(self, shape: tuple[int, ...], backend: Backend):
        """Draw Z ~ N(0,1) using backend."""
        return backend.randn(shape)

    @infer_backend
    def sample(
        self,
        shape: tuple[int, ...],
        stratify: bool = False,
        standardize: bool = True,
        backend: Backend | None = None,
    ):
        """Sample from normal distribution."""
        if stratify:
            return self.stratified_sample(shape, backend)

        z = self._raw_sample(shape, backend)
        return z if standardize else backend.add(backend.mul(z, self._std), self._mean)

    def stratified_sample(self, shape: tuple[int, ...], backend: Backend):
        """Stratified sampling using inverse transform."""
        import numpy as np
        from scipy.stats import norm

        n_total = int(np.prod(shape))
        u = (np.arange(1, n_total + 1) - 0.5) / n_total
        rng = np.random.default_rng(42)
        rng.shuffle(u)
        z = norm.ppf(u).reshape(shape)
        return backend.array(z)

    def __repr__(self) -> str:
        return f"NormalSampler(mean={self._mean:.4f}, std={self._std:.4f})"
