"""Base classes for stochastic simulations."""

from abc import ABC, abstractmethod
from laakhay.quantlab.backend import Backend
from ..utils import infer_backend


class BaseSampler(ABC):
    """Base sampler for distributions with abstract backend support."""

    def __init__(self, mean: float = 0.0, std: float = 1.0):
        if std <= 0:
            raise ValueError(f"Standard deviation must be positive, got {std}")
        self._mean = float(mean)
        self._std = float(std)

    @abstractmethod
    def _raw_sample(self, shape: tuple[int, ...], backend: Backend) -> object:
        """Draw from standardized distribution using backend."""
        pass

    @abstractmethod
    def sample(
        self,
        shape: tuple[int, ...],
        stratify: bool = False,
        standardize: bool = False,
        backend: Backend | None = None,
    ) -> object:
        """Sample from distribution using backend."""
        pass

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def std(self) -> float:
        return self._std


class BaseSimulation(ABC):
    """Base class for stochastic process simulations."""

    def __init__(self, sampler: BaseSampler):
        self.sampler = sampler

    @abstractmethod
    def generate_paths(
        self, n_paths: int, n_steps: int, expiry: float, backend: Backend | None = None, **kwargs
    ) -> object:
        """Generate stochastic process paths using backend."""
        pass
