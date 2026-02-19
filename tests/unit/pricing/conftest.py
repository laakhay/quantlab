"""Shared fixtures for pricing testing in QuantLab."""

import pytest
import numpy as np
from laakhay.quantlab.backend import get_backend, list_backends


@pytest.fixture(params=["numpy"])  # Start with numpy, can add others if needed
def backend(request):
    """Fixture to test backends."""
    return get_backend(request.param)


@pytest.fixture
def numpy_backend():
    """NumPy backend specifically."""
    return get_backend("numpy")


@pytest.fixture
def sample_prices():
    """Sample price data for testing."""
    np.random.seed(42)
    return np.array([100.0, 102.0, 98.0, 105.0, 95.0])


@pytest.fixture
def sample_price_paths():
    """Sample price paths for path-dependent option testing."""
    np.random.seed(42)
    # 3 paths, 5 time steps each
    return np.array(
        [
            [100.0, 102.0, 98.0, 105.0, 95.0],
            [100.0, 99.0, 101.0, 104.0, 108.0],
            [100.0, 103.0, 106.0, 102.0, 99.0],
        ]
    )


@pytest.fixture
def basic_params():
    """Basic option parameters."""
    return {"strike": 100.0, "expiry": 1.0}


@pytest.fixture
def tolerance():
    """Numerical tolerance for float comparisons."""
    return 1e-10
