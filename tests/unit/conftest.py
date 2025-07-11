"""Pytest configuration for unit tests."""

import pytest

from laakhay.quantlab.backend import get_backend, has_backend


@pytest.fixture(params=["numpy", "jax", "torch"])
def backend(request):
    """Fixture providing all available backends."""
    backend_name = request.param
    if not has_backend(backend_name):
        pytest.skip(f"{backend_name} backend not available")

    backend = get_backend(backend_name)

    # Skip if backend is fallback (not a real implementation)
    if hasattr(backend, "_is_fallback") or backend.name == "fallback":
        pytest.skip(f"{backend_name} backend not properly installed")

    return backend


@pytest.fixture
def numpy_backend():
    """Fixture providing numpy backend."""
    if not has_backend("numpy"):
        pytest.skip("numpy backend not available")
    return get_backend("numpy")


@pytest.fixture
def jax_backend():
    """Fixture providing JAX backend."""
    if not has_backend("jax"):
        pytest.skip("jax backend not available")
    return get_backend("jax")


@pytest.fixture
def torch_backend():
    """Fixture providing PyTorch backend."""
    if not has_backend("torch"):
        pytest.skip("torch backend not available")
    return get_backend("torch")


@pytest.fixture
def jax_key(jax_backend):
    """Fixture providing JAX random key."""
    if hasattr(jax_backend, "random_key"):
        return jax_backend.random_key(42)
    else:
        # Fallback for when random_key is not implemented
        import jax

        return jax.random.PRNGKey(42)


@pytest.fixture(params=[1e-6, 1e-8])
def tolerance(request):
    """Fixture providing different tolerance levels for numerical tests."""
    return request.param


@pytest.fixture(params=[100, 1000, 10000])
def sample_size(request):
    """Fixture providing different sample sizes for statistical tests."""
    return request.param


@pytest.fixture(params=[(0.0, 1.0), (5.0, 2.0), (-2.0, 0.5)])
def gaussian_params(request):
    """Fixture providing different Gaussian distribution parameters."""
    return request.param


@pytest.fixture(
    params=[
        (100.0, 0.05, 0.2, 1.0, 252),  # Standard equity parameters
        (1.0, 0.0, 0.3, 0.5, 100),  # FX-like parameters
        (50.0, 0.1, 0.4, 2.0, 504),  # High volatility, longer maturity
    ]
)
def gbm_params(request):
    """Fixture providing different GBM parameters."""
    spot, drift, volatility, time_to_maturity, num_steps = request.param
    return {
        "spot": spot,
        "drift": drift,
        "volatility": volatility,
        "time_to_maturity": time_to_maturity,
        "num_steps": num_steps,
    }
