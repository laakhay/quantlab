"""Backend system tests - registry, autodiff, random, devices."""

import numpy as np
import pytest

from laakhay.quantlab.backend import (
    ArrayBackend,
    backend,
    convert_array,
    get_backend,
    has_backend,
    infer_backend,
    infer_backend_from_arrays,
    list_backends,
    set_default,
)
from laakhay.quantlab.backend.device import Device, get_device
from laakhay.quantlab.exceptions import BackendNotFoundError, MixedBackendError


class TestBackendRegistry:
    """Test backend registry functionality."""

    def test_list_backends(self):
        """Test listing available backends."""
        backends = list_backends()

        assert isinstance(backends, list)
        assert "fallback" in backends
        # At least one real backend should be available in CI
        assert any(b in backends for b in ["numpy", "jax", "torch"])

    def test_has_backend(self):
        """Test backend availability check."""
        assert has_backend("fallback") is False  # fallback doesn't count

        # Check real backends
        for name in ["numpy", "jax", "torch"]:
            result = has_backend(name)
            assert isinstance(result, bool)

    def test_get_backend(self):
        """Test getting backend instance."""
        # Fallback always available
        fallback = get_backend("fallback")
        assert fallback.name == "fallback"

        # Default backend
        default = get_backend()
        assert default.name in list_backends()

        # Non-existent backend
        with pytest.raises(BackendNotFoundError):
            get_backend("nonexistent")

    def test_set_default(self):
        """Test setting default backend."""
        current_backends = list_backends()

        if "numpy" in current_backends:
            set_default("numpy")
            b = get_backend()
            assert b.name == "numpy"

        # Can't set non-existent as default
        with pytest.raises(ValueError):
            set_default("nonexistent")

    def test_infer_backend(self):
        """Test inferring backend from array."""
        if has_backend("numpy"):
            import numpy as np

            arr = np.array([1, 2, 3])
            assert infer_backend(arr) == "numpy"

        if has_backend("torch"):
            import torch

            arr = torch.tensor([1, 2, 3])
            assert infer_backend(arr) == "torch"

        # Unknown type
        assert infer_backend("not an array") is None

    def test_infer_backend_from_arrays(self):
        """Test inferring common backend from multiple arrays."""
        if has_backend("numpy"):
            b = backend("numpy")
            a1 = b.array([1, 2, 3])
            a2 = b.array([4, 5, 6])

            # Same backend
            assert infer_backend_from_arrays(a1, a2) == "numpy"

            # With None
            assert infer_backend_from_arrays(a1, None, a2) == "numpy"

        # Mixed backends
        if has_backend("numpy") and has_backend("torch"):
            b1 = backend("numpy")
            b2 = backend("torch")
            a1 = b1.array([1, 2, 3])
            a2 = b2.array([4, 5, 6])

            with pytest.raises(MixedBackendError):
                infer_backend_from_arrays(a1, a2)

    def test_convert_array(self):
        """Test array conversion between backends."""
        if has_backend("numpy") and has_backend("torch"):
            # NumPy to Torch
            np_arr = np.array([1, 2, 3])
            torch_arr = convert_array(np_arr, "torch", "numpy")

            b_torch = backend("torch")
            assert b_torch.is_array(torch_arr)
            np.testing.assert_array_equal(b_torch.to_numpy(torch_arr), [1, 2, 3])

            # Auto-infer source
            torch_arr2 = convert_array(np_arr, "torch")
            assert b_torch.is_array(torch_arr2)


class TestArrayBackendWrapper:
    """Test ArrayBackend high-level wrapper."""

    def test_initialization(self):
        """Test ArrayBackend initialization."""
        # Default
        b1 = ArrayBackend()
        assert b1.name in list_backends()

        # Specific backend
        if has_backend("numpy"):
            b2 = ArrayBackend("numpy")
            assert b2.name == "numpy"

        # From instance
        if has_backend("numpy"):
            numpy_backend = get_backend("numpy")
            b3 = ArrayBackend(numpy_backend)
            assert b3.name == "numpy"

    def test_from_array(self):
        """Test creating backend from array."""
        if has_backend("numpy"):
            arr = np.array([1, 2, 3])
            b = ArrayBackend.from_array(arr)
            assert b.name == "numpy"

    def test_supports_autodiff(self):
        """Test autodiff support check."""
        if has_backend("jax"):
            b = ArrayBackend("jax")
            assert b.supports_autodiff is True

        if has_backend("torch"):
            b = ArrayBackend("torch")
            assert b.supports_autodiff is True

        if has_backend("numpy"):
            b = ArrayBackend("numpy")
            assert b.supports_autodiff is False

    def test_attribute_delegation(self):
        """Test that attributes are delegated to backend."""
        b = ArrayBackend()

        # Should have backend methods
        assert hasattr(b, "array")
        assert hasattr(b, "add")
        assert hasattr(b, "sum")


class TestAutodiff:
    """Test automatic differentiation capabilities."""

    @pytest.fixture
    def b_autodiff(self):
        """Get backend with autodiff support."""
        for name in ["jax", "torch"]:
            if has_backend(name):
                return backend(name)
        pytest.skip("No autodiff backend available")

    def test_grad_scalar_function(self, b_autodiff):
        """Test gradient of scalar function."""

        def f(x):
            # f(x) = x^2 + 3x + 1
            return b_autodiff.add(
                b_autodiff.add(b_autodiff.mul(x, x), b_autodiff.mul(3, x)), 1
            )

        grad_f = b_autodiff.grad(f)

        # f'(x) = 2x + 3
        x = b_autodiff.array(2.0)
        grad = grad_f(x)

        expected = 2 * 2 + 3  # 7
        np.testing.assert_allclose(float(b_autodiff.to_numpy(grad)), expected)

    def test_grad_vector_input(self, b_autodiff):
        """Test gradient with vector input."""

        def f(x):
            # f(x) = sum(x^2)
            return b_autodiff.sum(b_autodiff.mul(x, x))

        grad_f = b_autodiff.grad(f)

        # f'(x) = 2x
        x = b_autodiff.array([1.0, 2.0, 3.0])
        grad = grad_f(x)

        expected = [2.0, 4.0, 6.0]
        np.testing.assert_allclose(b_autodiff.to_numpy(grad), expected)

    def test_value_and_grad(self, b_autodiff):
        """Test computing value and gradient together."""

        def f(x):
            # f(x) = exp(x) * sin(x)
            return b_autodiff.mul(b_autodiff.exp(x), b_autodiff.sin(x))

        value_and_grad_f = b_autodiff.value_and_grad(f)

        x = b_autodiff.array(1.0)
        value, grad = value_and_grad_f(x)

        # Check value
        expected_value = np.exp(1) * np.sin(1)
        np.testing.assert_allclose(
            float(b_autodiff.to_numpy(value)), expected_value, rtol=1e-6
        )

        # Check gradient (using product rule)
        # f'(x) = exp(x)*sin(x) + exp(x)*cos(x) = exp(x)*(sin(x) + cos(x))
        expected_grad = np.exp(1) * (np.sin(1) + np.cos(1))
        np.testing.assert_allclose(
            float(b_autodiff.to_numpy(grad)), expected_grad, rtol=1e-6
        )

    def test_grad_with_array_output(self, b_autodiff):
        """Test gradient when function returns array."""

        def f(x):
            # Returns [x^2, 2x, x+1]
            return b_autodiff.stack(
                [
                    b_autodiff.mul(x, x),
                    b_autodiff.mul(2, x),
                    b_autodiff.add(x, 1),
                ]
            )

        # Grad should automatically sum the output
        grad_f = b_autodiff.grad(f)

        x = b_autodiff.array(3.0)
        grad = grad_f(x)

        # Sum of derivatives: 2x + 2 + 1 = 2*3 + 2 + 1 = 9
        expected = 9.0
        np.testing.assert_allclose(float(b_autodiff.to_numpy(grad)), expected)

    def test_higher_order_derivatives(self, b_autodiff):
        """Test second derivatives if supported."""
        if b_autodiff.name != "jax":
            pytest.skip("Higher-order derivatives best supported in JAX")

        def f(x):
            # f(x) = x^3
            return b_autodiff.mul(b_autodiff.mul(x, x), x)

        # First derivative: 3x^2
        grad_f = b_autodiff.grad(f)

        # Second derivative: 6x
        grad_grad_f = b_autodiff.grad(grad_f)

        x = b_autodiff.array(2.0)
        first = grad_f(x)
        second = grad_grad_f(x)

        np.testing.assert_allclose(float(b_autodiff.to_numpy(first)), 12.0)  # 3*2^2
        np.testing.assert_allclose(float(b_autodiff.to_numpy(second)), 12.0)  # 6*2

    def test_vmap(self, b_autodiff):
        """Test vectorized mapping."""

        def f(x):
            return b_autodiff.mul(x, x)

        vmap_f = b_autodiff.vmap(f)

        xs = b_autodiff.array([1.0, 2.0, 3.0, 4.0])
        result = vmap_f(xs)

        expected = [1.0, 4.0, 9.0, 16.0]
        np.testing.assert_allclose(b_autodiff.to_numpy(result), expected)

    def test_jit(self, b_autodiff):
        """Test JIT compilation if supported."""

        def f(x):
            # Complex function that benefits from JIT
            result = x
            for _ in range(5):
                result = b_autodiff.add(b_autodiff.mul(result, result), x)
            return result

        jit_f = b_autodiff.jit(f)

        x = b_autodiff.array(0.5)

        # Results should match
        result_normal = f(x)
        result_jit = jit_f(x)

        np.testing.assert_allclose(
            b_autodiff.to_numpy(result_normal), b_autodiff.to_numpy(result_jit)
        )


class TestRandomGeneration:
    """Test random number generation."""

    @pytest.fixture
    def b_random(self):
        """Get backend with random support."""
        for name in ["jax", "torch", "numpy"]:
            if has_backend(name):
                b = backend(name)
                if hasattr(b, "random_key"):
                    return b
        pytest.skip("No random generation available")

    def test_random_key(self, b_random):
        """Test random key generation."""
        key1 = b_random.random_key(42)
        key2 = b_random.random_key(42)
        key3 = b_random.random_key(43)

        # Same seed should give same key
        if b_random.name == "jax":
            assert np.array_equal(key1, key2)
            assert not np.array_equal(key1, key3)

    def test_random_normal(self, b_random):
        """Test normal distribution generation."""
        key = b_random.random_key(42)
        samples = b_random.random_normal(key, (1000,))

        # Check shape
        assert b_random.shape(samples) == (1000,)

        # Check statistics (approximately)
        mean = b_random.mean(samples)
        std = b_random.std(samples)

        np.testing.assert_allclose(float(b_random.to_numpy(mean)), 0.0, atol=0.1)
        np.testing.assert_allclose(float(b_random.to_numpy(std)), 1.0, atol=0.1)

    def test_random_uniform(self, b_random):
        """Test uniform distribution generation."""
        if not hasattr(b_random, "random_uniform"):
            pytest.skip("random_uniform not implemented")

        key = b_random.random_key(42)
        low, high = -2.0, 3.0
        samples = b_random.random_uniform(key, (1000,), low=low, high=high)

        # Check bounds
        min_val = float(b_random.to_numpy(b_random.min(samples)))
        max_val = float(b_random.to_numpy(b_random.max(samples)))

        assert min_val >= low
        assert max_val <= high

        # Check mean
        mean = b_random.mean(samples)
        expected_mean = (low + high) / 2
        np.testing.assert_allclose(
            float(b_random.to_numpy(mean)), expected_mean, atol=0.1
        )

    def test_random_reproducibility(self, b_random):
        """Test that same seed gives same results."""
        key1 = b_random.random_key(42)
        samples1 = b_random.random_normal(key1, (100,))

        key2 = b_random.random_key(42)
        samples2 = b_random.random_normal(key2, (100,))

        np.testing.assert_array_equal(
            b_random.to_numpy(samples1), b_random.to_numpy(samples2)
        )

    def test_random_split(self, b_random):
        """Test key splitting for independent streams."""
        if not hasattr(b_random, "random_split"):
            pytest.skip("random_split not implemented")

        key = b_random.random_key(42)
        keys = b_random.random_split(key, 3)

        assert len(keys) == 3

        # Generate samples from each key
        samples = []
        for k in keys:
            s = b_random.random_normal(k, (100,))
            samples.append(b_random.to_numpy(s))

        # Samples should be different
        assert not np.array_equal(samples[0], samples[1])
        assert not np.array_equal(samples[1], samples[2])


class TestDeviceManagement:
    """Test device management functionality."""

    @pytest.fixture
    def b_device(self):
        """Get backend with device support."""
        for name in ["torch", "jax"]:
            if has_backend(name):
                b = backend(name)
                if hasattr(b, "is_gpu_available"):
                    return b
        pytest.skip("No device management available")

    def test_device_class(self):
        """Test Device class functionality."""
        cpu = Device.cpu()
        assert cpu.type == "cpu"
        assert str(cpu) == "cpu"

        gpu = Device.gpu(0)
        assert gpu.type == "cuda"
        assert str(gpu) == "cuda:0"

        # Equality
        assert cpu == Device.cpu()
        assert cpu == "cpu"
        assert gpu != cpu

    def test_gpu_availability(self, b_device):
        """Test GPU availability check."""
        gpu_available = b_device.is_gpu_available()
        assert isinstance(gpu_available, bool)

    def test_array_device(self, b_device):
        """Test getting device from array."""
        arr = b_device.ones((3, 4))
        device = get_device(arr)

        assert isinstance(device, Device)
        assert device.type in ["cpu", "cuda"]

    def test_to_device(self, b_device):
        """Test moving arrays between devices."""
        if not hasattr(b_device, "to_device"):
            pytest.skip("to_device not implemented")

        arr = b_device.ones((3, 4))

        # Move to CPU
        cpu_arr = b_device.to_device(arr, Device.cpu())
        assert get_device(cpu_arr).type == "cpu"

        # If GPU available, test GPU transfer
        if b_device.is_gpu_available():
            gpu_arr = b_device.to_device(arr, Device.gpu())
            assert get_device(gpu_arr).type == "cuda"

    def test_device_in_creation(self, b_device):
        """Test creating arrays on specific device."""
        if not hasattr(b_device, "device"):
            pytest.skip("device parameter not supported")

        # CPU creation
        cpu_arr = b_device.zeros((2, 3), device=Device.cpu())
        assert get_device(cpu_arr).type == "cpu"

        # GPU creation if available
        if b_device.is_gpu_available():
            gpu_arr = b_device.ones((2, 3), device=Device.gpu())
            assert get_device(gpu_arr).type == "cuda"


class TestBackendIntegration:
    """Integration tests across backend features."""

    def test_autodiff_with_special_functions(self):
        """Test autodiff with special functions if available."""
        if has_backend("jax"):
            b = backend("jax")

            if hasattr(b, "erf"):

                def f(x):
                    # f(x) = x * erf(x)
                    return b.mul(x, b.erf(x))

                grad_f = b.grad(f)
                x = b.array(1.0)
                grad = grad_f(x)

                # Gradient exists and is finite
                assert np.isfinite(float(b.to_numpy(grad)))

    def test_random_with_device(self):
        """Test random generation on different devices."""
        for name in ["torch", "jax"]:
            if has_backend(name):
                b = backend(name)

                if hasattr(b, "random_key") and hasattr(b, "device"):
                    key = b.random_key(42)

                    # Generate on CPU
                    cpu_samples = b.random_normal(key, (100,), device=Device.cpu())
                    assert get_device(cpu_samples).type == "cpu"

                    # Generate on GPU if available
                    if b.is_gpu_available():
                        gpu_samples = b.random_normal(key, (100,), device=Device.gpu())
                        assert get_device(gpu_samples).type == "cuda"

    def test_mixed_precision(self):
        """Test operations with mixed precision if supported."""
        if has_backend("torch"):
            b = backend("torch")
            import torch

            # Create arrays with different dtypes
            a_float32 = b.array([1, 2, 3], dtype=torch.float32)
            a_float64 = b.array([4, 5, 6], dtype=torch.float64)

            # Operations should handle mixed precision
            result = b.add(a_float32, a_float64)
            assert b.is_array(result)

    def test_backend_switching(self):
        """Test switching between backends in same session."""
        available = [name for name in ["numpy", "jax", "torch"] if has_backend(name)]

        if len(available) >= 2:
            # Create arrays with different backends
            b1 = backend(available[0])
            b2 = backend(available[1])

            a1 = b1.array([1, 2, 3])
            a2 = b2.array([4, 5, 6])

            # Each maintains its type
            assert b1.is_array(a1)
            assert b2.is_array(a2)

            # Can convert between them
            a1_converted = convert_array(a1, available[1])
            assert b2.is_array(a1_converted)

            # Now can operate
            result = b2.add(a1_converted, a2)
            np.testing.assert_array_equal(b2.to_numpy(result), [5, 7, 9])
