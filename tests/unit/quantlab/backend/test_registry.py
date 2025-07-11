"""Tests for backend registry."""

from laakhay.quantlab.backend import get_backend, has_backend, list_backends


class TestBackendRegistry:
    """Test backend registry functionality."""

    def test_backend_instantiation(self):
        """Test that we can instantiate registered backends."""
        # Test numpy backend
        numpy_backend = get_backend("numpy")
        assert numpy_backend.name == "numpy"

        # Test creating array
        arr = numpy_backend.array([1, 2, 3])
        assert numpy_backend.is_array(arr)

    def test_list_backends_content(self):
        """Test list_backends returns expected backends."""
        backends = list_backends()

        # Should at least have numpy
        assert "numpy" in backends

        # Check other backends if available
        expected_backends = ["numpy", "jax", "torch", "fallback"]
        for backend in backends:
            assert backend in expected_backends

    def test_has_backend(self):
        """Test has_backend functionality."""
        assert has_backend("numpy")
        assert not has_backend("nonexistent")

    def test_get_backend_specific(self):
        """Test getting specific backends."""
        if has_backend("torch"):
            torch_backend = get_backend("torch")
            assert torch_backend.name == "torch"

        if has_backend("jax"):
            jax_backend = get_backend("jax")
            assert jax_backend.name == "jax"
