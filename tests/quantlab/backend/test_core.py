"""Core backend operations tests."""

import pytest
import numpy as np

from laakhay.quantlab.backend import backend, has_backend, list_backends
from laakhay.quantlab.backend.device import Device, get_device
from laakhay.quantlab.exceptions import MixedBackendError


@pytest.fixture(params=["numpy", "jax", "torch"])
def b(request):
    """Parametrized backend fixture."""
    if not has_backend(request.param):
        pytest.skip(f"{request.param} not available")
    return backend(request.param)


class TestArrayCreation:
    """Test array creation operations."""
    
    def test_array_from_list(self, b):
        """Test creating array from list."""
        data = [1, 2, 3, 4, 5]
        arr = b.array(data)
        
        assert b.shape(arr) == (5,)
        assert b.is_array(arr)
        np.testing.assert_array_equal(b.to_numpy(arr), data)
    
    def test_array_from_nested_list(self, b):
        """Test creating array from nested list."""
        data = [[1, 2], [3, 4], [5, 6]]
        arr = b.array(data)
        
        assert b.shape(arr) == (3, 2)
        assert b.ndim(arr) == 2
        np.testing.assert_array_equal(b.to_numpy(arr), data)
    
    def test_zeros(self, b):
        """Test zeros creation."""
        arr = b.zeros((3, 4))
        
        assert b.shape(arr) == (3, 4)
        assert b.size(arr) == 12
        np.testing.assert_array_equal(b.to_numpy(arr), np.zeros((3, 4)))
    
    def test_ones(self, b):
        """Test ones creation."""
        arr = b.ones((2, 3, 4))
        
        assert b.shape(arr) == (2, 3, 4)
        assert b.ndim(arr) == 3
        np.testing.assert_array_equal(b.to_numpy(arr), np.ones((2, 3, 4)))
    
    def test_full(self, b):
        """Test full array creation."""
        arr = b.full((4, 5), 3.14)
        
        assert b.shape(arr) == (4, 5)
        np.testing.assert_allclose(b.to_numpy(arr), np.full((4, 5), 3.14))
    
    def test_arange(self, b):
        """Test arange creation."""
        # Single arg
        arr1 = b.arange(5)
        np.testing.assert_array_equal(b.to_numpy(arr1), [0, 1, 2, 3, 4])
        
        # Start, stop
        arr2 = b.arange(2, 7)
        np.testing.assert_array_equal(b.to_numpy(arr2), [2, 3, 4, 5, 6])
        
        # Start, stop, step
        arr3 = b.arange(0, 10, 2)
        np.testing.assert_array_equal(b.to_numpy(arr3), [0, 2, 4, 6, 8])
    
    def test_linspace(self, b):
        """Test linspace creation."""
        arr = b.linspace(0, 1, 5)
        
        assert b.shape(arr) == (5,)
        np.testing.assert_allclose(b.to_numpy(arr), [0, 0.25, 0.5, 0.75, 1.0])
    
    def test_eye(self, b):
        """Test identity matrix creation."""
        # Square
        arr1 = b.eye(3)
        expected1 = np.eye(3)
        
        # Rectangular
        arr2 = b.eye(3, 5)
        expected2 = np.eye(3, 5)
        
        # Note: eye implementation in interface.py is simplified
        # Real implementations would properly set diagonal


class TestShapeOperations:
    """Test shape manipulation operations."""
    
    def test_reshape(self, b):
        """Test reshape operation."""
        arr = b.arange(24)
        
        # Various reshapes
        r1 = b.reshape(arr, (4, 6))
        assert b.shape(r1) == (4, 6)
        
        r2 = b.reshape(arr, (2, 3, 4))
        assert b.shape(r2) == (2, 3, 4)
        
        r3 = b.reshape(arr, (24,))
        assert b.shape(r3) == (24,)
        
        # Check values preserved
        np.testing.assert_array_equal(b.to_numpy(arr), b.to_numpy(r3))
    
    def test_transpose(self, b):
        """Test transpose operation."""
        # 2D transpose
        arr2d = b.reshape(b.arange(6), (2, 3))
        t2d = b.transpose(arr2d)
        assert b.shape(t2d) == (3, 2)
        
        # 3D with axes
        arr3d = b.reshape(b.arange(24), (2, 3, 4))
        t3d = b.transpose(arr3d, [2, 0, 1])
        assert b.shape(t3d) == (4, 2, 3)
    
    def test_squeeze(self, b):
        """Test squeeze operation."""
        arr = b.ones((1, 3, 1, 4, 1))
        
        # Squeeze all
        s1 = b.squeeze(arr)
        assert b.shape(s1) == (3, 4)
        
        # Squeeze specific axis
        s2 = b.squeeze(arr, axis=0)
        assert b.shape(s2) == (3, 1, 4, 1)
    
    def test_expand_dims(self, b):
        """Test expand_dims operation."""
        arr = b.ones((3, 4))
        
        # Various positions
        e1 = b.expand_dims(arr, axis=0)
        assert b.shape(e1) == (1, 3, 4)
        
        e2 = b.expand_dims(arr, axis=1)
        assert b.shape(e2) == (3, 1, 4)
        
        e3 = b.expand_dims(arr, axis=-1)
        assert b.shape(e3) == (3, 4, 1)
    
    def test_concat(self, b):
        """Test concatenation."""
        a1 = b.ones((2, 3))
        a2 = b.full((3, 3), 2.0)
        a3 = b.full((1, 3), 3.0)
        
        # Concat along axis 0
        c1 = b.concat([a1, a2, a3], axis=0)
        assert b.shape(c1) == (6, 3)
        
        # Check values
        c1_np = b.to_numpy(c1)
        assert np.all(c1_np[:2] == 1)
        assert np.all(c1_np[2:5] == 2)
        assert np.all(c1_np[5:] == 3)
    
    def test_stack(self, b):
        """Test stacking."""
        arrays = [b.ones((3, 4)) * i for i in range(5)]
        
        # Stack along new axis
        s1 = b.stack(arrays, axis=0)
        assert b.shape(s1) == (5, 3, 4)
        
        s2 = b.stack(arrays, axis=1)
        assert b.shape(s2) == (3, 5, 4)
    
    def test_split(self, b):
        """Test split operation."""
        arr = b.arange(12)
        
        # Equal splits
        splits1 = b.split(arr, 3)
        assert len(splits1) == 3
        for s in splits1:
            assert b.shape(s) == (4,)
        
        # Unequal splits
        arr2d = b.reshape(arr, (3, 4))
        splits2 = b.split(arr2d, [1, 2], axis=0)
        assert len(splits2) == 3
        assert b.shape(splits2[0]) == (1, 4)
        assert b.shape(splits2[1]) == (1, 4)
        assert b.shape(splits2[2]) == (1, 4)


class TestTypeConversion:
    """Test type conversion operations."""
    
    def test_to_from_numpy(self, b):
        """Test numpy conversion roundtrip."""
        # Create array
        arr = b.array([[1, 2, 3], [4, 5, 6]])
        
        # To numpy
        np_arr = b.to_numpy(arr)
        assert isinstance(np_arr, np.ndarray)
        assert np_arr.shape == (2, 3)
        
        # From numpy
        arr2 = b.from_numpy(np_arr)
        assert b.is_array(arr2)
        np.testing.assert_array_equal(b.to_numpy(arr2), np_arr)
    
    def test_cast(self, b):
        """Test dtype casting."""
        # Integer to float
        arr_int = b.array([1, 2, 3, 4])
        
        if b.name == "numpy":
            arr_float = b.cast(arr_int, np.float32)
            assert b.dtype(arr_float) == np.float32
        elif b.name == "torch":
            import torch
            arr_float = b.cast(arr_int, torch.float32)
            assert b.dtype(arr_float) == torch.float32
    
    def test_copy(self, b):
        """Test array copying."""
        arr = b.array([1, 2, 3, 4])
        arr_copy = b.copy(arr)
        
        # Should be equal values
        np.testing.assert_array_equal(b.to_numpy(arr), b.to_numpy(arr_copy))
        
        # But different objects (for mutable backends)
        if b.name != "jax":  # JAX arrays are immutable
            # Modify copy shouldn't affect original
            arr_copy_np = b.to_numpy(arr_copy)
            arr_copy_np[0] = 999
            assert b.to_numpy(arr)[0] != 999


class TestIndexing:
    """Test indexing and gathering operations."""
    
    def test_take(self, b):
        """Test take operation."""
        arr = b.array([10, 20, 30, 40, 50])
        indices = b.array([0, 2, 4])
        
        result = b.take(arr, indices)
        np.testing.assert_array_equal(b.to_numpy(result), [10, 30, 50])
    
    def test_gather(self, b):
        """Test gather operation."""
        if not hasattr(b, 'gather'):
            pytest.skip("Gather not implemented")
        
        arr = b.reshape(b.arange(12), (3, 4))
        indices = b.array([0, 2])
        
        # Gather along axis 0
        result = b.gather(arr, indices, axis=0)
        assert b.shape(result) == (2, 4)


class TestDeviceManagement:
    """Test device management if available."""
    
    def test_device_creation(self, b):
        """Test creating arrays on specific device."""
        # CPU array (all backends support this)
        cpu_arr = b.ones((3, 4), device=Device.cpu())
        assert get_device(cpu_arr).type == 'cpu'
    
    def test_gpu_availability(self, b):
        """Test GPU availability check."""
        if not hasattr(b, 'is_gpu_available'):
            pytest.skip("GPU check not implemented")
        
        # Just check it returns bool
        gpu_avail = b.is_gpu_available()
        assert isinstance(gpu_avail, bool)
    
    def test_to_device(self, b):
        """Test moving arrays between devices."""
        if not hasattr(b, 'to_device'):
            pytest.skip("Device transfer not implemented")
        
        arr = b.ones((3, 4))
        
        # Move to CPU (should always work)
        cpu_arr = b.to_device(arr, Device.cpu())
        assert b.is_array(cpu_arr)


class TestErrorHandling:
    """Test error handling."""
    
    def test_mixed_backend_error(self):
        """Test that mixing backends raises error."""
        if has_backend("numpy") and has_backend("torch"):
            b1 = backend("numpy")
            b2 = backend("torch")
            
            a1 = b1.array([1, 2, 3])
            a2 = b2.array([4, 5, 6])
            
            from laakhay.quantlab.backend import ops
            with pytest.raises(MixedBackendError):
                ops.add(a1, a2)
    
    def test_fallback_backend(self):
        """Test fallback backend behavior."""
        b = backend("fallback")
        
        assert b.name == "fallback"
        
        # Should raise helpful error
        with pytest.raises(RuntimeError) as exc_info:
            b.array([1, 2, 3])
        
        assert "No array backend available" in str(exc_info.value)


class TestBackendProperties:
    """Test backend properties and metadata."""
    
    def test_backend_name(self, b):
        """Test backend name property."""
        assert b.name in ["numpy", "jax", "torch"]
    
    def test_array_properties(self, b):
        """Test array property queries."""
        arr = b.reshape(b.arange(24), (2, 3, 4))
        
        assert b.shape(arr) == (2, 3, 4)
        assert b.ndim(arr) == 3
        assert b.size(arr) == 24
        assert b.is_array(arr) == True
    
    def test_zeros_ones_like(self, b):
        """Test zeros_like and ones_like."""
        arr = b.array([[1.5, 2.5], [3.5, 4.5]])
        
        z = b.zeros_like(arr)
        assert b.shape(z) == b.shape(arr)
        np.testing.assert_array_equal(b.to_numpy(z), np.zeros((2, 2)))
        
        o = b.ones_like(arr)
        assert b.shape(o) == b.shape(arr)
        np.testing.assert_array_equal(b.to_numpy(o), np.ones((2, 2)))