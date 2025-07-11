"""Tests for torch backend coverage."""

import pytest

from laakhay.quantlab.backend import get_backend, has_backend


class TestTorchCoverage:
    """Test torch backend specific functionality."""
    
    @pytest.fixture
    def torch_backend(self):
        """Get torch backend if available."""
        if not has_backend("torch"):
            pytest.skip("PyTorch not available")
        return get_backend("torch")
        
    def test_torch_array_creation(self, torch_backend):
        """Test torch array creation methods."""
        b = torch_backend
        
        # Test with dtype
        import torch
        arr_float = b.array([1, 2, 3], dtype=torch.float32)
        assert str(arr_float.dtype) == "torch.float32"
        
        # Test with device (if CUDA available)
        if torch.cuda.is_available():
            arr_gpu = b.array([1, 2, 3], device="cuda:0")
            assert arr_gpu.device.type == "cuda"
            
        # Test zeros with dtype
        zeros = b.zeros((2, 3), dtype=torch.int64)
        assert str(zeros.dtype) == "torch.int64"
        
        # Test ones with dtype
        ones = b.ones((2, 3), dtype=torch.float64)
        assert str(ones.dtype) == "torch.float64"
        
        # Test arange with dtype
        arange = b.arange(5, dtype=torch.int32)
        assert str(arange.dtype) == "torch.int32"
        
        # Test eye
        eye = b.eye(3, 4)
        assert b.shape(eye) == (3, 4)
        
    def test_torch_device_operations(self, torch_backend):
        """Test torch device operations."""
        b = torch_backend
        
        # Test device property
        arr = b.array([1, 2, 3])
        device = b.device(arr)
        assert device.type == "cpu"
        
        # Test to_device
        arr_cpu = b.to_device(arr, "cpu")
        assert arr_cpu.device.type == "cpu"
        
    def test_torch_special_functions(self, torch_backend):
        """Test torch special mathematical functions."""
        b = torch_backend
        
        # Test tan
        x = b.array([0.0, 1.57, 3.14])
        tan_x = b.tan(x)
        assert b.shape(tan_x) == (3,)
        
        # Test statistical functions with fallback
        x = b.array([-1.0, 0.0, 1.0])
        
        # Test erf and erfc
        erf_x = b.erf(x)
        erfc_x = b.erfc(x)
        assert b.shape(erf_x) == (3,)
        assert b.shape(erfc_x) == (3,)
        
        # Test norm functions
        norm_cdf = b.norm_cdf(x)
        norm_pdf = b.norm_pdf(x)
        norm_ppf = b.norm_ppf(b.array([0.25, 0.5, 0.75]))
        
        # Test gamma functions
        gamma_x = b.gamma(b.array([1.0, 2.0, 3.0]))
        lgamma_x = b.lgamma(b.array([1.0, 2.0, 3.0]))
        
    def test_torch_random_with_dtype(self, torch_backend):
        """Test torch random operations with dtype."""
        b = torch_backend
        
        key = b.random_key(42)
        
        # Test with dtype
        import torch
        normal = b.random_normal(key, (10,), dtype=torch.float64)
        assert str(normal.dtype) == "torch.float64"
        
        uniform = b.random_uniform(key, (10,), dtype=torch.float32)
        assert str(uniform.dtype) == "torch.float32"
        
        # Test random split
        keys = b.random_split(key, 4)
        assert len(keys) == 4
        
    def test_torch_gather_operations(self, torch_backend):
        """Test torch gather operations."""
        b = torch_backend
        
        # Test 1D gather (index_select)
        arr = b.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        indices = b.array([0, 2])
        gathered = b.gather(arr, indices, axis=0)
        assert b.shape(gathered) == (2, 3)
        
        # Test multi-dimensional gather
        # PyTorch gather works differently, just test it works
        indices_2d = b.array([[0, 2], [1, 0], [2, 1]])
        result = b.gather(arr, indices_2d, axis=1)
        assert b.shape(result) == (3, 2)
            
    def test_torch_norm_operations(self, torch_backend):
        """Test torch norm operations."""
        b = torch_backend
        
        # Test with integer array (should convert to float)
        arr_int = b.array([[1, 2], [3, 4]])
        norm_int = b.norm(arr_int)
        assert norm_int.dtype.is_floating_point
        
        # Test with ord parameter
        arr = b.array([[1.0, 2.0], [3.0, 4.0]])
        norm_1 = b.norm(arr, ord=1)
        norm_inf = b.norm(arr, ord=float('inf'))
        
        # Test with axis
        norm_axis = b.norm(arr, axis=0)
        assert b.shape(norm_axis) == (2,)
        
    def test_torch_linalg_operations(self, torch_backend):
        """Test torch linear algebra operations."""
        b = torch_backend
        
        # Test solve
        A = b.array([[3.0, 1.0], [1.0, 2.0]])
        b_vec = b.array([9.0, 8.0])
        x = b.solve(A, b_vec)
        
        # Test inv
        A_inv = b.inv(A)
        identity = b.matmul(A, A_inv)
        
        # Test det
        det_A = b.det(A)
        assert b.shape(det_A) == ()  # scalar