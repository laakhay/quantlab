"""Focused tests for backend coverage."""

import pytest

from laakhay.quantlab.backend import get_backend, has_backend


class TestBackendCoverage:
    """Comprehensive backend tests for coverage."""

    @pytest.fixture
    def numpy_backend(self):
        """Get numpy backend."""
        return get_backend("numpy")

    def test_numpy_backend_all_methods(self, numpy_backend):
        """Test all numpy backend methods comprehensively."""
        b = numpy_backend
        
        # Array creation
        arr = b.array([1, 2, 3])
        assert b.shape(arr) == (3,)
        assert b.size(arr) == 3
        assert b.ndim(arr) == 1
        
        # More array creation methods
        zeros = b.zeros((2, 3))
        ones = b.ones((2, 3))
        arange = b.arange(5)
        linspace = b.linspace(0, 1, 5)
        eye = b.eye(3)
        eye2 = b.eye(3, 4)
        full = b.full((2, 2), 5.0)
        
        # Type checking
        assert b.is_array(arr)
        assert not b.is_array([1, 2, 3])
        
        # Array manipulation
        copied = b.copy(arr)
        # Use numpy's float32 dtype
        import numpy as np
        casted = b.cast(arr, np.float32)
        
        # Indexing and slicing
        indexed = b.take(arr, b.array([0, 2]))
        
        # Shape operations
        reshaped = b.reshape(b.arange(6), (2, 3))
        squeezed = b.squeeze(b.array([[[1]]]))
        expanded = b.expand_dims(arr, axis=0)
        transposed = b.transpose(reshaped)
        
        # Concatenation
        concat = b.concat([arr, arr])
        stacked = b.stack([arr, arr])
        
        # Math operations
        summed = b.sum(arr)
        summed_axis = b.sum(reshaped, axis=0)
        summed_keepdims = b.sum(reshaped, axis=0, keepdims=True)
        
        mean_val = b.mean(arr)
        std_val = b.std(arr)
        var_val = b.var(arr)
        prod_val = b.prod(arr)
        
        min_val = b.min(arr)
        max_val = b.max(arr)
        argmin = b.argmin(arr)
        argmax = b.argmax(arr)
        
        # Element-wise operations
        abs_val = b.abs(b.array([-1, 2, -3]))
        sqrt_val = b.sqrt(b.array([4, 9, 16]))
        exp_val = b.exp(arr)
        log_val = b.log(arr)
        
        sin_val = b.sin(arr)
        cos_val = b.cos(arr)
        tan_val = b.tan(arr)
        
        # Binary operations
        added = b.add(arr, arr)
        subbed = b.sub(arr, arr)
        mulled = b.mul(arr, arr)
        divided = b.div(arr, arr)
        powered = b.pow(arr, 2)
        
        # Matrix operations
        mat = b.array([[1, 2], [3, 4]])
        matmul_result = b.matmul(mat, mat)
        
        # Where operation
        where_result = b.where(arr > 1, arr, 0)
        
        # Random operations
        key = b.random_key(42)
        normal = b.random_normal(key, (100,))
        uniform = b.random_uniform(key, (100,))
        split_keys = b.random_split(key, 3)
        
        # Statistical functions
        x = b.array([0.0, 1.0, -1.0])
        norm_cdf = b.norm_cdf(x)
        norm_pdf = b.norm_pdf(x)
        norm_ppf = b.norm_ppf(b.array([0.1, 0.5, 0.9]))
        
        erf_val = b.erf(x)
        erfc_val = b.erfc(x)
        
        gamma_val = b.gamma(b.array([1.0, 2.0, 3.0]))
        lgamma_val = b.lgamma(b.array([1.0, 2.0, 3.0]))
        
        # Conversion
        numpy_arr = b.to_numpy(arr)
        from_numpy = b.from_numpy(numpy_arr)

    def test_jax_backend_specific(self):
        """Test JAX-specific functionality."""
        if has_backend("jax"):
            b = get_backend("jax")
            
            # Test JAX-specific operations
            arr = b.array([1.0, 2.0, 3.0])
            
            # Test device operations if available
            if hasattr(b, "device"):
                device = b.device(arr)
            
            # Test random with proper key
            key = b.random_key(42)
            normal = b.random_normal(key, (10,))
            uniform = b.random_uniform(key, (10,), low=-1.0, high=1.0)
            
            # Test key splitting
            keys = b.random_split(key, 4)
            assert len(keys) == 4
            
    def test_torch_backend_specific(self):
        """Test PyTorch-specific functionality."""
        if has_backend("torch"):
            b = get_backend("torch")
            
            # Test torch-specific operations
            arr = b.array([1.0, 2.0, 3.0])
            
            # Test device operations
            if hasattr(b, "device"):
                device = b.device(arr)
                
            # Test dtype
            dtype = b.dtype(arr)
            
            # Test random operations
            key = b.random_key(42)
            normal = b.random_normal(key, (10,))
            uniform = b.random_uniform(key, (10,))
            
            # Test mathematical operations that might have different implementations
            x = b.array([0.1, 0.5, 0.9])
            ppf = b.norm_ppf(x)
            
            # Test gamma function
            gamma_val = b.gamma(b.array([1.0, 2.0]))
            lgamma_val = b.lgamma(b.array([1.0, 2.0]))

    def test_backend_edge_cases(self, numpy_backend):
        """Test edge cases and error conditions."""
        b = numpy_backend
        
        # Empty arrays
        empty = b.zeros((0,))
        assert b.size(empty) == 0
        
        # Single element
        single = b.array(5)
        assert b.ndim(single) == 0
        
        # High dimensional
        high_dim = b.ones((2, 3, 4, 5))
        assert b.ndim(high_dim) == 4
        
        # Test squeeze with no dimensions to squeeze
        no_squeeze = b.squeeze(b.ones((2, 3)))
        assert b.shape(no_squeeze) == (2, 3)
        
        # Test with negative indices
        arr = b.arange(10)
        last = b.take(arr, b.array([-1]))
        
        # Test sum with single axis
        mat3d = b.ones((2, 3, 4))
        sum_axis = b.sum(mat3d, axis=0)
        
        # Test argmax/argmin with axis
        mat = b.array([[1, 3], [2, 4]])
        argmax_axis = b.argmax(mat, axis=1)
        argmin_axis = b.argmin(mat, axis=0)