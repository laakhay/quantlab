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
        assert b.shape(zeros) == (2, 3)
        assert b.sum(zeros) == 0.0

        ones = b.ones((2, 3))
        assert b.shape(ones) == (2, 3)
        assert b.sum(ones) == 6.0

        arange = b.arange(5)
        assert b.shape(arange) == (5,)
        assert b.sum(arange) == 10.0

        linspace = b.linspace(0, 1, 5)
        assert b.shape(linspace) == (5,)
        assert b.min(linspace) == 0.0
        assert b.max(linspace) == 1.0

        eye = b.eye(3)
        assert b.shape(eye) == (3, 3)
        assert b.sum(eye) == 3.0

        eye2 = b.eye(3, 4)
        assert b.shape(eye2) == (3, 4)
        assert b.sum(eye2) == 3.0

        full = b.full((2, 2), 5.0)
        assert b.shape(full) == (2, 2)
        assert b.sum(full) == 20.0

        # Type checking
        assert b.is_array(arr)
        assert not b.is_array([1, 2, 3])

        # Array manipulation
        copied = b.copy(arr)
        assert b.shape(copied) == b.shape(arr)
        assert b.sum(copied) == b.sum(arr)

        # Use numpy's float32 dtype
        import numpy as np

        casted = b.cast(arr, np.float32)
        assert b.shape(casted) == b.shape(arr)

        # Indexing and slicing
        indexed = b.take(arr, b.array([0, 2]))
        assert b.shape(indexed) == (2,)
        assert b.sum(indexed) == 4.0  # 1 + 3

        # Shape operations
        reshaped = b.reshape(b.arange(6), (2, 3))
        assert b.shape(reshaped) == (2, 3)

        squeezed = b.squeeze(b.array([[[1]]]))
        assert b.ndim(squeezed) == 0

        expanded = b.expand_dims(arr, axis=0)
        assert b.shape(expanded) == (1, 3)

        transposed = b.transpose(reshaped)
        assert b.shape(transposed) == (3, 2)

        # Concatenation
        concat = b.concat([arr, arr])
        assert b.shape(concat) == (6,)

        stacked = b.stack([arr, arr])
        assert b.shape(stacked) == (2, 3)

        # Math operations
        summed = b.sum(arr)
        assert summed == 6.0

        summed_axis = b.sum(reshaped, axis=0)
        assert b.shape(summed_axis) == (3,)

        summed_keepdims = b.sum(reshaped, axis=0, keepdims=True)
        assert b.shape(summed_keepdims) == (1, 3)

        mean_val = b.mean(arr)
        assert mean_val == 2.0

        std_val = b.std(arr)
        assert std_val > 0

        var_val = b.var(arr)
        assert var_val > 0

        prod_val = b.prod(arr)
        assert prod_val == 6.0

        min_val = b.min(arr)
        assert min_val == 1.0

        max_val = b.max(arr)
        assert max_val == 3.0

        argmin = b.argmin(arr)
        assert argmin == 0

        argmax = b.argmax(arr)
        assert argmax == 2

        # Element-wise operations
        abs_val = b.abs(b.array([-1, 2, -3]))
        assert b.sum(abs_val) == 6.0

        sqrt_val = b.sqrt(b.array([4, 9, 16]))
        assert b.sum(sqrt_val) == 9.0  # 2 + 3 + 4

        exp_val = b.exp(arr)
        assert b.shape(exp_val) == b.shape(arr)

        log_val = b.log(arr)
        assert b.shape(log_val) == b.shape(arr)

        sin_val = b.sin(arr)
        assert b.shape(sin_val) == b.shape(arr)

        cos_val = b.cos(arr)
        assert b.shape(cos_val) == b.shape(arr)

        tan_val = b.tan(arr)
        assert b.shape(tan_val) == b.shape(arr)

        # Binary operations
        added = b.add(arr, arr)
        assert b.sum(added) == 12.0

        subbed = b.sub(arr, arr)
        assert b.sum(subbed) == 0.0

        mulled = b.mul(arr, arr)
        assert b.sum(mulled) == 14.0  # 1 + 4 + 9

        divided = b.div(arr, arr)
        assert b.sum(divided) == 3.0  # 1 + 1 + 1

        powered = b.pow(arr, 2)
        assert b.sum(powered) == 14.0  # 1 + 4 + 9

        # Matrix operations
        mat = b.array([[1, 2], [3, 4]])
        matmul_result = b.matmul(mat, mat)
        assert b.shape(matmul_result) == (2, 2)

        # Where operation
        where_result = b.where(arr > 1, arr, 0)
        assert b.sum(where_result) == 5.0  # 0 + 2 + 3

        # Random operations
        key = b.random_key(42)
        normal = b.random_normal(key, (100,))
        assert b.shape(normal) == (100,)

        uniform = b.random_uniform(key, (100,))
        assert b.shape(uniform) == (100,)
        assert b.min(uniform) >= 0.0
        assert b.max(uniform) <= 1.0

        split_keys = b.random_split(key, 3)
        assert len(split_keys) == 3

        # Statistical functions
        x = b.array([0.0, 1.0, -1.0])
        norm_cdf = b.norm_cdf(x)
        assert b.shape(norm_cdf) == b.shape(x)

        norm_pdf = b.norm_pdf(x)
        assert b.shape(norm_pdf) == b.shape(x)

        norm_ppf = b.norm_ppf(b.array([0.1, 0.5, 0.9]))
        assert b.shape(norm_ppf) == (3,)

        erf_val = b.erf(x)
        assert b.shape(erf_val) == b.shape(x)

        erfc_val = b.erfc(x)
        assert b.shape(erfc_val) == b.shape(x)

        gamma_val = b.gamma(b.array([1.0, 2.0, 3.0]))
        assert b.shape(gamma_val) == (3,)

        lgamma_val = b.lgamma(b.array([1.0, 2.0, 3.0]))
        assert b.shape(lgamma_val) == (3,)

        # Conversion
        numpy_arr = b.to_numpy(arr)
        assert numpy_arr.shape == (3,)

        from_numpy = b.from_numpy(numpy_arr)
        assert b.shape(from_numpy) == b.shape(arr)

    def test_jax_backend_specific(self):
        """Test JAX-specific functionality."""
        if has_backend("jax"):
            b = get_backend("jax")

            # Test JAX-specific operations
            arr = b.array([1.0, 2.0, 3.0])

            # Test device operations if available
            if hasattr(b, "device"):
                device = b.device(arr)
                assert device is not None

            # Test random with proper key
            key = b.random_key(42)
            normal = b.random_normal(key, (10,))
            assert b.shape(normal) == (10,)

            uniform = b.random_uniform(key, (10,), low=-1.0, high=1.0)
            assert b.shape(uniform) == (10,)
            assert b.min(uniform) >= -1.0
            assert b.max(uniform) <= 1.0

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
                assert device is not None

            # Test dtype
            dtype = b.dtype(arr)
            assert dtype is not None

            # Test random operations
            key = b.random_key(42)
            normal = b.random_normal(key, (10,))
            assert b.shape(normal) == (10,)

            uniform = b.random_uniform(key, (10,))
            assert b.shape(uniform) == (10,)

            # Test mathematical operations that might have different implementations
            x = b.array([0.1, 0.5, 0.9])
            ppf = b.norm_ppf(x)
            assert b.shape(ppf) == b.shape(x)

            # Test gamma function
            gamma_val = b.gamma(b.array([1.0, 2.0]))
            assert b.shape(gamma_val) == (2,)

            lgamma_val = b.lgamma(b.array([1.0, 2.0]))
            assert b.shape(lgamma_val) == (2,)

    def test_backend_edge_cases(self, numpy_backend):
        """Test edge cases and error conditions."""
        b = numpy_backend

        # Empty arrays
        empty = b.zeros((0,))
        assert b.size(empty) == 0
        assert b.shape(empty) == (0,)

        # Single element
        single = b.array(5)
        assert b.ndim(single) == 0
        assert b.size(single) == 1

        # High dimensional
        high_dim = b.ones((2, 3, 4, 5))
        assert b.ndim(high_dim) == 4
        assert b.size(high_dim) == 120

        # Test squeeze with no dimensions to squeeze
        no_squeeze = b.squeeze(b.ones((2, 3)))
        assert b.shape(no_squeeze) == (2, 3)

        # Test with negative indices
        arr = b.arange(10)
        last = b.take(arr, b.array([-1]))
        assert b.sum(last) == 9.0

        # Test sum with single axis
        mat3d = b.ones((2, 3, 4))
        sum_axis = b.sum(mat3d, axis=0)
        assert b.shape(sum_axis) == (3, 4)

        # Test argmax/argmin with axis
        mat = b.array([[1, 3], [2, 4]])
        argmax_axis = b.argmax(mat, axis=1)
        assert b.shape(argmax_axis) == (2,)

        argmin_axis = b.argmin(mat, axis=0)
        assert b.shape(argmin_axis) == (2,)
