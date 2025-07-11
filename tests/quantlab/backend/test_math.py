"""Mathematical operations tests."""

import pytest
import numpy as np

from laakhay.quantlab.backend import backend, has_backend


@pytest.fixture(params=["numpy", "jax", "torch"])
def b(request):
    """Parametrized backend fixture."""
    if not has_backend(request.param):
        pytest.skip(f"{request.param} not available")
    return backend(request.param)


class TestArithmetic:
    """Test basic arithmetic operations."""
    
    def test_add(self, b):
        """Test addition."""
        a = b.array([1, 2, 3])
        b_arr = b.array([4, 5, 6])
        
        result = b.add(a, b_arr)
        np.testing.assert_array_equal(b.to_numpy(result), [5, 7, 9])
        
        # Broadcast
        scalar = 10
        result2 = b.add(a, scalar)
        np.testing.assert_array_equal(b.to_numpy(result2), [11, 12, 13])
    
    def test_sub(self, b):
        """Test subtraction."""
        a = b.array([10, 20, 30])
        b_arr = b.array([1, 2, 3])
        
        result = b.sub(a, b_arr)
        np.testing.assert_array_equal(b.to_numpy(result), [9, 18, 27])
    
    def test_mul(self, b):
        """Test multiplication."""
        a = b.array([2, 3, 4])
        b_arr = b.array([5, 6, 7])
        
        result = b.mul(a, b_arr)
        np.testing.assert_array_equal(b.to_numpy(result), [10, 18, 28])
    
    def test_div(self, b):
        """Test division."""
        a = b.array([10.0, 20.0, 30.0])
        b_arr = b.array([2.0, 4.0, 5.0])
        
        result = b.div(a, b_arr)
        np.testing.assert_allclose(b.to_numpy(result), [5.0, 5.0, 6.0])
    
    def test_pow(self, b):
        """Test power operation."""
        a = b.array([2, 3, 4])
        b_arr = b.array([3, 2, 2])
        
        result = b.pow(a, b_arr)
        np.testing.assert_array_equal(b.to_numpy(result), [8, 9, 16])
    
    def test_arithmetic_broadcast(self, b):
        """Test broadcasting in arithmetic."""
        # 2D x 1D
        a = b.reshape(b.arange(6), (2, 3))
        b_arr = b.array([1, 2, 3])
        
        result = b.add(a, b_arr)
        expected = np.array([[1, 3, 5], [4, 6, 8]])
        np.testing.assert_array_equal(b.to_numpy(result), expected)


class TestElementwise:
    """Test elementwise functions."""
    
    def test_abs(self, b):
        """Test absolute value."""
        a = b.array([-1, 2, -3, 0, -4.5])
        result = b.abs(a)
        np.testing.assert_allclose(b.to_numpy(result), [1, 2, 3, 0, 4.5])
    
    def test_sign(self, b):
        """Test sign function."""
        a = b.array([-5, 0, 3, -0.1, 0.2])
        result = b.sign(a)
        np.testing.assert_array_equal(b.to_numpy(result), [-1, 0, 1, -1, 1])
    
    def test_exp(self, b):
        """Test exponential."""
        a = b.array([0, 1, 2, -1])
        result = b.exp(a)
        expected = np.exp([0, 1, 2, -1])
        np.testing.assert_allclose(b.to_numpy(result), expected, rtol=1e-6)
    
    def test_log(self, b):
        """Test natural logarithm."""
        a = b.array([1, np.e, np.e**2, 10])
        result = b.log(a)
        expected = np.log([1, np.e, np.e**2, 10])
        np.testing.assert_allclose(b.to_numpy(result), expected, rtol=1e-6)
    
    def test_log10(self, b):
        """Test base-10 logarithm."""
        if not hasattr(b, 'log10'):
            pytest.skip("log10 not implemented")
        
        a = b.array([1, 10, 100, 1000])
        result = b.log10(a)
        np.testing.assert_allclose(b.to_numpy(result), [0, 1, 2, 3], rtol=1e-6)
    
    def test_sqrt(self, b):
        """Test square root."""
        a = b.array([0, 1, 4, 9, 16, 25])
        result = b.sqrt(a)
        np.testing.assert_allclose(b.to_numpy(result), [0, 1, 2, 3, 4, 5], rtol=1e-6)


class TestTrigonometric:
    """Test trigonometric functions."""
    
    def test_sin(self, b):
        """Test sine."""
        # Use values that are exactly representable to avoid π precision issues
        a = b.array([0, 0.5, 1.0, -0.5, -1.0])
        result = b.sin(a)
        expected = np.sin([0, 0.5, 1.0, -0.5, -1.0])
        np.testing.assert_allclose(b.to_numpy(result), expected, rtol=1e-6)
    
    def test_cos(self, b):
        """Test cosine."""
        # Use values that are exactly representable to avoid π precision issues
        a = b.array([0, 0.5, 1.0, -0.5, -1.0])
        result = b.cos(a)
        expected = np.cos([0, 0.5, 1.0, -0.5, -1.0])
        np.testing.assert_allclose(b.to_numpy(result), expected, rtol=1e-6)
    
    def test_tan(self, b):
        """Test tangent."""
        a = b.array([0, np.pi/4, -np.pi/4])
        result = b.tan(a)
        expected = np.tan([0, np.pi/4, -np.pi/4])
        np.testing.assert_allclose(b.to_numpy(result), expected, rtol=1e-6)
    
    def test_hyperbolic(self, b):
        """Test hyperbolic functions."""
        a = b.array([-1, 0, 1, 2])
        
        # tanh
        tanh_result = b.tanh(a)
        np.testing.assert_allclose(b.to_numpy(tanh_result), np.tanh([-1, 0, 1, 2]), rtol=1e-6)
        
        # sinh
        if hasattr(b, 'sinh'):
            sinh_result = b.sinh(a)
            np.testing.assert_allclose(b.to_numpy(sinh_result), np.sinh([-1, 0, 1, 2]), rtol=1e-6)
        
        # cosh
        if hasattr(b, 'cosh'):
            cosh_result = b.cosh(a)
            np.testing.assert_allclose(b.to_numpy(cosh_result), np.cosh([-1, 0, 1, 2]), rtol=1e-6)
    
    def test_trig_identities(self, b):
        """Test trigonometric identities."""
        x = b.linspace(-np.pi, np.pi, 50)
        
        # sin^2 + cos^2 = 1
        sin_sq = b.mul(b.sin(x), b.sin(x))
        cos_sq = b.mul(b.cos(x), b.cos(x))
        sum_sq = b.add(sin_sq, cos_sq)
        
        np.testing.assert_allclose(b.to_numpy(sum_sq), np.ones(50), rtol=1e-6)


class TestReductions:
    """Test reduction operations."""
    
    def test_sum(self, b):
        """Test sum reduction."""
        arr = b.reshape(b.arange(12), (3, 4))
        
        # Total sum
        total = b.sum(arr)
        assert float(b.to_numpy(total)) == 66  # sum(0:11)
        
        # Axis sum
        sum0 = b.sum(arr, axis=0)
        np.testing.assert_array_equal(b.to_numpy(sum0), [12, 15, 18, 21])
        
        sum1 = b.sum(arr, axis=1)
        np.testing.assert_array_equal(b.to_numpy(sum1), [6, 22, 38])
        
        # Keepdims
        sum_keep = b.sum(arr, axis=1, keepdims=True)
        assert b.shape(sum_keep) == (3, 1)
    
    def test_mean(self, b):
        """Test mean reduction."""
        arr = b.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        
        # Total mean
        mean_all = b.mean(arr)
        np.testing.assert_allclose(float(b.to_numpy(mean_all)), 3.5)
        
        # Axis mean
        mean0 = b.mean(arr, axis=0)
        np.testing.assert_allclose(b.to_numpy(mean0), [2.5, 3.5, 4.5])
        
        mean1 = b.mean(arr, axis=1)
        np.testing.assert_allclose(b.to_numpy(mean1), [2.0, 5.0])
    
    def test_std_var(self, b):
        """Test standard deviation and variance."""
        arr = b.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        # Variance
        var = b.var(arr)
        np.testing.assert_allclose(float(b.to_numpy(var)), 2.0, rtol=1e-6)
        
        # Std dev
        std = b.std(arr)
        np.testing.assert_allclose(float(b.to_numpy(std)), np.sqrt(2.0), rtol=1e-6)
    
    def test_min_max(self, b):
        """Test min/max reductions."""
        arr = b.array([[3, 1, 4], [1, 5, 9], [2, 6, 5]])
        
        # Global
        min_all = b.min(arr)
        max_all = b.max(arr)
        assert float(b.to_numpy(min_all)) == 1
        assert float(b.to_numpy(max_all)) == 9
        
        # Axis
        min0 = b.min(arr, axis=0)
        max1 = b.max(arr, axis=1)
        np.testing.assert_array_equal(b.to_numpy(min0), [1, 1, 4])
        np.testing.assert_array_equal(b.to_numpy(max1), [4, 9, 6])
    
    def test_argmin_argmax(self, b):
        """Test argmin/argmax."""
        arr = b.array([3, 1, 4, 1, 5, 9, 2, 6])
        
        # Global
        amin = b.argmin(arr)
        amax = b.argmax(arr)
        assert int(b.to_numpy(amin)) == 1  # First occurrence of 1
        assert int(b.to_numpy(amax)) == 5  # Position of 9
        
        # 2D with axis
        arr2d = b.reshape(arr[:6], (2, 3))
        amin0 = b.argmin(arr2d, axis=0)
        np.testing.assert_array_equal(b.to_numpy(amin0), [1, 0, 0])
    
    def test_prod(self, b):
        """Test product reduction."""
        arr = b.array([1, 2, 3, 4, 5])
        prod = b.prod(arr)
        assert float(b.to_numpy(prod)) == 120  # 5!
        
        # With axis
        mat = b.array([[1, 2, 3], [4, 5, 6]])
        prod0 = b.prod(mat, axis=0)
        prod1 = b.prod(mat, axis=1)
        np.testing.assert_array_equal(b.to_numpy(prod0), [4, 10, 18])
        np.testing.assert_array_equal(b.to_numpy(prod1), [6, 120])


class TestComparisons:
    """Test comparison operations."""
    
    def test_equality(self, b):
        """Test equality comparisons."""
        a = b.array([1, 2, 3, 4, 5])
        b_arr = b.array([1, 0, 3, 4, 6])
        
        eq = b.eq(a, b_arr)
        ne = b.ne(a, b_arr)
        
        np.testing.assert_array_equal(b.to_numpy(eq), [True, False, True, True, False])
        np.testing.assert_array_equal(b.to_numpy(ne), [False, True, False, False, True])
    
    def test_ordering(self, b):
        """Test ordering comparisons."""
        a = b.array([1, 2, 3, 4, 5])
        b_arr = b.array([3, 3, 3, 3, 3])
        
        lt = b.lt(a, b_arr)
        le = b.le(a, b_arr)
        gt = b.gt(a, b_arr)
        ge = b.ge(a, b_arr)
        
        np.testing.assert_array_equal(b.to_numpy(lt), [True, True, False, False, False])
        np.testing.assert_array_equal(b.to_numpy(le), [True, True, True, False, False])
        np.testing.assert_array_equal(b.to_numpy(gt), [False, False, False, True, True])
        np.testing.assert_array_equal(b.to_numpy(ge), [False, False, True, True, True])
    
    def test_special_values(self, b):
        """Test special value checks."""
        arr = b.array([float('nan'), float('inf'), float('-inf'), 0.0, 1.0])
        
        isnan = b.isnan(arr)
        isinf = b.isinf(arr)
        isfinite = b.isfinite(arr)
        
        np.testing.assert_array_equal(b.to_numpy(isnan), [True, False, False, False, False])
        np.testing.assert_array_equal(b.to_numpy(isinf), [False, True, True, False, False])
        np.testing.assert_array_equal(b.to_numpy(isfinite), [False, False, False, True, True])


class TestSelection:
    """Test selection operations."""
    
    def test_where(self, b):
        """Test conditional selection."""
        cond = b.array([True, False, True, False, True])
        x = b.array([1, 2, 3, 4, 5])
        y = b.array([10, 20, 30, 40, 50])
        
        result = b.where(cond, x, y)
        np.testing.assert_array_equal(b.to_numpy(result), [1, 20, 3, 40, 5])
    
    def test_clip(self, b):
        """Test clipping."""
        arr = b.array([-2, -1, 0, 1, 2, 3, 4, 5])
        
        # Both bounds
        clipped = b.clip(arr, min=0, max=3)
        np.testing.assert_array_equal(b.to_numpy(clipped), [0, 0, 0, 1, 2, 3, 3, 3])
        
        # Only min
        clipped_min = b.clip(arr, min=1)
        np.testing.assert_array_equal(b.to_numpy(clipped_min), [1, 1, 1, 1, 2, 3, 4, 5])
        
        # Only max
        clipped_max = b.clip(arr, max=2)
        np.testing.assert_array_equal(b.to_numpy(clipped_max), [-2, -1, 0, 1, 2, 2, 2, 2])
    
    def test_maximum_minimum(self, b):
        """Test elementwise max/min."""
        a = b.array([1, 5, 3, 8, 2])
        b_arr = b.array([3, 3, 3, 3, 3])
        
        max_result = b.maximum(a, b_arr)
        min_result = b.minimum(a, b_arr)
        
        np.testing.assert_array_equal(b.to_numpy(max_result), [3, 5, 3, 8, 3])
        np.testing.assert_array_equal(b.to_numpy(min_result), [1, 3, 3, 3, 2])


class TestLinearAlgebra:
    """Test linear algebra operations."""
    
    def test_matmul(self, b):
        """Test matrix multiplication."""
        a = b.array([[1, 2], [3, 4]])
        b_arr = b.array([[5, 6], [7, 8]])
        
        result = b.matmul(a, b_arr)
        expected = np.array([[19, 22], [43, 50]])
        np.testing.assert_array_equal(b.to_numpy(result), expected)
        
        # Vector-matrix
        v = b.array([1, 2])
        vm_result = b.matmul(v, a)
        np.testing.assert_array_equal(b.to_numpy(vm_result), [7, 10])
    
    def test_dot(self, b):
        """Test dot product."""
        if not hasattr(b, 'dot'):
            # Use matmul as fallback
            a = b.array([1, 2, 3])
            b_arr = b.array([4, 5, 6])
            result = b.sum(b.mul(a, b_arr))
            assert float(b.to_numpy(result)) == 32
        else:
            a = b.array([1, 2, 3])
            b_arr = b.array([4, 5, 6])
            result = b.dot(a, b_arr)
            assert float(b.to_numpy(result)) == 32
    
    def test_norm(self, b):
        """Test matrix/vector norms."""
        if not hasattr(b, 'norm'):
            pytest.skip("norm not implemented")
        
        # Vector 2-norm
        v = b.array([3, 4])
        norm2 = b.norm(v)
        np.testing.assert_allclose(float(b.to_numpy(norm2)), 5.0, rtol=1e-6)
        
        # Matrix Frobenius norm
        m = b.array([[1, 2], [3, 4]])
        norm_f = b.norm(m)
        expected = np.sqrt(1 + 4 + 9 + 16)
        np.testing.assert_allclose(float(b.to_numpy(norm_f)), expected, rtol=1e-6)
    
    def test_solve(self, b):
        """Test linear system solve."""
        if not hasattr(b, 'solve'):
            pytest.skip("solve not implemented")
        
        # Simple system: Ax = b
        A = b.array([[2.0, 1.0], [1.0, 3.0]])
        b_vec = b.array([5.0, 4.0])
        
        x = b.solve(A, b_vec)
        
        # Verify Ax = b
        result = b.matmul(A, x)
        np.testing.assert_allclose(b.to_numpy(result), b.to_numpy(b_vec), rtol=1e-6)
    
    def test_inv(self, b):
        """Test matrix inverse."""
        if not hasattr(b, 'inv'):
            pytest.skip("inv not implemented")
        
        A = b.array([[4.0, 7.0], [2.0, 6.0]])
        A_inv = b.inv(A)
        
        # Verify A @ A_inv = I
        I = b.matmul(A, A_inv)
        np.testing.assert_allclose(b.to_numpy(I), np.eye(2), rtol=1e-6)
    
    def test_det(self, b):
        """Test determinant."""
        if not hasattr(b, 'det'):
            pytest.skip("det not implemented")
        
        # 2x2 determinant
        A = b.array([[3.0, 8.0], [4.0, 6.0]])
        det = b.det(A)
        expected = 3*6 - 8*4  # -14
        np.testing.assert_allclose(float(b.to_numpy(det)), expected, rtol=1e-6)


class TestSpecialFunctions:
    """Test special mathematical functions."""
    
    def test_erf(self, b):
        """Test error function."""
        if not hasattr(b, 'erf'):
            pytest.skip("erf not implemented")
        
        x = b.array([-2, -1, 0, 1, 2])
        result = b.erf(x)
        
        # Check known values
        result_np = b.to_numpy(result)
        np.testing.assert_allclose(result_np[2], 0.0, atol=1e-6)  # erf(0) = 0
        np.testing.assert_allclose(result_np[3], -result_np[1], atol=1e-6)  # erf(-x) = -erf(x)
    
    def test_norm_cdf_pdf(self, b):
        """Test normal distribution functions."""
        if not hasattr(b, 'norm_cdf'):
            pytest.skip("norm_cdf not implemented")
        
        x = b.array([-2, -1, 0, 1, 2])
        
        # CDF
        cdf = b.norm_cdf(x)
        cdf_np = b.to_numpy(cdf)
        np.testing.assert_allclose(cdf_np[2], 0.5, atol=1e-6)  # Phi(0) = 0.5
        
        # PDF
        if hasattr(b, 'norm_pdf'):
            pdf = b.norm_pdf(x)
            pdf_np = b.to_numpy(pdf)
            np.testing.assert_allclose(pdf_np[2], 1/np.sqrt(2*np.pi), atol=1e-6)  # phi(0)


class TestNumericalStability:
    """Test numerical stability of operations."""
    
    def test_small_number_accumulation(self, b):
        """Test accumulation of small numbers."""
        n = 10000
        small = 1e-10
        arr = b.full((n,), small)
        total = b.sum(arr)
        
        expected = n * small
        # Use more reasonable tolerance for floating point accumulation
        # JAX may have different accumulation behavior
        np.testing.assert_allclose(float(b.to_numpy(total)), expected, rtol=5e-6)
    
    def test_log_exp_stability(self, b):
        """Test log/exp numerical stability."""
        # Large values
        x_large = b.array([100, 200, 300])
        
        # log(exp(x)) should equal x
        log_exp = b.log(b.exp(b.div(x_large, 100)))  # Scale down to avoid overflow
        np.testing.assert_allclose(b.to_numpy(log_exp), b.to_numpy(b.div(x_large, 100)), rtol=1e-6)
        
        # Small values
        x_small = b.array([1e-10, 1e-20, 1e-30])
        
        # exp(log(x)) should equal x
        exp_log = b.exp(b.log(b.add(1, x_small)))  # log(1+x) for stability
        np.testing.assert_allclose(b.to_numpy(exp_log), b.to_numpy(b.add(1, x_small)), rtol=1e-6)