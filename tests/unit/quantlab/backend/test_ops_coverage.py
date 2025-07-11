"""Tests for ops module coverage."""

import pytest

from laakhay.quantlab.backend import get_backend

from laakhay.quantlab.backend.ops import (  # isort: skip
    Ops,
    add,
    clip,
    div,
    exp,
    log,
    mul,
    reshape,
    sqrt,
    sub,
    where,
)
from laakhay.quantlab.backend.ops import max as ops_max  # isort: skip
from laakhay.quantlab.backend.ops import min as ops_min  # isort: skip


class TestOpsCoverage:
    """Test ops module functionality."""

    @pytest.fixture
    def backend(self):
        """Get numpy backend."""
        return get_backend("numpy")

    def test_ops_binary(self, backend):
        """Test binary operations through Ops class."""
        # Create test arrays
        a = backend.array([1.0, 2.0, 3.0])
        b = backend.array([4.0, 5.0, 6.0])

        # Test binary ops using module functions
        result_add = add(a, b)
        assert backend.shape(result_add) == (3,)
        assert float(backend.to_numpy(result_add[0])) == 5.0

        result_sub = sub(a, b)
        assert backend.shape(result_sub) == (3,)

        result_mul = mul(a, b)
        assert backend.shape(result_mul) == (3,)

        result_div = div(a, b)
        assert backend.shape(result_div) == (3,)

    def test_ops_unary(self, backend):
        """Test unary operations."""
        a = backend.array([1.0, 4.0, 9.0])

        # Test unary ops
        result_exp = exp(a)
        assert backend.shape(result_exp) == (3,)

        result_log = log(a)
        assert backend.shape(result_log) == (3,)

        result_sqrt = sqrt(a)
        assert backend.shape(result_sqrt) == (3,)
        assert float(backend.to_numpy(result_sqrt[1])) == 2.0

    def test_ops_class_methods(self, backend):
        """Test Ops class static methods."""
        a = backend.array([1.0, -2.0, 3.0])
        b = backend.array([4.0, 5.0, 6.0])

        # Test binary op through class
        result = Ops.binary_op("add", a, b)
        assert backend.shape(result) == (3,)

        # Test unary op through class
        result = Ops.unary_op("abs", a)
        assert backend.shape(result) == (3,)
        assert float(backend.to_numpy(result[1])) == 2.0

        # Test sum with kwargs
        result = Ops.sum(backend.array([[1, 2], [3, 4]]), axis=0)
        assert backend.shape(result) == (2,)

    def test_ops_shape(self, backend):
        """Test shape operations."""
        a = backend.array([[1, 2, 3], [4, 5, 6]])

        # Test reshape
        reshaped = reshape(a, (3, 2))
        assert backend.shape(reshaped) == (3, 2)

        # Test reshape through Ops
        reshaped2 = Ops.reshape(a, (6,))
        assert backend.shape(reshaped2) == (6,)

    def test_ops_comparison(self, backend):
        """Test comparison operations."""
        a = backend.array([1, 2, 3])
        b = backend.array([2, 2, 2])

        # Test through Ops class
        eq_result = Ops.eq(a, b)
        lt_result = Ops.lt(a, b)
        gt_result = Ops.gt(a, b)

        assert backend.shape(eq_result) == (3,)
        assert backend.shape(lt_result) == (3,)
        assert backend.shape(gt_result) == (3,)

    def test_ops_where(self, backend):
        """Test where operation."""
        condition = backend.array([True, False, True])
        x = backend.array([1, 2, 3])
        y = backend.array([4, 5, 6])

        result = where(condition, x, y)
        assert backend.shape(result) == (3,)

        # Also test through Ops
        result2 = Ops.where(condition, x, y)
        assert backend.shape(result2) == (3,)

    def test_ops_clip(self, backend):
        """Test clip operation."""
        a = backend.array([-1, 0, 5, 10])

        # Test clip
        clipped = clip(a, 0, 5)
        assert backend.shape(clipped) == (4,)

        # Also test through Ops
        clipped2 = Ops.clip(a, min_val=0, max_val=5)
        assert backend.shape(clipped2) == (4,)

    def test_ops_minmax(self, backend):
        """Test min/max operations."""
        a = backend.array([[1, 2, 3], [4, 5, 6]])

        # Test min/max
        min_val = ops_min(a)
        max_val = ops_max(a)

        assert backend.ndim(min_val) == 0  # scalar
        assert backend.ndim(max_val) == 0  # scalar

        # Test with axis
        min_axis = Ops.min(a, axis=0)
        max_axis = Ops.max(a, axis=1)

        assert backend.shape(min_axis) == (3,)
        assert backend.shape(max_axis) == (2,)
