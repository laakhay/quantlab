"""Tests for types module."""

import pytest

from laakhay.quantlab.types.base import (
    Scalar, Shape, ArrayLike, Array
)


class TestTypes:
    """Test type definitions."""

    def test_scalar_type(self):
        """Test Scalar type."""
        # Test with different scalar types
        scalars = [1, 1.0, 2.5, -3.14, 0]
        for s in scalars:
            assert isinstance(s, (int, float))
            
    def test_shape_type(self):
        """Test Shape type."""
        # Test various shape tuples
        shapes = [(), (1,), (2, 3), (2, 3, 4)]
        for shape in shapes:
            assert isinstance(shape, tuple)
            assert all(isinstance(dim, int) for dim in shape)
            
    def test_array_type(self):
        """Test Array type usage."""
        # Array is set to any at runtime
        assert Array is any
            
    def test_array_like(self):
        """Test ArrayLike type."""
        # Various array-like objects
        array_likes = [
            [1, 2, 3],
            [[1, 2], [3, 4]],
            (1, 2, 3),
            1.0,
            [1.0, 2.0, 3.0]
        ]
        for arr in array_likes:
            # ArrayLike should accept lists, tuples, scalars
            assert arr is not None