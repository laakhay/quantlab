"""Centralized type definitions for laakhay-quantlab."""

from .base import (
    Shape, Axis, Array, ArrayLike,
    NumpyNDArray, TorchTensor, JaxArray, Scalar
)

from .backend import (
    BackendName, ArrayFunction, GradFunction, UnaryOp, BinaryOp, BackendProtocol
)

__all__ = [
    'Shape', 'Axis', 'Array', 'ArrayLike', 'Scalar',
    'NumpyNDArray', 'TorchTensor', 'JaxArray',
    'BackendName', 'ArrayFunction', 'GradFunction', 'UnaryOp', 'BinaryOp', 'BackendProtocol'
]