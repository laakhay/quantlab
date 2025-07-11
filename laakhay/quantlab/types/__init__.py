"""Centralized type definitions for laakhay-quantlab."""

from .backend import (
    ArrayFunction,
    BackendName,
    BackendProtocol,
    BinaryOp,
    GradFunction,
    UnaryOp,
)
from .base import (
    Array,
    ArrayLike,
    Axis,
    JaxArray,
    NumpyNDArray,
    Scalar,
    Shape,
    TorchTensor,
)

__all__ = [
    "Shape",
    "Axis",
    "Array",
    "ArrayLike",
    "Scalar",
    "NumpyNDArray",
    "TorchTensor",
    "JaxArray",
    "BackendName",
    "ArrayFunction",
    "GradFunction",
    "UnaryOp",
    "BinaryOp",
    "BackendProtocol",
]
