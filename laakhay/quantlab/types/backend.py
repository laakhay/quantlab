"""Backend-specific type definitions."""

from .base import Array, Shape, Axis, NumpyNDArray, TorchTensor, JaxArray

BackendName = str

ArrayFunction = any
GradFunction = any
UnaryOp = any
BinaryOp = any

BackendProtocol = any
