"""Backend operations."""

from __future__ import annotations

from ..types import Array, Axis
from .registry import get_backend, infer_backend_from_arrays


def ensure_same_backend(*arrays: Array) -> tuple[Array, ...]:
    """Ensure all arrays use same backend."""
    backend_name = infer_backend_from_arrays(*arrays)
    if backend_name is None:
        return arrays

    backend = get_backend(backend_name)
    return tuple(backend.from_numpy(backend.to_numpy(a)) for a in arrays)


class Ops:
    """Backend operations."""

    @staticmethod
    def binary_op(op_name: str, a: Array, b: Array) -> Array:
        backend = get_backend(infer_backend_from_arrays(a, b))
        op = getattr(backend, op_name)
        return op(a, b)

    @staticmethod
    def unary_op(op_name: str, a: Array, **kwargs) -> Array:
        backend = get_backend(infer_backend_from_arrays(a))
        op = getattr(backend, op_name)
        return op(a, **kwargs)

    @staticmethod
    def add(a: Array, b: Array) -> Array:
        return Ops.binary_op("add", a, b)

    @staticmethod
    def sub(a: Array, b: Array) -> Array:
        return Ops.binary_op("sub", a, b)

    @staticmethod
    def mul(a: Array, b: Array) -> Array:
        return Ops.binary_op("mul", a, b)

    @staticmethod
    def div(a: Array, b: Array) -> Array:
        return Ops.binary_op("div", a, b)

    @staticmethod
    def matmul(a: Array, b: Array) -> Array:
        return Ops.binary_op("matmul", a, b)

    @staticmethod
    def sum(a: Array, axis: Axis = None, keepdims: bool = False) -> Array:
        return Ops.unary_op("sum", a, axis=axis, keepdims=keepdims)

    @staticmethod
    def mean(a: Array, axis: Axis = None, keepdims: bool = False) -> Array:
        return Ops.unary_op("mean", a, axis=axis, keepdims=keepdims)

    @staticmethod
    def std(a: Array, axis: Axis = None, keepdims: bool = False) -> Array:
        return Ops.unary_op("std", a, axis=axis, keepdims=keepdims)

    @staticmethod
    def exp(a: Array) -> Array:
        return Ops.unary_op("exp", a)

    @staticmethod
    def log(a: Array) -> Array:
        return Ops.unary_op("log", a)

    @staticmethod
    def sqrt(a: Array) -> Array:
        return Ops.unary_op("sqrt", a)

    @staticmethod
    def abs(a: Array) -> Array:
        return Ops.unary_op("abs", a)

    @staticmethod
    def reshape(a: Array, shape: tuple[int, ...]) -> Array:
        return Ops.unary_op("reshape", a, shape=shape)

    @staticmethod
    def transpose(a: Array, axes: list[int] | None = None) -> Array:
        return Ops.unary_op("transpose", a, axes=axes)

    @staticmethod
    def eq(a: Array, b: Array) -> Array:
        return Ops.binary_op("eq", a, b)

    @staticmethod
    def lt(a: Array, b: Array) -> Array:
        return Ops.binary_op("lt", a, b)

    @staticmethod
    def gt(a: Array, b: Array) -> Array:
        return Ops.binary_op("gt", a, b)

    @staticmethod
    def where(cond: Array, x: Array, y: Array) -> Array:
        backend = get_backend(infer_backend_from_arrays(cond, x, y))
        return backend.where(cond, x, y)

    @staticmethod
    def clip(a: Array, min_val: float | None = None, max_val: float | None = None) -> Array:
        return Ops.unary_op("clip", a, min=min_val, max=max_val)

    @staticmethod
    def max(a: Array, axis: Axis = None, keepdims: bool = False) -> Array:
        return Ops.unary_op("max", a, axis=axis, keepdims=keepdims)

    @staticmethod
    def min(a: Array, axis: Axis = None, keepdims: bool = False) -> Array:
        return Ops.unary_op("min", a, axis=axis, keepdims=keepdims)


add = Ops.add
sub = Ops.sub
mul = Ops.mul
div = Ops.div
matmul = Ops.matmul
exp = Ops.exp
log = Ops.log
sqrt = Ops.sqrt
abs = Ops.abs
sum = Ops.sum
mean = Ops.mean
std = Ops.std
max = Ops.max
min = Ops.min
where = Ops.where
clip = Ops.clip
reshape = Ops.reshape
transpose = Ops.transpose
eq = Ops.eq
lt = Ops.lt
gt = Ops.gt
