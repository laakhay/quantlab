"""JAX backend."""

from __future__ import annotations

from ...types import Array, Shape, ArrayLike
from .numpy import NumpyBackend

try:
    import numpy as np
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    np = jax = jnp = None


class JaxBackend(NumpyBackend):
    """JAX backend."""
    
    name = "jax"
    
    def __init__(self) -> None:
        if not HAS_JAX:
            raise ImportError("JAX not installed. Run: pip install jax jaxlib")
    
    def array(self, data: ArrayLike, dtype: any = None, device: any = None) -> Array:
        return jnp.array(data, dtype=dtype)
    
    def zeros(self, shape: Shape, dtype: any = None, device: any = None) -> Array:
        return jnp.zeros(shape, dtype=dtype)
    
    def ones(self, shape: Shape, dtype: any = None, device: any = None) -> Array:
        return jnp.ones(shape, dtype=dtype)
    
    def arange(self, *args, dtype: any = None, device: any = None) -> Array:
        return jnp.arange(*args, dtype=dtype)
    
    def linspace(self, start: float, stop: float, num: int = 50, device: any = None) -> Array:
        return jnp.linspace(start, stop, num)
    
    def eye(self, n: int, m: int | None = None, device: any = None) -> Array:
        return jnp.eye(n, m)
    
    def is_array(self, obj: any) -> bool:
        return isinstance(obj, jnp.ndarray)
    
    def to_numpy(self, a: Array) -> Array:
        return np.array(a) if HAS_JAX else a
    
    def from_numpy(self, a: Array) -> Array:
        return jnp.array(a)
    
    def pow(self, a: Array, b: Array) -> Array:
        return jnp.power(a, b)
    
    def matmul(self, a: Array, b: Array) -> Array:
        return jnp.matmul(a, b)
    
    def concat(self, arrays: list[Array], axis: int = 0) -> Array:
        return jnp.concatenate(arrays, axis)
    
    def stack(self, arrays: list[Array], axis: int = 0) -> Array:
        return jnp.stack(arrays, axis)
    
    def add(self, a: Array, b: Array) -> Array:
        return jnp.add(a, b)
    
    def sub(self, a: Array, b: Array) -> Array:
        return jnp.subtract(a, b)
    
    def mul(self, a: Array, b: Array) -> Array:
        return jnp.multiply(a, b)
    
    def div(self, a: Array, b: Array) -> Array:
        return jnp.divide(a, b)
    
    def exp(self, a: Array) -> Array:
        return jnp.exp(a)
    
    def sin(self, a: Array) -> Array:
        return jnp.sin(a)
    
    def cos(self, a: Array) -> Array:
        return jnp.cos(a)
    
    def sum(self, a: Array, axis: any = None, keepdims: bool = False) -> Array:
        return jnp.sum(a, axis=axis, keepdims=keepdims)
    
    def take(self, a: Array, indices: Array, axis: int | None = None) -> Array:
        return jnp.take(a, indices, axis=axis)