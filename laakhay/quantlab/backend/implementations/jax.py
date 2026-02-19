"""JAX backend."""

from __future__ import annotations

from ...types import Array, ArrayLike, Axis, Shape
from .numpy import NumpyBackend

try:
    import jax
    import jax.numpy as jnp
    import numpy as np

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

    def gather(self, a: Array, indices: Array, axis: int = 0) -> Array:
        return jnp.take(a, indices, axis=axis)

    def norm(self, a: Array, ord: any = None, axis: Axis = None) -> Array:
        return jnp.linalg.norm(a, ord=ord, axis=axis)

    def solve(self, a: Array, b: Array) -> Array:
        return jnp.linalg.solve(a, b)

    def inv(self, a: Array) -> Array:
        return jnp.linalg.inv(a)

    def det(self, a: Array) -> Array:
        return jnp.linalg.det(a)

    def erf(self, x: Array) -> Array:
        from jax.scipy.special import erf

        return erf(x)

    def erfc(self, x: Array) -> Array:
        from jax.scipy.special import erfc

        return erfc(x)

    def norm_cdf(self, x: Array) -> Array:
        from jax.scipy.stats import norm

        return norm.cdf(x)

    def norm_pdf(self, x: Array) -> Array:
        from jax.scipy.stats import norm

        return norm.pdf(x)

    def norm_ppf(self, q: Array) -> Array:
        from jax.scipy.stats import norm

        return norm.ppf(q)

    def gamma(self, x: Array) -> Array:
        from jax.scipy.special import gamma

        return gamma(x)

    def lgamma(self, x: Array) -> Array:
        from jax.scipy.special import gammaln

        return gammaln(x)

    def random_key(self, seed: int) -> any:
        return jax.random.PRNGKey(seed)

    def random_normal(
        self,
        key: any,
        shape: Shape,
        dtype: any = None,
        device: any = None,
    ) -> Array:
        kwargs = {}
        if dtype is not None:
            kwargs["dtype"] = dtype
        return jax.random.normal(key, shape, **kwargs)

    def random_uniform(
        self,
        key: any,
        shape: Shape,
        dtype: any = None,
        device: any = None,
        low: float = 0.0,
        high: float = 1.0,
    ) -> Array:
        kwargs = {}
        if dtype is not None:
            kwargs["dtype"] = dtype
        return jax.random.uniform(key, shape, minval=low, maxval=high, **kwargs)

    def random_split(self, key: any, num: int = 2) -> list[any]:
        return jax.random.split(key, num)
