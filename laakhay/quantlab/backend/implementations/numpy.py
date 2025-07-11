"""NumPy backend."""

from __future__ import annotations

from ...types import Array, ArrayLike, Axis, Shape
from ..interface import AbstractBackend

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None


class NumpyBackend(AbstractBackend):
    """NumPy backend."""

    name = "numpy"

    def __init__(self) -> None:
        if not HAS_NUMPY:
            raise ImportError("NumPy not installed. Run: pip install numpy")

    def array(self, data: ArrayLike, dtype: any = None, device: any = None) -> Array:
        return np.array(data, dtype=dtype)

    def zeros(self, shape: Shape, dtype: any = None, device: any = None) -> Array:
        return np.zeros(shape, dtype=dtype)

    def ones(self, shape: Shape, dtype: any = None, device: any = None) -> Array:
        return np.ones(shape, dtype=dtype)

    def arange(self, *args, dtype: any = None, device: any = None) -> Array:
        return np.arange(*args, dtype=dtype)

    def linspace(
        self, start: float, stop: float, num: int = 50, device: any = None
    ) -> Array:
        return np.linspace(start, stop, num)

    def eye(self, n: int, m: int | None = None, device: any = None) -> Array:
        return np.eye(n, m)

    def reshape(self, a: Array, shape: Shape) -> Array:
        return np.reshape(a, shape)

    def transpose(self, a: Array, axes: list[int] | None = None) -> Array:
        return np.transpose(a, axes)

    def concat(self, arrays: list[Array], axis: int = 0) -> Array:
        return np.concatenate(arrays, axis)

    def stack(self, arrays: list[Array], axis: int = 0) -> Array:
        return np.stack(arrays, axis)

    def split(self, a: Array, indices: int | list[int], axis: int = 0) -> list[Array]:
        return np.split(a, indices, axis)

    def sum(self, a: Array, axis: Axis = None, keepdims: bool = False) -> Array:
        return np.sum(a, axis=axis, keepdims=keepdims)

    def mean(self, a: Array, axis: Axis = None, keepdims: bool = False) -> Array:
        return np.mean(a, axis=axis, keepdims=keepdims)

    def std(self, a: Array, axis: Axis = None, keepdims: bool = False) -> Array:
        return np.std(a, axis=axis, keepdims=keepdims)

    def min(self, a: Array, axis: Axis = None, keepdims: bool = False) -> Array:
        return np.min(a, axis=axis, keepdims=keepdims)

    def max(self, a: Array, axis: Axis = None, keepdims: bool = False) -> Array:
        return np.max(a, axis=axis, keepdims=keepdims)

    def argmin(self, a: Array, axis: Axis = None) -> Array:
        return np.argmin(a, axis=axis)

    def argmax(self, a: Array, axis: Axis = None) -> Array:
        return np.argmax(a, axis=axis)

    def pow(self, a: Array, b: Array) -> Array:
        return np.power(a, b)

    def matmul(self, a: Array, b: Array) -> Array:
        return np.matmul(a, b)

    def abs(self, a: Array) -> Array:
        return np.abs(a)

    def sign(self, a: Array) -> Array:
        return np.sign(a)

    def exp(self, a: Array) -> Array:
        return np.exp(a)

    def log(self, a: Array) -> Array:
        return np.log(a)

    def sqrt(self, a: Array) -> Array:
        return np.sqrt(a)

    def sin(self, a: Array) -> Array:
        return np.sin(a)

    def cos(self, a: Array) -> Array:
        return np.cos(a)

    def tanh(self, a: Array) -> Array:
        return np.tanh(a)

    def where(self, cond: Array, x: Array, y: Array) -> Array:
        return np.where(cond, x, y)

    def clip(
        self, a: Array, min: float | None = None, max: float | None = None
    ) -> Array:
        return np.clip(a, min, max)

    def maximum(self, a: Array, b: Array) -> Array:
        return np.maximum(a, b)

    def minimum(self, a: Array, b: Array) -> Array:
        return np.minimum(a, b)

    def shape(self, a: Array) -> Shape:
        return a.shape

    def dtype(self, a: Array) -> any:
        return a.dtype

    def ndim(self, a: Array) -> int:
        return a.ndim

    def size(self, a: Array) -> int:
        return a.size

    def is_array(self, obj: any) -> bool:
        return isinstance(obj, np.ndarray)

    def to_numpy(self, a: Array) -> Array:
        return a

    def from_numpy(self, a: Array) -> Array:
        return a

    def cast(self, a: Array, dtype: any) -> Array:
        return a.astype(dtype)

    def eq(self, a: Array, b: Array) -> Array:
        return np.equal(a, b)

    def ne(self, a: Array, b: Array) -> Array:
        return np.not_equal(a, b)

    def lt(self, a: Array, b: Array) -> Array:
        return np.less(a, b)

    def le(self, a: Array, b: Array) -> Array:
        return np.less_equal(a, b)

    def gt(self, a: Array, b: Array) -> Array:
        return np.greater(a, b)

    def ge(self, a: Array, b: Array) -> Array:
        return np.greater_equal(a, b)

    def isnan(self, a: Array) -> Array:
        return np.isnan(a)

    def isinf(self, a: Array) -> Array:
        return np.isinf(a)

    def isfinite(self, a: Array) -> Array:
        return np.isfinite(a)

    def add(self, a: Array, b: Array) -> Array:
        return np.add(a, b)

    def sub(self, a: Array, b: Array) -> Array:
        return np.subtract(a, b)

    def mul(self, a: Array, b: Array) -> Array:
        return np.multiply(a, b)

    def div(self, a: Array, b: Array) -> Array:
        return np.divide(a, b)

    def dot(self, a: Array, b: Array) -> Array:
        return np.dot(a, b)

    def tan(self, a: Array) -> Array:
        return np.tan(a)

    def sinh(self, a: Array) -> Array:
        return np.sinh(a)

    def cosh(self, a: Array) -> Array:
        return np.cosh(a)

    def log10(self, a: Array) -> Array:
        return np.log10(a)

    def var(self, a: Array, axis: Axis = None, keepdims: bool = False) -> Array:
        return np.var(a, axis=axis, keepdims=keepdims)

    def prod(self, a: Array, axis: Axis = None, keepdims: bool = False) -> Array:
        return np.prod(a, axis=axis, keepdims=keepdims)

    def squeeze(self, a: Array, axis: Axis = None) -> Array:
        return np.squeeze(a, axis=axis)

    def expand_dims(self, a: Array, axis: int) -> Array:
        return np.expand_dims(a, axis=axis)

    def take(self, a: Array, indices: Array, axis: int | None = None) -> Array:
        return np.take(a, indices, axis=axis)

    def is_gpu_available(self) -> bool:
        return False

    def device(self, array: Array) -> any:
        from ..device import Device

        return Device.cpu()

    def to_device(self, array: Array, device: any) -> Array:
        return array

    def gather(self, a: Array, indices: Array, axis: int = 0) -> Array:
        return np.take(a, indices, axis=axis)

    def norm(self, a: Array, ord: any = None, axis: Axis = None) -> Array:
        return np.linalg.norm(a, ord=ord, axis=axis)

    def solve(self, a: Array, b: Array) -> Array:
        return np.linalg.solve(a, b)

    def inv(self, a: Array) -> Array:
        return np.linalg.inv(a)

    def det(self, a: Array) -> Array:
        return np.linalg.det(a)

    def erf(self, x: Array) -> Array:
        try:
            from scipy.special import erf

            return erf(x)
        except ImportError:
            import numpy as _np

            a1 = 0.254829592
            a2 = -0.284496736
            a3 = 1.421413741
            a4 = -1.453152027
            a5 = 1.061405429
            p = 0.3275911
            sign = _np.sign(x)
            x = _np.abs(x)
            t = 1.0 / (1.0 + p * x)
            y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * _np.exp(
                -x * x
            )
            return sign * y

    def erfc(self, x: Array) -> Array:
        from scipy.special import erfc

        return erfc(x)

    def norm_cdf(self, x: Array) -> Array:
        try:
            from scipy.stats import norm

            return norm.cdf(x)
        except ImportError:
            return 0.5 * (1 + self.erf(x / np.sqrt(2)))

    def norm_pdf(self, x: Array) -> Array:
        from scipy.stats import norm

        return norm.pdf(x)

    def norm_ppf(self, q: Array) -> Array:
        try:
            from scipy.stats import norm

            return norm.ppf(q)
        except ImportError:
            raise NotImplementedError(
                "norm_ppf requires scipy for numpy backend"
            ) from None

    def gamma(self, x: Array) -> Array:
        from scipy.special import gamma

        return gamma(x)

    def lgamma(self, x: Array) -> Array:
        from scipy.special import gammaln

        return gammaln(x)

    def random_key(self, seed: int) -> any:
        np.random.seed(seed)
        return seed

    def random_normal(
        self,
        key: any,
        shape: Shape,
        dtype: any = None,
        device: any = None,
    ) -> Array:
        result = np.random.normal(0.0, 1.0, shape)
        if dtype is not None:
            result = result.astype(dtype)
        return result

    def random_uniform(
        self,
        key: any,
        shape: Shape,
        dtype: any = None,
        device: any = None,
        low: float = 0.0,
        high: float = 1.0,
    ) -> Array:
        result = np.random.uniform(low, high, shape)
        if dtype is not None:
            result = result.astype(dtype)
        return result

    def random_split(self, key: any, num: int = 2) -> list[any]:
        return [key + i for i in range(num)]
