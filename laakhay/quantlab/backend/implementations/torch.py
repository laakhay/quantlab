"""PyTorch backend."""

from __future__ import annotations

from ...types import Array, ArrayLike, Axis, Shape
from ..interface import AbstractBackend

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None


class TorchBackend(AbstractBackend):
    """PyTorch backend."""

    name = "torch"

    def __init__(self) -> None:
        if not HAS_TORCH:
            raise ImportError("PyTorch not installed. Run: pip install torch")

    def array(self, data: ArrayLike, dtype: any = None, device: any = None) -> Array:
        device_str = str(device) if device else None
        return torch.tensor(data, dtype=dtype, device=device_str)

    def zeros(self, shape: Shape, dtype: any = None, device: any = None) -> Array:
        device_str = str(device) if device else None
        return torch.zeros(shape, dtype=dtype, device=device_str)

    def ones(self, shape: Shape, dtype: any = None, device: any = None) -> Array:
        device_str = str(device) if device else None
        return torch.ones(shape, dtype=dtype, device=device_str)

    def arange(self, *args, dtype: any = None, device: any = None) -> Array:
        device_str = str(device) if device else None
        return torch.arange(*args, dtype=dtype, device=device_str)

    def linspace(
        self, start: float, stop: float, num: int = 50, device: any = None
    ) -> Array:
        device_str = str(device) if device else None
        return torch.linspace(start, stop, num, device=device_str)

    def eye(self, n: int, m: int | None = None, device: any = None) -> Array:
        device_str = str(device) if device else None
        return torch.eye(n, m or n, device=device_str)

    def reshape(self, a: Array, shape: Shape) -> Array:
        return torch.reshape(a, shape)

    def transpose(self, a: Array, axes: list[int] | None = None) -> Array:
        if axes is None:
            return a.T
        return a.permute(axes)

    def concat(self, arrays: list[Array], axis: int = 0) -> Array:
        return torch.cat(arrays, dim=axis)

    def stack(self, arrays: list[Array], axis: int = 0) -> Array:
        return torch.stack(arrays, dim=axis)

    def split(self, a: Array, indices: int | list[int], axis: int = 0) -> list[Array]:
        if isinstance(indices, int):
            return list(torch.chunk(a, indices, dim=axis))
        shape_axis = a.shape[axis]
        split_sizes = []
        prev_idx = 0
        for idx in indices:
            split_sizes.append(idx - prev_idx)
            prev_idx = idx
        split_sizes.append(shape_axis - prev_idx)
        return list(torch.split(a, split_sizes, dim=axis))

    def sum(self, a: Array, axis: Axis = None, keepdims: bool = False) -> Array:
        if axis is None:
            return torch.sum(a)
        return torch.sum(a, dim=axis, keepdim=keepdims)

    def mean(self, a: Array, axis: Axis = None, keepdims: bool = False) -> Array:
        if axis is None:
            return torch.mean(a)
        return torch.mean(a, dim=axis, keepdim=keepdims)

    def std(self, a: Array, axis: Axis = None, keepdims: bool = False) -> Array:
        if axis is None:
            return torch.std(a, unbiased=False)
        return torch.std(a, dim=axis, keepdim=keepdims, unbiased=False)

    def min(self, a: Array, axis: Axis = None, keepdims: bool = False) -> Array:
        if axis is None:
            return torch.min(a)
        return torch.min(a, dim=axis, keepdim=keepdims).values

    def max(self, a: Array, axis: Axis = None, keepdims: bool = False) -> Array:
        if axis is None:
            return torch.max(a)
        return torch.max(a, dim=axis, keepdim=keepdims).values

    def argmin(self, a: Array, axis: Axis = None) -> Array:
        return torch.argmin(a, dim=axis)

    def argmax(self, a: Array, axis: Axis = None) -> Array:
        return torch.argmax(a, dim=axis)

    def pow(self, a: Array, b: Array) -> Array:
        return torch.pow(a, b)

    def matmul(self, a: Array, b: Array) -> Array:
        return torch.matmul(a, b)

    def abs(self, a: Array) -> Array:
        return torch.abs(a)

    def sign(self, a: Array) -> Array:
        return torch.sign(a)

    def exp(self, a: Array) -> Array:
        return torch.exp(a)

    def log(self, a: Array) -> Array:
        return torch.log(a)

    def sqrt(self, a: Array) -> Array:
        return torch.sqrt(a)

    def sin(self, a: Array) -> Array:
        return torch.sin(a)

    def cos(self, a: Array) -> Array:
        return torch.cos(a)

    def tanh(self, a: Array) -> Array:
        return torch.tanh(a)

    def where(self, cond: Array, x: Array, y: Array) -> Array:
        return torch.where(cond, x, y)

    def clip(
        self, a: Array, min: float | None = None, max: float | None = None
    ) -> Array:
        return torch.clamp(a, min=min, max=max)

    def maximum(self, a: Array, b: Array) -> Array:
        return torch.maximum(a, b)

    def minimum(self, a: Array, b: Array) -> Array:
        return torch.minimum(a, b)

    def shape(self, a: Array) -> Shape:
        return tuple(a.shape)

    def dtype(self, a: Array) -> any:
        return a.dtype

    def ndim(self, a: Array) -> int:
        return a.ndim

    def size(self, a: Array) -> int:
        return a.numel()

    def is_array(self, obj: any) -> bool:
        return isinstance(obj, torch.Tensor)

    def to_numpy(self, a: Array) -> Array:
        return a.detach().cpu().numpy()

    def from_numpy(self, a: Array) -> Array:
        return torch.from_numpy(a)

    def cast(self, a: Array, dtype: any) -> Array:
        return a.type(dtype)

    def eq(self, a: Array, b: Array) -> Array:
        return torch.eq(a, b)

    def ne(self, a: Array, b: Array) -> Array:
        return torch.ne(a, b)

    def lt(self, a: Array, b: Array) -> Array:
        return torch.lt(a, b)

    def le(self, a: Array, b: Array) -> Array:
        return torch.le(a, b)

    def gt(self, a: Array, b: Array) -> Array:
        return torch.gt(a, b)

    def ge(self, a: Array, b: Array) -> Array:
        return torch.ge(a, b)

    def isnan(self, a: Array) -> Array:
        return torch.isnan(a)

    def isinf(self, a: Array) -> Array:
        return torch.isinf(a)

    def isfinite(self, a: Array) -> Array:
        return torch.isfinite(a)

    def add(self, a: Array, b: Array) -> Array:
        return torch.add(a, b)

    def sub(self, a: Array, b: Array) -> Array:
        return torch.sub(a, b)

    def mul(self, a: Array, b: Array) -> Array:
        return torch.mul(a, b)

    def div(self, a: Array, b: Array) -> Array:
        return torch.div(a, b)

    def dot(self, a: Array, b: Array) -> Array:
        return torch.dot(a, b) if a.ndim == 1 and b.ndim == 1 else torch.matmul(a, b)

    def tan(self, a: Array) -> Array:
        return torch.tan(a)

    def sinh(self, a: Array) -> Array:
        return torch.sinh(a)

    def cosh(self, a: Array) -> Array:
        return torch.cosh(a)

    def log10(self, a: Array) -> Array:
        return torch.log10(a)

    def var(self, a: Array, axis: Axis = None, keepdims: bool = False) -> Array:
        if axis is None:
            return torch.var(a, unbiased=False)
        return torch.var(a, dim=axis, keepdim=keepdims, unbiased=False)

    def prod(self, a: Array, axis: Axis = None, keepdims: bool = False) -> Array:
        if axis is None:
            return torch.prod(a)
        return torch.prod(a, dim=axis, keepdim=keepdims)

    def squeeze(self, a: Array, axis: Axis = None) -> Array:
        if axis is None:
            return torch.squeeze(a)
        return torch.squeeze(a, dim=axis)

    def expand_dims(self, a: Array, axis: int) -> Array:
        return torch.unsqueeze(a, dim=axis)

    def take(self, a: Array, indices: Array, axis: int | None = None) -> Array:
        if axis is None:
            return a.flatten()[indices]
        return torch.index_select(a, dim=axis, index=indices)

    def is_gpu_available(self) -> bool:
        return torch.cuda.is_available()

    def device(self, array: Array) -> any:
        from ..device import Device

        device = array.device
        if device.type == "cuda":
            return Device.gpu(device.index or 0)
        return Device.cpu()

    def to_device(self, array: Array, device: any) -> Array:
        return array.to(str(device))
