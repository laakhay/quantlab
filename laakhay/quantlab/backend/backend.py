"""Backend wrapper with autodiff."""

from ..types import Array
from .registry import get_backend as _get_backend
from .registry import infer_backend


class ArrayBackend:
    """Backend wrapper with autodiff support."""

    def __init__(self, backend=None):
        if backend is None:
            self._backend = _get_backend()
        elif isinstance(backend, str):
            self._backend = _get_backend(backend)
        else:
            self._backend = backend

    @property
    def name(self) -> str:
        return self._backend.name

    @property
    def supports_autodiff(self) -> bool:
        """Check autodiff support."""
        return self.name in ["jax", "torch"]

    def __getattr__(self, name):
        return getattr(self._backend, name)

    @staticmethod
    def from_array(array: Array) -> "ArrayBackend":
        """Create backend from array."""
        backend_name = infer_backend(array)
        if backend_name:
            return ArrayBackend(backend_name)
        return ArrayBackend()

    def grad(self, func, argnums=0):
        """Compute gradient."""
        if self.name == "jax":
            import jax

            def scalar_func(*args):
                output = func(*args)
                if hasattr(output, "shape") and output.shape:
                    return output.sum()
                return output

            return jax.grad(scalar_func, argnums=argnums)
        elif self.name == "torch":

            def torch_grad(*args):
                import torch

                tensors = []
                for i, arg in enumerate(args):
                    if i == argnums:
                        t = (
                            arg
                            if isinstance(arg, torch.Tensor)
                            else torch.tensor(arg, dtype=torch.float32)
                        )
                        t.requires_grad_(True)
                        tensors.append(t)
                    else:
                        t = arg if isinstance(arg, torch.Tensor) else torch.tensor(arg)
                        tensors.append(t)

                output = func(*tensors)

                if output.numel() > 1:
                    output = output.sum()

                output.backward()
                return tensors[argnums].grad

            return torch_grad
        else:
            raise RuntimeError(f"Backend {self.name} doesn't support autodiff")

    def value_and_grad(self, func, argnums=0):
        """Compute value and gradient."""
        if self.name == "jax":
            import jax

            def wrapped_func(*args):
                output = func(*args)
                scalar = output.sum() if hasattr(output, "shape") and output.shape else output
                return scalar

            scalar_value_and_grad = jax.value_and_grad(wrapped_func, argnums=argnums)

            def value_and_grad_wrapper(*args):
                original_output = func(*args)
                _, grad = scalar_value_and_grad(*args)
                return original_output, grad

            return value_and_grad_wrapper
        elif self.name == "torch":

            def torch_value_and_grad(*args):
                import torch

                tensors = []
                for i, arg in enumerate(args):
                    if i == argnums:
                        t = (
                            arg
                            if isinstance(arg, torch.Tensor)
                            else torch.tensor(arg, dtype=torch.float32)
                        )
                        t.requires_grad_(True)
                        tensors.append(t)
                    else:
                        t = arg if isinstance(arg, torch.Tensor) else torch.tensor(arg)
                        tensors.append(t)

                output = func(*tensors)
                value = output.detach().clone()

                if output.numel() > 1:
                    output = output.sum()

                output.backward()
                return value, tensors[argnums].grad

            return torch_value_and_grad
        else:
            raise RuntimeError(f"Backend {self.name} doesn't support value_and_grad")

    def vmap(self, func):
        """Vectorize function."""
        if self.name == "jax":
            import jax

            return jax.vmap(func)
        else:

            def vmapped(*args):
                results = [func(x, *args[1:]) for x in args[0]]
                return self._backend.stack(results)

            return vmapped

    def jit(self, func):
        """JIT compile function."""
        if self.name == "jax":
            import jax

            return jax.jit(func)
        else:
            return func


def backend(name=None) -> ArrayBackend:
    """Get backend instance."""
    return ArrayBackend(name)


def active_backend() -> ArrayBackend:
    """Get active backend."""
    return ArrayBackend()
