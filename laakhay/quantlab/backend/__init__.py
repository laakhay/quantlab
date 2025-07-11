"""Backend abstraction for array operations."""

from .interface import Backend, AbstractBackend
from .registry import (
    register, get_backend, set_default, list_backends,
    infer_backend, infer_backend_from_arrays, convert_array,
    has_backend
)


def _init():
    """Initialize available backends."""
    available = []
    
    try:
        from .implementations.numpy import NumpyBackend
        register("numpy", NumpyBackend())
        available.append("numpy")
    except ImportError:
        pass
    
    try:
        from .implementations.jax import JaxBackend
        register("jax", JaxBackend())
        available.append("jax")
    except ImportError:
        pass
    
    try:
        from .implementations.torch import TorchBackend
        register("torch", TorchBackend())
        available.append("torch")
    except ImportError:
        pass
    
    # Always register fallback
    from .implementations.fallback import FallbackBackend
    register("fallback", FallbackBackend())
    
    # Set default to first available real backend, or fallback
    for name in ["numpy", "jax", "torch"]:
        if name in available:
            set_default(name)
            return
    
    set_default("fallback")


_init()


# Export convenience namespace
from . import ops
from .backend import backend, active_backend, ArrayBackend


__all__ = [
    # Core
    'Backend', 'AbstractBackend', 'ArrayBackend',
    # High-level API
    'backend', 'active_backend',
    # Registry
    'register', 'get_backend', 'set_default', 'list_backends',
    'infer_backend', 'infer_backend_from_arrays', 'convert_array',
    'has_backend',
    # Ops module
    'ops',
]