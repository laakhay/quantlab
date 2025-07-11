"""Backend registry."""

from __future__ import annotations

from .interface import Backend
from ..types import Array
from ..exceptions import MixedBackendError, BackendNotFoundError

_backends: dict[str, Backend] = {}
_default: str | None = None


def register(name: str, backend: Backend) -> None:
    """Register backend."""
    _backends[name] = backend


def get_backend(name: str | None = None) -> Backend:
    """Get backend by name or default."""
    name = name or _default
    if not name or name not in _backends:
        raise BackendNotFoundError(f"Backend '{name}' not found. Available: {list(_backends.keys())}")
    return _backends[name]


def set_default(name: str) -> None:
    """Set default backend."""
    global _default
    if name not in _backends:
        raise ValueError(f"Backend '{name}' not registered")
    _default = name


def list_backends() -> list[str]:
    """List registered backends."""
    return list(_backends.keys())


def has_backend(name: str) -> bool:
    """Check if backend is available."""
    return name in _backends and name != "fallback"


def infer_backend(array: any) -> str | None:
    """Infer backend from array type."""
    for name, backend in _backends.items():
        if name == "fallback":
            continue
        try:
            if backend.is_array(array):
                return name
        except:
            pass
    return None


def infer_backend_from_arrays(*arrays: any) -> str | None:
    """Infer common backend from arrays."""
    backends = {infer_backend(a) for a in arrays if a is not None}
    backends.discard(None)
    
    if len(backends) == 0:
        return _default
    elif len(backends) == 1:
        return backends.pop()
    else:
        raise MixedBackendError(f"Arrays from different backends cannot be mixed: {backends}")


def convert_array(array: any, to_backend: str, 
                  from_backend: str | None = None) -> any:
    """Convert array between backends."""
    from_backend = from_backend or infer_backend(array)
    if not from_backend:
        raise ValueError("Could not infer source backend")
    
    if from_backend == to_backend:
        return array
    
    source = get_backend(from_backend)
    target = get_backend(to_backend)
    
    numpy_array = source.to_numpy(array)
    return target.from_numpy(numpy_array)
