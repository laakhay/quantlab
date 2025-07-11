"""Backend implementations."""

from .fallback import FallbackBackend

__all__ = ['FallbackBackend']

# Import actual backends only if available
try:
    from .numpy import NumpyBackend
    __all__.append('NumpyBackend')
except ImportError:
    pass

try:
    from .jax import JaxBackend
    __all__.append('JaxBackend')
except ImportError:
    pass

try:
    from .torch import TorchBackend
    __all__.append('TorchBackend')
except ImportError:
    pass