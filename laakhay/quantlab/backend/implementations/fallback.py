"""Fallback backend for when no array libraries are installed."""

from ..interface import AbstractBackend


class FallbackBackend(AbstractBackend):
    """Fallback backend that raises helpful errors."""

    name = "fallback"

    def __init__(self):
        self._error_msg = (
            "No array backend available. Install one of:\n"
            "  pip install numpy\n"
            "  pip install jax jaxlib\n"
            "  pip install torch"
        )

    def _not_available(self, *args, **kwargs):
        raise RuntimeError(self._error_msg)

    def __getattribute__(self, name):
        if name in (
            "name",
            "_error_msg",
            "_not_available",
            "__class__",
            "__init__",
        ):
            return object.__getattribute__(self, name)
        return object.__getattribute__(self, "_not_available")
    
    # Add explicit methods to satisfy the protocol
    def gather(self, a, indices, axis=0):
        return self._not_available()
    
    def norm(self, a, ord=None, axis=None):
        return self._not_available()
    
    def solve(self, a, b):
        return self._not_available()
    
    def inv(self, a):
        return self._not_available()
    
    def det(self, a):
        return self._not_available()
