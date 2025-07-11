"""Exceptions for laakhay-quantlab."""

class BackendError(Exception):
    """Base exception for backend errors."""
    pass

class MixedBackendError(BackendError):
    """Raised when arrays from different backends are mixed."""
    pass

class BackendNotFoundError(BackendError):
    """Raised when requested backend is not available."""
    pass

class GuidelineError(Exception):
    """Raised when implementation violates CLAUDE.md guidelines."""
    pass

__all__ = [
    'BackendError',
    'MixedBackendError', 
    'BackendNotFoundError',
    'GuidelineError'
]