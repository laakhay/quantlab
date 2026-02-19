"""Utilities for pricing module."""

from functools import wraps

from laakhay.quantlab.backend import infer_backend_from_arrays


def infer_backend(func):
    """Decorator to infer backend from arguments if not provided."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        backend = kwargs.get("backend")
        if backend is None:
            # Try to infer from args[1:] (skipping self)
            arrays = [arg for arg in args[1:] if arg is not None]
            # Also check kwargs
            for k, v in kwargs.items():
                if k != "backend" and v is not None:
                    arrays.append(v)

            backend_name = infer_backend_from_arrays(*arrays)
            from laakhay.quantlab.backend import get_backend

            backend = get_backend(backend_name)
            kwargs["backend"] = backend

        return func(*args, **kwargs)

    return wrapper
