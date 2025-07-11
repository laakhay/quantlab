"""Base type definitions used throughout the library."""

from __future__ import annotations

# Import for type checking only
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import torch
    import jax.numpy as jnp
    
    Array = np.ndarray | torch.Tensor | jnp.ndarray
else:
    Array = any

# For runtime, keep the class definitions for when libraries aren't available
try:
    import numpy as np
    NumpyNDArray = np.ndarray
except ImportError:
    class _NumpyNDArray:
        pass
    NumpyNDArray = _NumpyNDArray

try:
    import torch
    TorchTensor = torch.Tensor
except ImportError:
    class _TorchTensor:
        pass
    TorchTensor = _TorchTensor

try:
    import jax.numpy as jnp
    JaxArray = jnp.ndarray
except ImportError:
    class _JaxArray:
        pass
    JaxArray = _JaxArray

# Runtime type definitions - use typing-compatible syntax
import sys
if sys.version_info >= (3, 10):
    Scalar = float | int
    Shape = tuple[int, ...]
    Axis = int | tuple[int, ...] | None
else:
    # For Python 3.9 and below
    Scalar = any  # Will be float | int in annotations
    Shape = tuple  # Will be tuple[int, ...] in annotations
    Axis = any  # Will be int | tuple[int, ...] | None in annotations

ArrayLike = any
Size = int