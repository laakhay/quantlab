"""Base type definitions used throughout the library."""

from __future__ import annotations

# Import for type checking only
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import jax.numpy as jnp
    import numpy as np
    import torch

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
from typing import Union

Scalar = Union[float, int]
Shape = tuple
Axis = Union[int, tuple, None]
ArrayLike = any
Size = int
