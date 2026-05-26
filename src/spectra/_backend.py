"""Array-library abstraction.

This module provides a single accessor, :func:`get_array_module`, that returns
the active numerical backend. For the entire Horizon 1 of the project the
backend is NumPy and nothing else; the indirection exists so a CuPy (or any
other numpy-compatible) backend can be slotted in later without rewriting the
Tensor class or the operations on top of it.

The pattern mirrors how mature numpy-compatible libraries (CuPy, JAX,
PyTorch's numpy interop) are typically wired together, where every numerical
call site goes through a single module accessor instead of importing numpy
directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from types import ModuleType


_DEFAULT_DTYPE: np.dtype[np.floating] = np.dtype(np.float64)


def get_array_module() -> ModuleType:
    """Return the active numerical backend.

    For Horizon 1 this is always :mod:`numpy`. The indirection is kept so that
    future versions can dispatch to other numpy-compatible backends (CuPy on
    GPU is the planned candidate) by changing this single function.
    """
    return np


def default_dtype() -> np.dtype[np.floating]:
    """Return the default floating-point dtype used by new tensors.

    Set to :class:`numpy.float64`. The choice favours numerical precision over
    memory and throughput, which suits the spectral and geometric analysis
    parts of the project (eigendecompositions, singular value decompositions,
    Hessian top-eigenvalue iterations). For deep-learning workloads that
    benefit from lower precision, individual tensors can still be constructed
    with an explicit ``dtype`` argument.
    """
    return _DEFAULT_DTYPE
