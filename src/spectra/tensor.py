"""The :class:`Tensor` class.

A :class:`Tensor` is a thin wrapper around a :class:`numpy.ndarray` together
with the metadata required by automatic differentiation. This module defines
the data structure and its construction, inspection, and equality semantics
only; the autograd engine and the mathematical operations on tensors are
introduced in subsequent modules.

A Tensor is mutable: its underlying data buffer can be overwritten in place,
the same way :class:`numpy.ndarray` is mutable. This mirrors PyTorch's design
and is chosen for memory efficiency, with the understanding that in-place
mutations on tensors that participate in a backward pass are the user's
responsibility to avoid.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray

from spectra._backend import default_dtype, get_array_module


class Tensor:
    """An n-dimensional array with optional gradient tracking.

    Parameters
    ----------
    data
        The data to wrap. Accepts anything :func:`numpy.asarray` accepts:
        a :class:`numpy.ndarray`, a Python scalar, a (possibly nested) sequence,
        or another :class:`Tensor` (in which case its underlying array is
        re-used without copying).
    dtype
        The desired dtype of the resulting tensor. If ``None``, integer inputs
        are kept as integers and floating-point inputs are coerced to the
        framework default (see :func:`spectra._backend.default_dtype`). To force
        the default explicitly, pass ``dtype=spectra._backend.default_dtype()``.
    requires_grad
        Whether this tensor should accumulate gradients during a backward
        pass. The flag is stored but unused in this module; the autograd
        engine introduced in a later module reads it.

    Attributes
    ----------
    data : numpy.ndarray
        The underlying array. Mutating ``data`` mutates the tensor.
    requires_grad : bool
        Whether gradients should be tracked.
    grad : numpy.ndarray | None
        The accumulated gradient, or ``None`` if no backward pass has run.
    """

    __slots__ = ("_data", "grad", "requires_grad")

    def __init__(
        self,
        data: ArrayLike | Tensor,
        dtype: DTypeLike | None = None,
        requires_grad: bool = False,
    ) -> None:
        xp = get_array_module()
        source = data._data if isinstance(data, Tensor) else xp.asarray(data)

        if dtype is None:
            if np.issubdtype(source.dtype, np.floating):
                target_dtype: np.dtype[Any] = default_dtype()
            else:
                target_dtype = source.dtype
        else:
            target_dtype = np.dtype(dtype)

        if source.dtype != target_dtype:
            source = source.astype(target_dtype)

        self._data: NDArray[Any] = source
        self.requires_grad: bool = bool(requires_grad)
        self.grad: NDArray[Any] | None = None

    @property
    def data(self) -> NDArray[Any]:
        """The underlying numpy array."""
        return self._data

    @data.setter
    def data(self, value: ArrayLike) -> None:
        xp = get_array_module()
        new = xp.asarray(value)
        if new.shape != self._data.shape:
            msg = (
                f"cannot assign data with shape {new.shape} to a tensor of "
                f"shape {self._data.shape}; use Tensor(...) to create a new tensor"
            )
            raise ValueError(msg)
        self._data = new.astype(self._data.dtype, copy=False)

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape of the tensor as a tuple of dimensions."""
        return self._data.shape

    @property
    def ndim(self) -> int:
        """The number of dimensions."""
        return self._data.ndim

    @property
    def size(self) -> int:
        """The total number of elements."""
        return int(self._data.size)

    @property
    def dtype(self) -> np.dtype[Any]:
        """The dtype of the underlying array."""
        return self._data.dtype

    def numpy(self) -> NDArray[Any]:
        """Return a copy of the underlying array.

        Detaches the tensor from any gradient bookkeeping by virtue of
        returning a plain :class:`numpy.ndarray` that does not share state with
        the autograd engine.
        """
        return self._data.copy()

    def __eq__(self, other: object) -> bool:
        """Two tensors are equal iff their data, dtype, and grad flag match.

        Equality is structural: two tensors with NaN in the same positions and
        equal non-NaN values are considered equal, in deviation from IEEE 754
        which holds that ``NaN != NaN``. This is the same convention
        :func:`numpy.array_equal` exposes via its ``equal_nan`` argument.

        Elementwise comparison (the ``==`` operator that returns a boolean
        array, the way it works on :class:`numpy.ndarray`) is intentionally
        not what this method does. That operation will be exposed later as a
        differentiable ``Tensor.eq`` method, leaving Python's ``==`` free to
        mean structural equality as :class:`object` defines it.
        """
        if not isinstance(other, Tensor):
            return NotImplemented
        if self._data.shape != other._data.shape:
            return False
        if self._data.dtype != other._data.dtype:
            return False
        if self.requires_grad != other.requires_grad:
            return False
        if np.issubdtype(self._data.dtype, np.floating):
            return bool(np.array_equal(self._data, other._data, equal_nan=True))
        return bool(np.array_equal(self._data, other._data))

    # Tensors are unhashable: their underlying array is mutable, which is
    # incompatible with hashing. Setting ``__hash__ = None`` is the canonical
    # Python pattern for marking a class as unhashable.
    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        cls = type(self).__name__
        if self._data.size > 8:
            with np.printoptions(threshold=8, edgeitems=2):
                body = np.array2string(self._data, separator=", ")
        else:
            body = np.array2string(self._data, separator=", ")
        parts = [f"{cls}({body}", f"dtype={self._data.dtype}"]
        if self.requires_grad:
            parts.append("requires_grad=True")
        return ", ".join(parts) + ")"
