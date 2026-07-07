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

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray

from spectra._backend import default_dtype, get_array_module

if TYPE_CHECKING:
    from spectra.autograd import Function


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

    __slots__ = ("_ctx", "_data", "grad", "requires_grad")

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
        self._ctx: Function | None = None

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

    # -- Differentiable operations ------------------------------------------

    def __add__(self, other: Tensor | ArrayLike) -> Tensor:
        from spectra import ops

        return ops.Add.apply(self, _coerce(other))

    def __radd__(self, other: Tensor | ArrayLike) -> Tensor:
        return self.__add__(other)

    def __mul__(self, other: Tensor | ArrayLike) -> Tensor:
        from spectra import ops

        return ops.Mul.apply(self, _coerce(other))

    def __rmul__(self, other: Tensor | ArrayLike) -> Tensor:
        return self.__mul__(other)

    def __neg__(self) -> Tensor:
        from spectra import ops

        return ops.Neg.apply(self)

    def __sub__(self, other: Tensor | ArrayLike) -> Tensor:
        return self.__add__(-_coerce(other))

    def __rsub__(self, other: Tensor | ArrayLike) -> Tensor:
        return _coerce(other).__add__(-self)

    def __matmul__(self, other: Tensor | ArrayLike) -> Tensor:
        from spectra import ops

        return ops.MatMul.apply(self, _coerce(other))

    def __pow__(self, exponent: float) -> Tensor:
        from spectra import ops

        return ops.Pow.apply(self, exponent=exponent)

    def sum(self, axis: int | None = None, keepdims: bool = False) -> Tensor:
        """Sum over all elements, or along ``axis``."""
        from spectra import ops

        return ops.Sum.apply(self, axis=axis, keepdims=keepdims)

    def mean(self, axis: int | None = None, keepdims: bool = False) -> Tensor:
        """Arithmetic mean over all elements, or along ``axis``."""
        from spectra import ops

        return ops.Mean.apply(self, axis=axis, keepdims=keepdims)

    def relu(self) -> Tensor:
        """Rectified linear unit, elementwise max(x, 0)."""
        from spectra import ops

        return ops.ReLU.apply(self)

    def exp(self) -> Tensor:
        """Elementwise exponential."""
        from spectra import ops

        return ops.Exp.apply(self)

    def log(self) -> Tensor:
        """Elementwise natural logarithm."""
        from spectra import ops

        return ops.Log.apply(self)

    # -- Backward pass -------------------------------------------------------

    def backward(self, grad: ArrayLike | None = None) -> None:
        """Run reverse-mode differentiation from this tensor.

        Computes dself/dx for every tensor x in the graph with
        ``requires_grad=True`` and accumulates the result into ``x.grad``.

        Parameters
        ----------
        grad
            The gradient of the final objective with respect to this tensor.
            May be omitted only for single-element tensors, where it defaults
            to 1, the usual convention for a scalar loss.
        """
        if not self.requires_grad:
            msg = "backward() called on a tensor that does not require grad"
            raise RuntimeError(msg)
        if grad is None:
            if self._data.size != 1:
                msg = "grad must be provided when calling backward() on a non-scalar tensor"
                raise RuntimeError(msg)
            seed = np.ones_like(self._data)
        else:
            seed = np.asarray(grad, dtype=self._data.dtype)
            if seed.shape != self._data.shape:
                msg = f"grad shape {seed.shape} does not match tensor shape {self._data.shape}"
                raise ValueError(msg)

        # Iterative depth-first topological sort. Recursion is avoided so
        # arbitrarily deep graphs cannot overflow the Python call stack.
        order: list[Tensor] = []
        visited: set[int] = set()
        stack: list[tuple[Tensor, bool]] = [(self, False)]
        while stack:
            node, processed = stack.pop()
            if processed:
                order.append(node)
                continue
            if id(node) in visited:
                continue
            visited.add(id(node))
            stack.append((node, True))
            if node._ctx is not None:
                for parent in node._ctx.parents:
                    if id(parent) not in visited:
                        stack.append((parent, False))

        self.grad = seed
        for node in reversed(order):
            if node._ctx is None or node.grad is None:
                continue
            input_grads = node._ctx.backward(node.grad)
            for parent, g in zip(node._ctx.parents, input_grads, strict=True):
                if g is None or not parent.requires_grad:
                    continue
                parent.grad = g if parent.grad is None else parent.grad + g

    def zero_grad(self) -> None:
        """Reset the accumulated gradient to ``None``."""
        self.grad = None

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


def _coerce(value: Tensor | ArrayLike) -> Tensor:
    """Wrap non-Tensor operands so mixed expressions like ``t + 2.0`` work."""
    return value if isinstance(value, Tensor) else Tensor(value)
