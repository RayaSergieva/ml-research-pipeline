"""Differentiable operations on tensors.

Each operation is a :class:`~spectra.autograd.Function` subclass implementing
the forward computation and its vector-Jacobian product (the backward rule).
The derivations for every rule are standard matrix calculus; the notebook
accompanying this project derives each one from first principles.

Broadcasting note: NumPy broadcasting expands operand shapes implicitly during
the forward pass. The backward pass must reverse this, because the gradient
with respect to an operand must have the operand's shape. :func:`_unbroadcast`
sums the gradient over the dimensions that broadcasting introduced or
stretched, which is the adjoint of the broadcast operation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from spectra.autograd import Function

Array = NDArray[Any]


def _unbroadcast(grad: Array, shape: tuple[int, ...]) -> Array:
    """Sum ``grad`` over broadcast dimensions so it matches ``shape``.

    Broadcasting a tensor is a linear map; its adjoint is summation over the
    replicated axes. Leading axes that broadcasting prepended are summed away,
    and axes of size one that were stretched are summed with ``keepdims``.
    """
    extra = grad.ndim - len(shape)
    for _ in range(extra):
        grad = grad.sum(axis=0)
    for axis, dim in enumerate(shape):
        if dim == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


class Add(Function):
    """Elementwise addition. d(a+b)/da = 1, d(a+b)/db = 1."""

    def forward(self, a: Array, b: Array) -> Array:  # type: ignore[override]
        self.save_for_backward(np.asarray(a.shape), np.asarray(b.shape))
        return a + b

    def backward(self, grad_output: Array) -> tuple[Array | None, ...]:
        a_shape, b_shape = self.saved
        return (
            _unbroadcast(grad_output, tuple(int(d) for d in a_shape)),
            _unbroadcast(grad_output, tuple(int(d) for d in b_shape)),
        )


class Mul(Function):
    """Elementwise product. d(ab)/da = b, d(ab)/db = a."""

    def forward(self, a: Array, b: Array) -> Array:  # type: ignore[override]
        self.save_for_backward(a, b)
        return a * b

    def backward(self, grad_output: Array) -> tuple[Array | None, ...]:
        a, b = self.saved
        return (
            _unbroadcast(grad_output * b, a.shape),
            _unbroadcast(grad_output * a, b.shape),
        )


class Neg(Function):
    """Negation. d(-a)/da = -1."""

    def forward(self, a: Array) -> Array:  # type: ignore[override]
        return -a

    def backward(self, grad_output: Array) -> tuple[Array | None, ...]:
        return (-grad_output,)


class MatMul(Function):
    """Matrix product of two 2-D arrays.

    For C = A B with upstream gradient G = dL/dC, the chain rule in matrix
    form gives dL/dA = G B^T and dL/dB = A^T G. Higher-rank inputs are
    rejected for now; batched matmul is planned alongside the attention
    layers.
    """

    def forward(self, a: Array, b: Array) -> Array:  # type: ignore[override]
        if a.ndim != 2 or b.ndim != 2:
            msg = f"matmul currently supports 2-D operands only, got {a.ndim}-D and {b.ndim}-D"
            raise ValueError(msg)
        self.save_for_backward(a, b)
        return a @ b

    def backward(self, grad_output: Array) -> tuple[Array | None, ...]:
        a, b = self.saved
        return (grad_output @ b.T, a.T @ grad_output)


class Sum(Function):
    """Summation over all elements or along an axis.

    Summation is a linear map whose adjoint is broadcasting: the gradient of
    the output is replicated back across the summed dimensions.
    """

    def forward(self, a: Array, *, axis: int | None = None, keepdims: bool = False) -> Array:  # type: ignore[override]
        self.save_for_backward(np.asarray(a.shape))
        self._axis = axis
        self._keepdims = keepdims
        if keepdims:
            return np.asarray(np.sum(a, axis=axis, keepdims=True))
        return np.asarray(np.sum(a, axis=axis))

    def backward(self, grad_output: Array) -> tuple[Array | None, ...]:
        (shape_arr,) = self.saved
        shape = tuple(int(d) for d in shape_arr)
        grad = grad_output
        if self._axis is not None and not self._keepdims:
            grad = np.expand_dims(grad, self._axis)
        return (np.broadcast_to(grad, shape).copy(),)


class Mean(Function):
    """Arithmetic mean over all elements or along an axis.

    The mean is the sum scaled by 1/n, so its backward rule is the sum's
    backward rule scaled by the same factor.
    """

    def forward(self, a: Array, *, axis: int | None = None, keepdims: bool = False) -> Array:  # type: ignore[override]
        self.save_for_backward(np.asarray(a.shape))
        self._axis = axis
        self._keepdims = keepdims
        self._count = a.size if axis is None else a.shape[axis]
        if keepdims:
            return np.asarray(np.mean(a, axis=axis, keepdims=True))
        return np.asarray(np.mean(a, axis=axis))

    def backward(self, grad_output: Array) -> tuple[Array | None, ...]:
        (shape_arr,) = self.saved
        shape = tuple(int(d) for d in shape_arr)
        grad = grad_output
        if self._axis is not None and not self._keepdims:
            grad = np.expand_dims(grad, self._axis)
        return (np.broadcast_to(grad, shape).copy() / self._count,)


class ReLU(Function):
    """Rectified linear unit, max(a, 0).

    The derivative is the indicator of the positive orthant. The function is
    not differentiable at 0; the subgradient 0 is used there, the standard
    convention in deep learning frameworks.
    """

    def forward(self, a: Array) -> Array:  # type: ignore[override]
        mask = (a > 0).astype(a.dtype)
        self.save_for_backward(mask)
        return np.asarray(a * mask)

    def backward(self, grad_output: Array) -> tuple[Array | None, ...]:
        (mask,) = self.saved
        return (grad_output * mask,)


class Exp(Function):
    """Elementwise exponential. d(e^a)/da = e^a."""

    def forward(self, a: Array) -> Array:  # type: ignore[override]
        out = np.exp(a)
        self.save_for_backward(out)
        return np.asarray(out)

    def backward(self, grad_output: Array) -> tuple[Array | None, ...]:
        (out,) = self.saved
        return (grad_output * out,)


class Log(Function):
    """Elementwise natural logarithm. d(ln a)/da = 1/a."""

    def forward(self, a: Array) -> Array:  # type: ignore[override]
        self.save_for_backward(a)
        return np.asarray(np.log(a))

    def backward(self, grad_output: Array) -> tuple[Array | None, ...]:
        (a,) = self.saved
        return (grad_output / a,)


class Pow(Function):
    """Elementwise power with a constant exponent. d(a^p)/da = p a^(p-1)."""

    def forward(self, a: Array, *, exponent: float) -> Array:  # type: ignore[override]
        self.save_for_backward(a)
        self._exponent = exponent
        return np.asarray(a**exponent)

    def backward(self, grad_output: Array) -> tuple[Array | None, ...]:
        (a,) = self.saved
        p = self._exponent
        return (grad_output * p * a ** (p - 1),)
