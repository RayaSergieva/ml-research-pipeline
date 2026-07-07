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


class Reshape(Function):
    """Shape change without data movement. Linear with identity Jacobian in
    the flattened view, so the backward pass is the inverse reshape."""

    def forward(self, a: Array, *, shape: tuple[int, ...]) -> Array:  # type: ignore[override]
        self._original_shape = a.shape
        return np.asarray(a.reshape(shape))

    def backward(self, grad_output: Array) -> tuple[Array | None, ...]:
        return (grad_output.reshape(self._original_shape),)


class Transpose(Function):
    """Axis permutation. The adjoint of a permutation is its inverse."""

    def forward(self, a: Array, *, axes: tuple[int, ...] | None = None) -> Array:  # type: ignore[override]
        if axes is None:
            axes = tuple(range(a.ndim))[::-1]
        self._axes = axes
        return np.asarray(np.transpose(a, axes))

    def backward(self, grad_output: Array) -> tuple[Array | None, ...]:
        inverse = np.argsort(self._axes)
        return (np.transpose(grad_output, inverse),)


class BatchedMatMul(Function):
    """Matrix product on the last two axes, broadcasting over batch axes.

    For C = A B with G = dL/dC the last-two-axes chain rule is unchanged,
    dL/dA = G B^T and dL/dB = A^T G with the transpose taken on the last two
    axes; batch axes that were broadcast in the forward pass are summed in
    the backward pass, the same adjoint-of-broadcast rule as elementwise ops.
    """

    def forward(self, a: Array, b: Array) -> Array:  # type: ignore[override]
        if a.ndim < 2 or b.ndim < 2:
            msg = f"batched matmul needs operands of rank >= 2, got {a.ndim}-D and {b.ndim}-D"
            raise ValueError(msg)
        self.save_for_backward(a, b)
        return np.asarray(a @ b)

    def backward(self, grad_output: Array) -> tuple[Array | None, ...]:
        a, b = self.saved
        grad_a = grad_output @ np.swapaxes(b, -1, -2)
        grad_b = np.swapaxes(a, -1, -2) @ grad_output
        return (_unbroadcast(grad_a, a.shape), _unbroadcast(grad_b, b.shape))


class Softmax(Function):
    """Softmax along one axis, computed with the max-shift for stability.

    With s = softmax(z) along the axis, the Jacobian action is
    ds = s * (g - <g, s>), the projection of g onto the tangent of the
    probability simplex scaled by s - derived in the notebook from
    d s_i / d z_j = s_i (delta_ij - s_j).
    """

    def forward(self, a: Array, *, axis: int = -1) -> Array:  # type: ignore[override]
        shifted = a - a.max(axis=axis, keepdims=True)
        exp = np.exp(shifted)
        out = exp / exp.sum(axis=axis, keepdims=True)
        self._axis = axis
        self.save_for_backward(out)
        return np.asarray(out)

    def backward(self, grad_output: Array) -> tuple[Array | None, ...]:
        (s,) = self.saved
        inner = (grad_output * s).sum(axis=self._axis, keepdims=True)
        return (s * (grad_output - inner),)


class GELU(Function):
    """Gaussian error linear unit, tanh approximation (Hendrycks &
    Gimpel, 2016).

    gelu(x) ~= 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3))). The
    derivative follows by the product and chain rules and is spelled out in
    the notebook; both are validated against finite differences.
    """

    _C = float(np.sqrt(2.0 / np.pi))
    _K = 0.044715

    def forward(self, a: Array) -> Array:  # type: ignore[override]
        self.save_for_backward(a)
        inner = self._C * (a + self._K * a**3)
        return np.asarray(0.5 * a * (1.0 + np.tanh(inner)))

    def backward(self, grad_output: Array) -> tuple[Array | None, ...]:
        (a,) = self.saved
        inner = self._C * (a + self._K * a**3)
        t = np.tanh(inner)
        d_inner = self._C * (1.0 + 3.0 * self._K * a**2)
        grad = 0.5 * (1.0 + t) + 0.5 * a * (1.0 - t**2) * d_inner
        return (grad_output * grad,)


class EmbeddingLookup(Function):
    """Row gather from an embedding table, out = W[indices].

    The lookup is linear in W - it selects rows - so the adjoint scatters the
    output gradient back onto the selected rows, summing where an index
    appears more than once (np.add.at performs the unbuffered accumulation).
    """

    def forward(self, weight: Array, *, indices: Array) -> Array:  # type: ignore[override]
        if weight.ndim != 2:
            msg = f"embedding weight must be 2-D, got {weight.ndim}-D"
            raise ValueError(msg)
        self._indices = indices
        self._weight_shape = weight.shape
        return np.asarray(weight[indices])

    def backward(self, grad_output: Array) -> tuple[Array | None, ...]:
        grad = np.zeros(self._weight_shape, dtype=grad_output.dtype)
        np.add.at(grad, self._indices, grad_output)
        return (grad,)
