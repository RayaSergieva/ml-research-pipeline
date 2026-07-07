"""Tests for the autograd engine.

The central tool here is finite-difference gradient checking: for a scalar
function L(x), each analytic gradient entry is compared against the central
difference (L(x + eps e_i) - L(x - eps e_i)) / (2 eps), which approximates the
true derivative with O(eps^2) error. Agreement across random inputs is strong
evidence that the backward rules implement the correct vector-Jacobian
products.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from spectra import Tensor

RNG = np.random.default_rng(0)
EPS = 1e-6
ATOL = 1e-4


def numerical_grad(
    f: Callable[[np.ndarray], float],
    x: np.ndarray,
    eps: float = EPS,
) -> np.ndarray:
    """Central-difference approximation of dL/dx for scalar-valued f."""
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        original = x[idx]
        x[idx] = original + eps
        plus = f(x)
        x[idx] = original - eps
        minus = f(x)
        x[idx] = original
        grad[idx] = (plus - minus) / (2 * eps)
        it.iternext()
    return grad


def check_gradient(
    build_loss: Callable[[Tensor], Tensor],
    x_data: np.ndarray,
) -> None:
    """Assert analytic and numerical gradients agree for a scalar loss."""
    x = Tensor(x_data.copy(), requires_grad=True)
    loss = build_loss(x)
    assert loss.size == 1, "gradient checking requires a scalar loss"
    loss.backward()
    assert x.grad is not None

    def f(arr: np.ndarray) -> float:
        return float(build_loss(Tensor(arr.copy())).data)

    expected = numerical_grad(f, x_data.copy())
    np.testing.assert_allclose(x.grad, expected, atol=ATOL)


# ---------------------------------------------------------------------------
# Graph mechanics
# ---------------------------------------------------------------------------


def test_backward_requires_grad_flag() -> None:
    t = Tensor([1.0, 2.0])
    with pytest.raises(RuntimeError, match="does not require grad"):
        t.backward()


def test_backward_on_nonscalar_requires_explicit_grad() -> None:
    t = Tensor([1.0, 2.0], requires_grad=True)
    with pytest.raises(RuntimeError, match="non-scalar"):
        t.backward()


def test_backward_grad_shape_mismatch_raises() -> None:
    t = Tensor([1.0, 2.0], requires_grad=True)
    with pytest.raises(ValueError, match="does not match"):
        t.backward(grad=np.ones((3,)))


def test_result_requires_grad_iff_any_input_does() -> None:
    a = Tensor([1.0], requires_grad=True)
    b = Tensor([2.0])
    assert (a + b).requires_grad is True
    assert (b + b).requires_grad is False


def test_no_graph_recorded_without_requires_grad() -> None:
    a = Tensor([1.0])
    out = a + a
    assert out._ctx is None


def test_gradients_accumulate_across_calls() -> None:
    a = Tensor([2.0], requires_grad=True)
    (a * a).sum().backward()
    (a * a).sum().backward()
    assert a.grad is not None
    np.testing.assert_allclose(a.grad, [8.0])  # 2 * (2a) at a=2


def test_zero_grad_resets() -> None:
    a = Tensor([2.0], requires_grad=True)
    (a * a).sum().backward()
    a.zero_grad()
    assert a.grad is None


def test_fan_out_gradients_sum() -> None:
    """A tensor used twice receives the sum of both path gradients."""
    a = Tensor([3.0], requires_grad=True)
    out = (a * a) + a  # dL/da = 2a + 1 = 7 at a=3
    out.sum().backward()
    assert a.grad is not None
    np.testing.assert_allclose(a.grad, [7.0])


def test_deep_chain_does_not_overflow() -> None:
    """The iterative topological sort must handle graphs deeper than the
    Python recursion limit."""
    a = Tensor([1.0], requires_grad=True)
    out = a
    for _ in range(5000):
        out = out + 0.0
    out.sum().backward()
    assert a.grad is not None
    np.testing.assert_allclose(a.grad, [1.0])


# ---------------------------------------------------------------------------
# Gradient checks per operation
# ---------------------------------------------------------------------------


def test_grad_add() -> None:
    check_gradient(lambda x: (x + x).sum(), RNG.standard_normal((3, 4)))


def test_grad_add_broadcast() -> None:
    b = RNG.standard_normal((4,))
    check_gradient(lambda x: (x + Tensor(b)).sum(), RNG.standard_normal((3, 4)))


def test_grad_add_broadcast_other_operand() -> None:
    """The broadcast (smaller) operand must receive a summed gradient."""
    big = RNG.standard_normal((3, 4))
    check_gradient(lambda x: (Tensor(big) + x).sum(), RNG.standard_normal((4,)))


def test_grad_mul() -> None:
    b = RNG.standard_normal((3, 4))
    check_gradient(lambda x: (x * Tensor(b)).sum(), RNG.standard_normal((3, 4)))


def test_grad_mul_broadcast() -> None:
    b = RNG.standard_normal((4,))
    check_gradient(lambda x: (x * Tensor(b)).sum(), RNG.standard_normal((3, 4)))


def test_grad_neg_and_sub() -> None:
    b = RNG.standard_normal((3,))
    check_gradient(lambda x: (x - Tensor(b)).sum(), RNG.standard_normal((3,)))
    check_gradient(lambda x: (Tensor(b) - x).sum(), RNG.standard_normal((3,)))


def test_grad_matmul_left() -> None:
    b = RNG.standard_normal((4, 2))
    check_gradient(lambda x: (x @ Tensor(b)).sum(), RNG.standard_normal((3, 4)))


def test_grad_matmul_right() -> None:
    a = RNG.standard_normal((3, 4))
    check_gradient(lambda x: (Tensor(a) @ x).sum(), RNG.standard_normal((4, 2)))


def test_matmul_rejects_non_2d() -> None:
    a = Tensor(RNG.standard_normal((3,)), requires_grad=True)
    b = Tensor(RNG.standard_normal((3, 2)))
    with pytest.raises(ValueError, match="2-D"):
        _ = a @ b


def test_grad_sum_axis() -> None:
    check_gradient(lambda x: x.sum(axis=0).sum(), RNG.standard_normal((3, 4)))
    check_gradient(
        lambda x: x.sum(axis=1, keepdims=True).sum(),
        RNG.standard_normal((3, 4)),
    )


def test_grad_mean() -> None:
    check_gradient(lambda x: x.mean(), RNG.standard_normal((3, 4)))
    check_gradient(lambda x: x.mean(axis=1).sum(), RNG.standard_normal((3, 4)))


def test_grad_relu() -> None:
    # Keep inputs away from the kink at 0 where the derivative is undefined.
    x = RNG.standard_normal((3, 4))
    x[np.abs(x) < 0.1] = 0.5
    check_gradient(lambda t: t.relu().sum(), x)


def test_grad_exp() -> None:
    check_gradient(lambda x: x.exp().sum(), RNG.standard_normal((3, 4)))


def test_grad_log() -> None:
    check_gradient(lambda x: x.log().sum(), RNG.uniform(0.5, 2.0, (3, 4)))


def test_grad_pow() -> None:
    check_gradient(lambda x: (x**3.0).sum(), RNG.uniform(0.5, 2.0, (3, 4)))


def test_grad_scalar_mixing() -> None:
    check_gradient(lambda x: (2.0 * x + 1.0).sum(), RNG.standard_normal((3,)))


# ---------------------------------------------------------------------------
# Composite expressions
# ---------------------------------------------------------------------------


def test_grad_mlp_like_composition() -> None:
    """One hidden-layer network: mean(relu(x W1) W2), checked end to end."""
    W1 = RNG.standard_normal((4, 5))
    W2 = RNG.standard_normal((5, 2))

    def loss(x: Tensor) -> Tensor:
        h = (x @ Tensor(W1)).relu()
        return (h @ Tensor(W2)).mean()

    x = RNG.standard_normal((3, 4))
    x[np.abs(x) < 0.1] = 0.5
    check_gradient(loss, x)


def test_grad_weights_of_mlp() -> None:
    """Gradient with respect to a weight matrix, the training-relevant case."""
    x_data = RNG.standard_normal((3, 4))

    def loss(w: Tensor) -> Tensor:
        return ((Tensor(x_data) @ w).relu()).mean()

    w_data = RNG.standard_normal((4, 5))
    check_gradient(loss, w_data)


def test_grad_log_softmax_style_expression() -> None:
    """A numerically simple log-sum-exp composition."""

    def loss(x: Tensor) -> Tensor:
        return (x.exp().sum(axis=1, keepdims=True).log()).sum()

    check_gradient(loss, RNG.standard_normal((3, 4)))


def test_grad_broadcast_stretched_axis() -> None:
    """A size-1 axis stretched by broadcasting must be summed in backward."""
    b = RNG.standard_normal((3, 4))
    check_gradient(lambda x: (x + Tensor(b)).sum(), RNG.standard_normal((3, 1)))


def test_reflected_operators() -> None:
    a = Tensor([1.0, 2.0], requires_grad=True)
    out = (3.0 + a).sum() + (5.0 - a).sum()
    out.backward()
    assert a.grad is not None
    np.testing.assert_allclose(a.grad, [0.0, 0.0])


def test_grad_mean_keepdims() -> None:
    check_gradient(
        lambda x: x.mean(axis=0, keepdims=True).sum(),
        RNG.standard_normal((3, 4)),
    )


# ---------------------------------------------------------------------------
# Shape ops and transformer-support ops
# ---------------------------------------------------------------------------


def test_grad_reshape() -> None:
    check_gradient(lambda x: (x.reshape(6, 2) * 2.0).sum(), RNG.standard_normal((3, 4)))


def test_reshape_roundtrip_gradient_is_identity() -> None:
    x = Tensor(RNG.standard_normal((3, 4)), requires_grad=True)
    x.reshape(12).reshape(3, 4).sum().backward()
    assert x.grad is not None
    np.testing.assert_allclose(x.grad, np.ones((3, 4)))


def test_grad_transpose() -> None:
    b = RNG.standard_normal((4, 3))
    check_gradient(lambda x: (x.transpose() * Tensor(b)).sum(), RNG.standard_normal((3, 4)))


def test_grad_transpose_with_axes() -> None:
    check_gradient(
        lambda x: (x.transpose(1, 0, 2) ** 2.0).sum(),
        RNG.uniform(0.5, 1.5, (2, 3, 4)),
    )


def test_grad_batched_matmul_left() -> None:
    b = RNG.standard_normal((2, 4, 3))
    check_gradient(lambda x: x.bmm(Tensor(b)).sum(), RNG.standard_normal((2, 5, 4)))


def test_grad_batched_matmul_right() -> None:
    a = RNG.standard_normal((2, 5, 4))
    check_gradient(lambda x: Tensor(a).bmm(x).sum(), RNG.standard_normal((2, 4, 3)))


def test_grad_batched_matmul_broadcast_batch() -> None:
    """A single matrix broadcast across the batch must receive summed grads."""
    a = RNG.standard_normal((3, 5, 4))
    check_gradient(lambda x: Tensor(a).bmm(x).sum(), RNG.standard_normal((4, 2)))


def test_batched_matmul_rejects_vectors() -> None:
    a = Tensor(RNG.standard_normal(4), requires_grad=True)
    with pytest.raises(ValueError, match="rank"):
        a.bmm(Tensor(RNG.standard_normal((4, 2))))


def test_grad_softmax() -> None:
    weights = Tensor(RNG.standard_normal((3, 5)))
    check_gradient(lambda x: (x.softmax() * weights).sum(), RNG.standard_normal((3, 5)))


def test_grad_softmax_other_axis() -> None:
    weights = Tensor(RNG.standard_normal((3, 5)))
    check_gradient(lambda x: (x.softmax(axis=0) * weights).sum(), RNG.standard_normal((3, 5)))


def test_softmax_rows_sum_to_one() -> None:
    s = Tensor(RNG.standard_normal((4, 7))).softmax()
    np.testing.assert_allclose(s.data.sum(axis=-1), np.ones(4))


def test_softmax_is_stable_for_large_inputs() -> None:
    s = Tensor(np.array([[1e4, 0.0, -1e4]])).softmax()
    assert np.all(np.isfinite(s.data))


def test_grad_gelu() -> None:
    check_gradient(lambda x: x.gelu().sum(), RNG.standard_normal((3, 4)))


def test_gelu_matches_reference_values() -> None:
    """gelu(0) = 0 and gelu is close to identity for large positive x."""
    out = Tensor(np.array([0.0, 6.0, -6.0])).gelu().data
    assert out[0] == 0.0
    np.testing.assert_allclose(out[1], 6.0, atol=1e-3)
    np.testing.assert_allclose(out[2], 0.0, atol=1e-3)
