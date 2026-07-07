"""Tests for the loss functions and optimizers."""

from __future__ import annotations

import numpy as np
import pytest

from spectra import Tensor
from spectra.losses import softmax_cross_entropy
from spectra.nn import Linear, Module
from spectra.optim import SGD, Adam

RNG = np.random.default_rng(3)


# ---------------------------------------------------------------------------
# Softmax cross-entropy
# ---------------------------------------------------------------------------


def test_cross_entropy_uniform_logits() -> None:
    """Equal logits give loss log(C) for C classes, the entropy of a guess."""
    logits = Tensor(np.zeros((4, 10)))
    loss = softmax_cross_entropy(logits, np.arange(4))
    np.testing.assert_allclose(float(loss.data), np.log(10), rtol=1e-12)


def test_cross_entropy_confident_correct_prediction() -> None:
    logits_data = np.full((2, 3), -100.0)
    logits_data[0, 1] = 100.0
    logits_data[1, 2] = 100.0
    loss = softmax_cross_entropy(Tensor(logits_data), np.array([1, 2]))
    assert float(loss.data) < 1e-6


def test_cross_entropy_is_numerically_stable_for_large_logits() -> None:
    logits = Tensor(np.array([[1e4, -1e4], [-1e4, 1e4]]))
    loss = softmax_cross_entropy(logits, np.array([0, 1]))
    assert np.isfinite(float(loss.data))


def test_cross_entropy_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError, match="2-D"):
        softmax_cross_entropy(Tensor(np.zeros(3)), np.array([0]))
    with pytest.raises(ValueError, match="targets"):
        softmax_cross_entropy(Tensor(np.zeros((2, 3))), np.array([0, 1, 2]))


def test_cross_entropy_gradient_matches_finite_differences() -> None:
    targets = np.array([0, 2, 1])
    logits_data = RNG.standard_normal((3, 4))

    x = Tensor(logits_data.copy(), requires_grad=True)
    softmax_cross_entropy(x, targets).backward()
    assert x.grad is not None

    eps = 1e-6
    numerical = np.zeros_like(logits_data)
    for i in range(3):
        for j in range(4):
            bumped = logits_data.copy()
            bumped[i, j] += eps
            plus = float(softmax_cross_entropy(Tensor(bumped), targets).data)
            bumped[i, j] -= 2 * eps
            minus = float(softmax_cross_entropy(Tensor(bumped), targets).data)
            numerical[i, j] = (plus - minus) / (2 * eps)

    np.testing.assert_allclose(x.grad, numerical, atol=1e-6)


def test_cross_entropy_gradient_rows_sum_to_zero() -> None:
    """Each row of the gradient is softmax - onehot, which sums to zero."""
    x = Tensor(RNG.standard_normal((5, 3)), requires_grad=True)
    softmax_cross_entropy(x, np.array([0, 1, 2, 0, 1])).backward()
    assert x.grad is not None
    np.testing.assert_allclose(x.grad.sum(axis=1), np.zeros(5), atol=1e-12)


# ---------------------------------------------------------------------------
# Optimizers
# ---------------------------------------------------------------------------


def quadratic_loss(p: Tensor) -> Tensor:
    """L(p) = mean(p^2), minimized at p = 0."""
    return (p * p).mean()


def test_optimizer_rejects_empty_parameters() -> None:
    with pytest.raises(ValueError, match="empty"):
        SGD([], lr=0.1)


def test_sgd_rejects_bad_hyperparameters() -> None:
    p = Tensor([1.0], requires_grad=True)
    with pytest.raises(ValueError, match="learning rate"):
        SGD([p], lr=0.0)
    with pytest.raises(ValueError, match="momentum"):
        SGD([p], lr=0.1, momentum=1.0)


def test_adam_rejects_bad_hyperparameters() -> None:
    p = Tensor([1.0], requires_grad=True)
    with pytest.raises(ValueError, match="learning rate"):
        Adam([p], lr=-1.0)
    with pytest.raises(ValueError, match="betas"):
        Adam([p], betas=(1.0, 0.999))


def test_sgd_single_step_matches_hand_computation() -> None:
    p = Tensor([2.0], requires_grad=True)
    quadratic_loss(p).backward()  # dL/dp = 2p/1 = 4
    SGD([p], lr=0.25).step()
    np.testing.assert_allclose(p.data, [1.0])


def test_sgd_skips_parameters_without_grad() -> None:
    p = Tensor([2.0], requires_grad=True)
    SGD([p], lr=0.1).step()  # no backward has run
    np.testing.assert_allclose(p.data, [2.0])


@pytest.mark.parametrize("momentum", [0.0, 0.9])
def test_sgd_converges_on_quadratic(momentum: float) -> None:
    p = Tensor(RNG.standard_normal(5), requires_grad=True)
    opt = SGD([p], lr=0.1, momentum=momentum)
    for _ in range(200):
        opt.zero_grad()
        quadratic_loss(p).backward()
        opt.step()
    np.testing.assert_allclose(p.data, np.zeros(5), atol=1e-3)


def test_adam_converges_on_quadratic() -> None:
    p = Tensor(RNG.standard_normal(5), requires_grad=True)
    opt = Adam([p], lr=0.05)
    for _ in range(400):
        opt.zero_grad()
        quadratic_loss(p).backward()
        opt.step()
    np.testing.assert_allclose(p.data, np.zeros(5), atol=1e-3)


def test_adam_bias_correction_first_step() -> None:
    """After one step from m=v=0 the bias-corrected update equals lr * sign(g)
    up to the eps term, independent of the gradient's magnitude."""
    p = Tensor([10.0], requires_grad=True)
    (p * 3.0).sum().backward()  # constant gradient 3
    Adam([p], lr=0.1).step()
    np.testing.assert_allclose(p.data, [10.0 - 0.1], rtol=1e-6)


def test_optimizer_zero_grad() -> None:
    p = Tensor([1.0], requires_grad=True)
    quadratic_loss(p).backward()
    opt = SGD([p], lr=0.1)
    opt.zero_grad()
    assert p.grad is None


# ---------------------------------------------------------------------------
# End to end - a small classifier learns
# ---------------------------------------------------------------------------


class TwoLayerNet(Module):
    def __init__(self, rng: np.random.Generator) -> None:
        super().__init__()
        self.fc1 = Linear(2, 16, rng=rng)
        self.fc2 = Linear(16, 2, rng=rng)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.fc1(x).relu())


@pytest.mark.parametrize(
    "make_optimizer",
    [
        lambda params: SGD(params, lr=0.5, momentum=0.9),
        lambda params: Adam(params, lr=0.01),
    ],
    ids=["sgd", "adam"],
)
def test_classifier_learns_linearly_separable_data(make_optimizer) -> None:  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(0)
    n = 200
    x_data = rng.standard_normal((n, 2))
    y_data = (x_data[:, 0] + x_data[:, 1] > 0).astype(np.int64)

    model = TwoLayerNet(rng)
    opt = make_optimizer(model.parameters())

    x = Tensor(x_data)
    first_loss = float(softmax_cross_entropy(model(x), y_data).data)
    for _ in range(100):
        opt.zero_grad()
        softmax_cross_entropy(model(x), y_data).backward()
        opt.step()
    final_loss = float(softmax_cross_entropy(model(x), y_data).data)

    predictions = model(x).data.argmax(axis=1)
    accuracy = (predictions == y_data).mean()
    assert final_loss < first_loss / 5
    assert accuracy > 0.95


def test_adam_skips_parameters_without_grad() -> None:
    p = Tensor([2.0], requires_grad=True)
    Adam([p], lr=0.1).step()
    np.testing.assert_allclose(p.data, [2.0])
