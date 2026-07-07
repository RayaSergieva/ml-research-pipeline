"""Tests for the loss landscape probe.

The oracle throughout is the quadratic loss L(theta) = 0.5 theta^T A theta
whose Hessian is exactly A, so every estimate the probe produces can be
compared against closed-form linear algebra.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from spectra import Tensor
from spectra.analysis.landscape import (
    flatten_parameters,
    gradient_vector,
    hessian_vector_product,
    loss_plane,
    set_parameters,
    top_hessian_eigenvalue,
)
from spectra.nn import Linear, Module

RNG = np.random.default_rng(21)


class VectorModel(Module):
    """A model that is just one parameter vector."""

    def __init__(self, theta0: np.ndarray) -> None:
        super().__init__()
        self.theta = Tensor(theta0.copy(), requires_grad=True)


def quadratic(model: VectorModel, a: np.ndarray) -> Callable[[], Tensor]:
    """Loss closure whose gradient field is exactly A theta.

    Inside one backward pass the factor ``Tensor(a @ model.theta.data)`` is a
    constant, so the closure's gradient is A theta at every point and its
    (finite-difference) Hessian is exactly A - a clean oracle for the probe.
    The value itself is theta^T A theta, twice the textbook quadratic, which
    none of the assertions below depend on.
    """

    def loss() -> Tensor:
        return (model.theta * Tensor(a @ model.theta.data)).sum()

    return loss


def random_spd(n: int) -> np.ndarray:
    m = RNG.standard_normal((n, n))
    return np.asarray(m @ m.T + n * np.eye(n))


# ---------------------------------------------------------------------------
# Flatten / set
# ---------------------------------------------------------------------------


def test_flatten_and_set_roundtrip() -> None:
    model = Linear(3, 2, rng=np.random.default_rng(0))
    flat = flatten_parameters(model)
    assert flat.shape == (3 * 2 + 2,)
    modified = flat + 1.0
    set_parameters(model, modified)
    np.testing.assert_allclose(flatten_parameters(model), modified)


def test_set_parameters_wrong_size_raises() -> None:
    model = Linear(3, 2, rng=np.random.default_rng(0))
    with pytest.raises(ValueError, match="entries"):
        set_parameters(model, np.zeros(5))


def test_gradient_vector_on_quadratic() -> None:
    """grad of 0.5 theta^T A theta is A theta (A symmetric)."""
    a = random_spd(4)
    theta0 = RNG.standard_normal(4)
    model = VectorModel(theta0)
    g = gradient_vector(model, quadratic(model, a))
    np.testing.assert_allclose(g, a @ theta0, atol=1e-10)


def test_gradient_vector_restores_zero_grads() -> None:
    model = VectorModel(RNG.standard_normal(3))
    gradient_vector(model, quadratic(model, random_spd(3)))
    assert model.theta.grad is None


# ---------------------------------------------------------------------------
# HVP and power iteration
# ---------------------------------------------------------------------------


def test_hvp_matches_exact_hessian() -> None:
    a = random_spd(5)
    model = VectorModel(RNG.standard_normal(5))
    v = RNG.standard_normal(5)
    hv = hessian_vector_product(model, quadratic(model, a), v)
    np.testing.assert_allclose(hv, a @ v, rtol=1e-5, atol=1e-6)


def test_hvp_restores_parameters() -> None:
    a = random_spd(4)
    theta0 = RNG.standard_normal(4)
    model = VectorModel(theta0)
    hessian_vector_product(model, quadratic(model, a), RNG.standard_normal(4))
    np.testing.assert_allclose(model.theta.data, theta0)


def test_hvp_rejects_wrong_shape() -> None:
    model = VectorModel(RNG.standard_normal(4))
    with pytest.raises(ValueError, match="shape"):
        hessian_vector_product(model, quadratic(model, random_spd(4)), np.zeros(3))


def test_top_eigenvalue_matches_exact() -> None:
    a = random_spd(6)
    model = VectorModel(RNG.standard_normal(6))
    estimate, v = top_hessian_eigenvalue(model, quadratic(model, a), iterations=60)
    exact = float(np.linalg.eigvalsh(a)[-1])
    assert estimate == pytest.approx(exact, rel=1e-3)
    # The returned vector is the corresponding eigenvector.
    np.testing.assert_allclose(a @ v, exact * v, rtol=1e-2, atol=1e-2)


def test_top_eigenvalue_on_real_layer_loss() -> None:
    """Sharpness of an MSE loss through a Linear layer is finite and positive."""
    model = Linear(4, 3, rng=np.random.default_rng(1))
    x = Tensor(RNG.standard_normal((16, 4)))
    y = Tensor(RNG.standard_normal((16, 3)))

    def loss() -> Tensor:
        diff = model(x) - y
        return (diff * diff).mean()

    sharpness, _ = top_hessian_eigenvalue(model, loss, iterations=40)
    assert np.isfinite(sharpness)
    assert sharpness > 0  # MSE through a linear map is convex


# ---------------------------------------------------------------------------
# Loss plane
# ---------------------------------------------------------------------------


def test_loss_plane_center_is_current_loss() -> None:
    a = random_spd(4)
    model = VectorModel(RNG.standard_normal(4))
    loss_fn = quadratic(model, a)
    alphas = np.array([-1.0, 0.0, 1.0])
    grid = loss_plane(model, loss_fn, alphas, alphas)
    assert grid.shape == (3, 3)
    np.testing.assert_allclose(grid[1, 1], float(loss_fn().data))


def test_loss_plane_restores_parameters() -> None:
    theta0 = RNG.standard_normal(4)
    model = VectorModel(theta0)
    loss_plane(model, quadratic(model, random_spd(4)), np.linspace(-1, 1, 3), np.linspace(-1, 1, 3))
    np.testing.assert_allclose(model.theta.data, theta0)


def test_loss_plane_of_quadratic_is_convex() -> None:
    """The restriction of a convex quadratic to any plane is convex, so every
    grid midpoint lies below the average of its neighbours."""
    a = random_spd(5)
    model = VectorModel(np.ones(5))
    alphas = np.array([-0.5, 0.0, 0.5])
    grid = loss_plane(model, quadratic(model, a), alphas, alphas)
    assert grid.shape == (3, 3)
    # Midpoint convexity along rows, columns, and both diagonals.
    for i in range(3):
        assert grid[i, 0] + grid[i, 2] >= 2 * grid[i, 1] - 1e-10
        assert grid[0, i] + grid[2, i] >= 2 * grid[1, i] - 1e-10
    assert grid[0, 0] + grid[2, 2] >= 2 * grid[1, 1] - 1e-10
    assert grid[0, 2] + grid[2, 0] >= 2 * grid[1, 1] - 1e-10
