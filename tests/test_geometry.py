"""Tests for the representation geometry instrument.

Each measure is validated on synthetic clouds whose geometry is known by
construction - exact-rank Gaussians for the participation ratio, manifolds of
known dimension for TwoNN, and blobs at controlled separation for the class
separability score.
"""

from __future__ import annotations

import numpy as np
import pytest

from spectra.analysis.geometry import (
    class_separability,
    participation_ratio,
    twonn_intrinsic_dimension,
)

RNG = np.random.default_rng(33)


# ---------------------------------------------------------------------------
# Participation ratio
# ---------------------------------------------------------------------------


def test_pr_of_isotropic_gaussian_near_full_dimension() -> None:
    x = RNG.standard_normal((5000, 10))
    assert participation_ratio(x) == pytest.approx(10.0, rel=0.05)


def test_pr_of_rank_k_cloud_is_k() -> None:
    k, ambient = 3, 20
    basis, _ = np.linalg.qr(RNG.standard_normal((ambient, k)))
    x = RNG.standard_normal((5000, k)) @ basis.T
    assert participation_ratio(x) == pytest.approx(k, rel=0.05)


def test_pr_of_constant_cloud_is_zero() -> None:
    assert participation_ratio(np.ones((50, 4))) == 0.0


def test_pr_is_translation_invariant() -> None:
    x = RNG.standard_normal((500, 6))
    np.testing.assert_allclose(participation_ratio(x), participation_ratio(x + 100.0), rtol=1e-10)


def test_pr_rejects_non_2d() -> None:
    with pytest.raises(ValueError, match="n_samples"):
        participation_ratio(np.zeros(5))


# ---------------------------------------------------------------------------
# TwoNN intrinsic dimension
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("d", [1, 2, 5])
def test_twonn_recovers_flat_dimension(d: int) -> None:
    x = RNG.uniform(size=(3000, d))
    estimate = twonn_intrinsic_dimension(x)
    assert estimate == pytest.approx(d, rel=0.15)


def test_twonn_sees_through_embedding() -> None:
    """A 2-D plane embedded in R^50 still measures dimension 2."""
    plane = RNG.uniform(size=(3000, 2))
    basis, _ = np.linalg.qr(RNG.standard_normal((50, 2)))
    embedded = plane @ basis.T
    assert twonn_intrinsic_dimension(embedded) == pytest.approx(2.0, rel=0.15)


def test_twonn_sees_through_curvature() -> None:
    """A circle in R^10 is a 1-D manifold."""
    t = RNG.uniform(0, 2 * np.pi, size=2000)
    circle = np.stack([np.cos(t), np.sin(t)], axis=1)
    basis, _ = np.linalg.qr(RNG.standard_normal((10, 2)))
    embedded = circle @ basis.T
    assert twonn_intrinsic_dimension(embedded) == pytest.approx(1.0, rel=0.15)


def test_twonn_ignores_duplicate_points() -> None:
    x = RNG.uniform(size=(500, 2))
    with_duplicates = np.concatenate([x, x[:50]])
    estimate = twonn_intrinsic_dimension(with_duplicates)
    assert np.isfinite(estimate)
    assert estimate == pytest.approx(2.0, rel=0.25)


def test_twonn_requires_three_points() -> None:
    with pytest.raises(ValueError, match="at least 3"):
        twonn_intrinsic_dimension(np.zeros((2, 3)))


def test_twonn_rejects_non_2d() -> None:
    with pytest.raises(ValueError, match="n_samples"):
        twonn_intrinsic_dimension(np.zeros(5))


# ---------------------------------------------------------------------------
# Class separability
# ---------------------------------------------------------------------------


def make_blobs(separation: float, n_per_class: int = 400) -> tuple[np.ndarray, np.ndarray]:
    centers = np.array([[0.0, 0.0], [separation, 0.0], [0.0, separation]])
    xs, ys = [], []
    for c, center in enumerate(centers):
        xs.append(RNG.standard_normal((n_per_class, 2)) + center)
        ys.append(np.full(n_per_class, c))
    return np.concatenate(xs), np.concatenate(ys)


def test_separability_zero_when_means_coincide() -> None:
    x = RNG.standard_normal((600, 4))
    labels = np.repeat([0, 1, 2], 200)
    assert class_separability(x, labels) == pytest.approx(0.0, abs=0.05)


def test_separability_grows_with_separation() -> None:
    x_close, y_close = make_blobs(1.0)
    x_far, y_far = make_blobs(10.0)
    assert class_separability(x_far, y_far) > 10 * class_separability(x_close, y_close)


def test_separability_infinite_for_collapsed_classes() -> None:
    x = np.repeat(np.array([[0.0, 0.0], [5.0, 5.0]]), 10, axis=0)
    labels = np.repeat([0, 1], 10)
    assert class_separability(x, labels) == float("inf")


def test_separability_validates_lengths() -> None:
    with pytest.raises(ValueError, match="labels"):
        class_separability(np.zeros((4, 2)), np.zeros(3))


def test_separability_rejects_non_2d() -> None:
    with pytest.raises(ValueError, match="n_samples"):
        class_separability(np.zeros(5), np.zeros(5))
