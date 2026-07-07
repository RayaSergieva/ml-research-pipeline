"""Tests for the spectral analysis instrument.

Statistics are validated against matrices with spectra known in closed form,
mostly built by explicit SVD synthesis U diag(s) V^T from random orthogonal
factors.
"""

from __future__ import annotations

import numpy as np
import pytest

from spectra.analysis import SpectralTracker, matrix_spectrum
from spectra.analysis.spectral import SpectralSnapshot
from spectra.nn import Linear, Module

RNG = np.random.default_rng(11)


def matrix_with_spectrum(singular_values: np.ndarray, m: int, n: int) -> np.ndarray:
    """Build an m x n matrix whose singular values are exactly the given ones."""
    r = len(singular_values)
    assert r <= min(m, n)
    u, _ = np.linalg.qr(RNG.standard_normal((m, r)))
    v, _ = np.linalg.qr(RNG.standard_normal((n, r)))
    return np.asarray(u @ np.diag(singular_values) @ v.T)


# ---------------------------------------------------------------------------
# matrix_spectrum
# ---------------------------------------------------------------------------


def test_spectrum_of_synthesized_matrix() -> None:
    s = np.array([5.0, 2.0, 1.0])
    w = matrix_with_spectrum(s, 6, 4)
    recovered = matrix_spectrum(w)
    np.testing.assert_allclose(recovered[:3], s, atol=1e-10)
    np.testing.assert_allclose(recovered[3:], 0.0, atol=1e-10)


def test_spectrum_is_descending() -> None:
    w = RNG.standard_normal((8, 5))
    s = matrix_spectrum(w)
    assert np.all(np.diff(s) <= 0)


def test_spectrum_rejects_non_matrix() -> None:
    with pytest.raises(ValueError, match="2-D"):
        matrix_spectrum(RNG.standard_normal(4))


def test_spectrum_of_identity() -> None:
    np.testing.assert_allclose(matrix_spectrum(np.eye(4)), np.ones(4))


def test_spectrum_invariant_under_orthogonal_maps() -> None:
    """Singular values depend only on the map's geometry, not orientation."""
    w = RNG.standard_normal((5, 5))
    q, _ = np.linalg.qr(RNG.standard_normal((5, 5)))
    np.testing.assert_allclose(matrix_spectrum(q @ w), matrix_spectrum(w), atol=1e-10)


# ---------------------------------------------------------------------------
# SpectralSnapshot statistics
# ---------------------------------------------------------------------------


def snapshot_for(s: np.ndarray) -> SpectralSnapshot:
    return SpectralSnapshot(step=0, name="w", shape=(4, 4), singular_values=s)


def test_spectral_and_frobenius_norms() -> None:
    snap = snapshot_for(np.array([3.0, 4.0][::-1]))  # descending [4, 3]
    assert snap.spectral_norm == 4.0
    np.testing.assert_allclose(snap.frobenius_norm, 5.0)  # sqrt(16 + 9)


def test_condition_number() -> None:
    snap = snapshot_for(np.array([10.0, 2.0]))
    assert snap.condition_number == 5.0
    singular = snapshot_for(np.array([1.0, 0.0]))
    assert singular.condition_number == float("inf")


def test_stable_rank_extremes() -> None:
    flat = snapshot_for(np.ones(6))
    np.testing.assert_allclose(flat.stable_rank, 6.0)
    spiked = snapshot_for(np.array([100.0, 1e-8, 1e-8]))
    assert spiked.stable_rank == pytest.approx(1.0, abs=1e-10)


def test_effective_rank_extremes() -> None:
    flat = snapshot_for(np.ones(6))
    np.testing.assert_allclose(flat.effective_rank, 6.0, rtol=1e-12)
    one_direction = snapshot_for(np.array([7.0]))
    np.testing.assert_allclose(one_direction.effective_rank, 1.0)


def test_effective_rank_of_zero_matrix() -> None:
    assert snapshot_for(np.zeros(3)).effective_rank == 0.0


def test_summary_contains_all_statistics() -> None:
    keys = set(snapshot_for(np.array([2.0, 1.0])).summary())
    assert keys == {
        "spectral_norm",
        "frobenius_norm",
        "condition_number",
        "stable_rank",
        "effective_rank",
    }


# ---------------------------------------------------------------------------
# SpectralTracker
# ---------------------------------------------------------------------------


class SmallNet(Module):
    def __init__(self) -> None:
        super().__init__()
        rng = np.random.default_rng(0)
        self.fc1 = Linear(4, 6, rng=rng)
        self.fc2 = Linear(6, 2, rng=rng)


def test_tracker_tracks_only_matrices() -> None:
    tracker = SpectralTracker(SmallNet())
    assert len(tracker.names) == 2  # two weight matrices, biases skipped


def test_tracker_capture_and_history() -> None:
    net = SmallNet()
    tracker = SpectralTracker(net)
    tracker.capture(step=0)
    net.fc1.weight.data = net.fc1.weight.data * 2.0
    tracker.capture(step=10)

    assert len(tracker.snapshots) == 4
    name = tracker.names[0]
    history = tracker.history(name)
    assert [s.step for s in history] == [0, 10]
    # Doubling the matrix doubles every singular value.
    np.testing.assert_allclose(
        history[1].singular_values,
        2.0 * history[0].singular_values,
        rtol=1e-10,
    )


def test_tracker_snapshot_shapes_match_parameters() -> None:
    net = SmallNet()
    tracker = SpectralTracker(net)
    taken = tracker.capture(step=0)
    shapes = {s.shape for s in taken}
    assert shapes == {(4, 6), (6, 2)}
    for s in taken:
        assert len(s.singular_values) == min(s.shape)
