"""Spectral analysis of weight matrices during training.

The instrument of this module is the singular value decomposition. For a
weight matrix W of shape (m, n), the SVD factorizes W = U S V^T with S the
diagonal matrix of singular values s_1 >= s_2 >= ... >= s_r >= 0. The
singular values are the semi-axis lengths of the ellipsoid into which W maps
the unit sphere, so the spectrum is a complete description of how the layer
stretches and compresses its input space, independent of orientation.

Summary statistics recorded per snapshot

- spectral norm, s_1, the largest amplification the layer can apply
- condition number, s_1 / s_r, sensitivity of the layer as a linear system
- stable rank, ||W||_F^2 / s_1^2 = (sum s_i^2) / s_1^2, a smooth, outlier-
  robust surrogate for rank that does not depend on a threshold
- effective rank, exp(H(p)) with p_i = s_i / sum(s_j) and H the Shannon
  entropy (Roy & Vetterli, 2007), the exponential of the spectrum's entropy,
  which counts how many directions carry meaningful energy
- Frobenius norm, sqrt(sum s_i^2), the overall energy of the map

Tracking these along a training run reveals how a layer reorganizes: rank
collapse or expansion, growth of dominant directions, and conditioning
trends, which is the empirical core of the project's research question.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from spectra.nn import Module

Array = NDArray[Any]


@dataclass(frozen=True)
class SpectralSnapshot:
    """The spectrum of one matrix at one training step."""

    step: int
    name: str
    shape: tuple[int, ...]
    singular_values: Array

    @property
    def spectral_norm(self) -> float:
        """The largest singular value s_1."""
        return float(self.singular_values[0])

    @property
    def frobenius_norm(self) -> float:
        """sqrt(sum of squared singular values)."""
        return float(np.sqrt(np.sum(self.singular_values**2)))

    @property
    def condition_number(self) -> float:
        """s_1 / s_r, infinite for a singular matrix."""
        smallest = float(self.singular_values[-1])
        if smallest == 0.0:
            return float("inf")
        return self.spectral_norm / smallest

    @property
    def stable_rank(self) -> float:
        """||W||_F^2 / s_1^2, a smooth surrogate for rank in [1, rank(W)]."""
        return float(np.sum(self.singular_values**2) / self.singular_values[0] ** 2)

    @property
    def effective_rank(self) -> float:
        """exp of the Shannon entropy of the normalized spectrum.

        Equals the number of singular values when all are equal, and 1 when a
        single direction dominates completely (Roy & Vetterli, 2007).
        """
        total = float(np.sum(self.singular_values))
        if total == 0.0:
            return 0.0
        p = self.singular_values / total
        p = p[p > 0]
        entropy = float(-np.sum(p * np.log(p)))
        return float(np.exp(entropy))

    def summary(self) -> dict[str, float]:
        """All scalar statistics as a flat dict, ready for the run logger."""
        return {
            "spectral_norm": self.spectral_norm,
            "frobenius_norm": self.frobenius_norm,
            "condition_number": self.condition_number,
            "stable_rank": self.stable_rank,
            "effective_rank": self.effective_rank,
        }


def matrix_spectrum(w: Array) -> Array:
    """Singular values of a 2-D array, in descending order."""
    if w.ndim != 2:
        msg = f"expected a 2-D matrix, got {w.ndim}-D"
        raise ValueError(msg)
    return np.asarray(np.linalg.svd(w, compute_uv=False))


@dataclass
class SpectralTracker:
    """Records the spectra of every 2-D parameter of a model over training.

    Vectors (biases) are skipped since their singular value structure is
    trivial. Snapshots accumulate in memory; a full MLP training run stores a
    few hundred small vectors, which is negligible.
    """

    model: Module
    names: dict[int, str] = field(init=False, default_factory=dict)
    snapshots: list[SpectralSnapshot] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        for index, param in enumerate(self.model.parameters()):
            if param.ndim == 2:
                self.names[index] = f"param{index}_{param.shape[0]}x{param.shape[1]}"

    def capture(self, step: int) -> list[SpectralSnapshot]:
        """Snapshot the spectrum of every tracked matrix at ``step``."""
        taken: list[SpectralSnapshot] = []
        for index, param in enumerate(self.model.parameters()):
            if index not in self.names:
                continue
            snap = SpectralSnapshot(
                step=step,
                name=self.names[index],
                shape=param.shape,
                singular_values=matrix_spectrum(param.data),
            )
            self.snapshots.append(snap)
            taken.append(snap)
        return taken

    def history(self, name: str) -> list[SpectralSnapshot]:
        """All snapshots of one matrix, in capture order."""
        return [s for s in self.snapshots if s.name == name]
