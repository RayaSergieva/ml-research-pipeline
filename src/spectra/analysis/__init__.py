"""Analytical instruments for studying trained and training networks."""

from spectra.analysis.geometry import (
    class_separability,
    participation_ratio,
    twonn_intrinsic_dimension,
)
from spectra.analysis.landscape import (
    loss_plane,
    top_hessian_eigenvalue,
)
from spectra.analysis.spectral import SpectralSnapshot, SpectralTracker, matrix_spectrum

__all__ = [
    "SpectralSnapshot",
    "SpectralTracker",
    "class_separability",
    "loss_plane",
    "matrix_spectrum",
    "participation_ratio",
    "top_hessian_eigenvalue",
    "twonn_intrinsic_dimension",
]
