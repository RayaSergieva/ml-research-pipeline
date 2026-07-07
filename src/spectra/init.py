"""Weight initialization schemes.

Initialization sets the spectral scale of a network before training starts.
The variance-preserving arguments behind Xavier and He initialization are
derived in the project notebook; in brief, both choose the weight variance so
that the variance of activations (and of back-propagated gradients) neither
explodes nor vanishes as depth grows. Xavier assumes a linear or symmetric
activation and balances forward and backward variance via the average of
fan-in and fan-out; He accounts for ReLU halving the activation variance and
therefore doubles the weight variance, using fan-in alone.

These functions return plain numpy arrays; wrapping them in trainable tensors
is the caller's job (see :class:`spectra.nn.Linear`). A dedicated random
generator is threaded through every function so experiments are reproducible.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from spectra._backend import default_dtype

Array = NDArray[Any]


def _fans(shape: tuple[int, int]) -> tuple[int, int]:
    fan_in, fan_out = shape
    return fan_in, fan_out


def zeros(shape: tuple[int, ...]) -> Array:
    """An all-zeros array, the standard choice for biases."""
    return np.zeros(shape, dtype=default_dtype())


def xavier_uniform(shape: tuple[int, int], rng: np.random.Generator) -> Array:
    """Glorot & Bengio (2010) uniform initialization.

    Samples U(-a, a) with a = sqrt(6 / (fan_in + fan_out)), giving weight
    variance 2 / (fan_in + fan_out).
    """
    fan_in, fan_out = _fans(shape)
    bound = float(np.sqrt(6.0 / (fan_in + fan_out)))
    return rng.uniform(-bound, bound, size=shape).astype(default_dtype())


def xavier_normal(shape: tuple[int, int], rng: np.random.Generator) -> Array:
    """Glorot & Bengio (2010) normal initialization.

    Samples N(0, sigma^2) with sigma = sqrt(2 / (fan_in + fan_out)).
    """
    fan_in, fan_out = _fans(shape)
    std = float(np.sqrt(2.0 / (fan_in + fan_out)))
    return (rng.standard_normal(size=shape) * std).astype(default_dtype())


def he_uniform(shape: tuple[int, int], rng: np.random.Generator) -> Array:
    """He et al. (2015) uniform initialization for ReLU networks.

    Samples U(-a, a) with a = sqrt(6 / fan_in), giving weight variance
    2 / fan_in, which compensates for ReLU zeroing half the activations.
    """
    fan_in, _ = _fans(shape)
    bound = float(np.sqrt(6.0 / fan_in))
    return rng.uniform(-bound, bound, size=shape).astype(default_dtype())


def he_normal(shape: tuple[int, int], rng: np.random.Generator) -> Array:
    """He et al. (2015) normal initialization for ReLU networks.

    Samples N(0, sigma^2) with sigma = sqrt(2 / fan_in).
    """
    fan_in, _ = _fans(shape)
    std = float(np.sqrt(2.0 / fan_in))
    return (rng.standard_normal(size=shape) * std).astype(default_dtype())


def spectral(
    shape: tuple[int, int],
    rng: np.random.Generator,
    alpha: float = 0.5,
) -> Array:
    """Spectrum-aware initialization, W = U diag(s) V^T with a designed
    spectrum.

    Motivated by this project's empirical finding that training concentrates
    every layer's spectrum into a small number of dominant directions, the
    scheme starts the matrix with that anisotropy already present instead of
    the flat Marchenko-Pastur bulk of i.i.d. schemes.

    The singular values follow a power law, s_i proportional to i^(-alpha)
    for i = 1..r with r = min(shape), and the whole spectrum is rescaled so
    the Frobenius norm equals that of a He-normal draw of the same shape,
    E||W||_F^2 = 2 * fan_out (with the (fan_in, fan_out) layout used by the
    framework, He variance 2/fan_in times fan_in*fan_out entries). Energy is
    therefore matched to the He baseline and only its distribution across
    directions changes, isolating the effect under study. alpha = 0 gives a
    flat spectrum of the same energy; larger alpha concentrates it.

    The orthogonal factors are drawn from the Haar measure via QR of
    Gaussian matrices, so the singular vectors carry no preferred
    orientation - the spectrum is the only structure injected.
    """
    if alpha < 0:
        msg = f"alpha must be non-negative, got {alpha}"
        raise ValueError(msg)
    fan_in, fan_out = _fans(shape)
    r = min(fan_in, fan_out)

    s = np.arange(1, r + 1, dtype=np.float64) ** (-alpha)
    target_frobenius_sq = 2.0 * fan_out  # matches E||W||_F^2 under He
    s *= np.sqrt(target_frobenius_sq / np.sum(s**2))

    u, _ = np.linalg.qr(rng.standard_normal((fan_in, r)))
    v, _ = np.linalg.qr(rng.standard_normal((fan_out, r)))
    return np.asarray((u * s) @ v.T, dtype=default_dtype())
