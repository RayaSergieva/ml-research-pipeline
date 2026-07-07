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
