"""Loss landscape probing - curvature and planar slices.

Two views of the loss surface around the current parameters are provided.

**Curvature.** The Hessian H of the loss with respect to all parameters is
far too large to form, but its action on a vector is accessible without
second-order autograd through the central difference of gradients,

    H v  ~=  ( g(theta + eps v) - g(theta - eps v) ) / (2 eps),

accurate to O(eps^2) since the odd error terms cancel. Power iteration on
this operator converges to the eigenvector of the largest-magnitude
eigenvalue, and the Rayleigh quotient v^T H v recovers the eigenvalue
itself - the *sharpness* of the minimum, the quantity that governs the
stability threshold of gradient descent (section 5 of the project notebook,
eta < 2 / lambda_max).

**Slices.** Following Li et al. (2018), the loss is evaluated on the plane
theta + alpha d1 + beta d2 spanned by two random directions. Each direction
is filter-normalized - rescaled parameter-by-parameter to match the norm of
the corresponding parameter - so that slices are comparable between layers
and networks of different scales.

Everything here treats the model's parameters as one flat vector; the
flatten/unflatten helpers below define that correspondence.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from spectra.nn import Module
from spectra.tensor import Tensor

Array = NDArray[Any]
LossFn = Callable[[], Tensor]


def flatten_parameters(model: Module) -> Array:
    """Concatenate every parameter into one flat vector (a copy)."""
    return np.concatenate([p.data.ravel() for p in model.parameters()])


def set_parameters(model: Module, flat: Array) -> None:
    """Write a flat vector back into the model's parameters."""
    total = sum(p.size for p in model.parameters())
    if flat.size != total:
        msg = f"vector has {flat.size} entries but the model has {total} parameters"
        raise ValueError(msg)
    offset = 0
    for p in model.parameters():
        count = p.size
        p.data = flat[offset : offset + count].reshape(p.shape)
        offset += count


def gradient_vector(model: Module, loss_fn: LossFn) -> Array:
    """The gradient of the loss at the current parameters, flattened."""
    model.zero_grad()
    loss_fn().backward()
    parts = []
    for p in model.parameters():
        if p.grad is None:
            parts.append(np.zeros(p.size))
        else:
            parts.append(p.grad.ravel().copy())
    model.zero_grad()
    return np.concatenate(parts)


def hessian_vector_product(
    model: Module,
    loss_fn: LossFn,
    v: Array,
    eps: float = 1e-4,
) -> Array:
    """H v by the central difference of gradients, restoring parameters."""
    theta = flatten_parameters(model)
    if v.shape != theta.shape:
        msg = f"direction has shape {v.shape}, parameters have shape {theta.shape}"
        raise ValueError(msg)
    try:
        set_parameters(model, theta + eps * v)
        g_plus = gradient_vector(model, loss_fn)
        set_parameters(model, theta - eps * v)
        g_minus = gradient_vector(model, loss_fn)
    finally:
        set_parameters(model, theta)
    return (g_plus - g_minus) / (2.0 * eps)


def top_hessian_eigenvalue(
    model: Module,
    loss_fn: LossFn,
    iterations: int = 30,
    rng: np.random.Generator | None = None,
    eps: float = 1e-4,
) -> tuple[float, Array]:
    """Largest-magnitude Hessian eigenvalue and its eigenvector.

    Power iteration - repeatedly apply H and renormalize. The iterate aligns
    with the dominant eigenvector at a geometric rate set by the eigenvalue
    gap; 30 iterations resolve the sharpness of small networks to several
    digits, as the tests verify against exact eigenvalues.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    n = flatten_parameters(model).size
    v = rng.standard_normal(n)
    v /= np.linalg.norm(v)
    eigenvalue = 0.0
    for _ in range(iterations):
        hv = hessian_vector_product(model, loss_fn, v, eps=eps)
        norm = float(np.linalg.norm(hv))
        if norm == 0.0:
            return 0.0, v
        eigenvalue = float(v @ hv)  # Rayleigh quotient
        v = hv / norm
    return eigenvalue, v


def filter_normalized_direction(model: Module, rng: np.random.Generator) -> Array:
    """A random direction rescaled per-parameter to that parameter's norm.

    Raw Gaussian directions make slices incomparable, because a layer with
    large weights looks flat along a unit direction while a small layer
    looks steep. Matching each block's norm to its parameter's norm (Li et
    al., 2018) removes this scale artifact.
    """
    blocks = []
    for p in model.parameters():
        d = rng.standard_normal(p.size)
        d_norm = float(np.linalg.norm(d))
        p_norm = float(np.linalg.norm(p.data))
        if d_norm > 0:
            d *= p_norm / d_norm
        blocks.append(d)
    return np.concatenate(blocks)


def loss_plane(
    model: Module,
    loss_fn: LossFn,
    alphas: Array,
    betas: Array,
    rng: np.random.Generator | None = None,
) -> Array:
    """Loss values on the plane theta + alpha d1 + beta d2.

    Returns a grid of shape (len(alphas), len(betas)). Directions are drawn
    filter-normalized; parameters are restored afterwards.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    theta = flatten_parameters(model)
    d1 = filter_normalized_direction(model, rng)
    d2 = filter_normalized_direction(model, rng)
    grid = np.zeros((len(alphas), len(betas)))
    try:
        for i, a in enumerate(alphas):
            for j, b in enumerate(betas):
                set_parameters(model, theta + a * d1 + b * d2)
                grid[i, j] = float(loss_fn().data)
    finally:
        set_parameters(model, theta)
    return grid
