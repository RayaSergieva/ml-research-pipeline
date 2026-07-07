"""Representation geometry - measuring the shape of activation clouds.

A trained layer maps inputs to a cloud of activation vectors. Three numbers
summarize the geometry of that cloud, each with a precise meaning and a
closed-form oracle the tests exploit.

**Participation ratio.** With C the covariance of the activations and
lambda_i its eigenvalues, PR = (sum lambda_i)^2 / sum lambda_i^2. When k
eigenvalues are equal and the rest zero, PR = k exactly, so PR counts the
directions in which the cloud has appreciable variance - a linear notion of
dimensionality, the same statistic as the stable rank of the centered data
matrix.

**TwoNN intrinsic dimension** (Facco et al., 2017). If points are sampled
from a density on a d-dimensional manifold, the ratio mu = r_2 / r_1 of each
point's second to first nearest-neighbour distance follows the law
F(mu) = 1 - mu^(-d), independently of the density. The maximum-likelihood
estimate given ratios mu_1..mu_N is

    d_hat = N / sum_i ln(mu_i),

a nonlinear notion of dimensionality that sees through curvature - points on
a curved surface embedded in R^100 still measure d ~= 2.

**Class separability.** With S_B the between-class scatter (variance of
class means around the global mean, weighted by class size) and S_W the
within-class scatter (pooled variance around each class mean), the ratio
trace(S_B) / trace(S_W) is a Fisher-style score - zero when class means
coincide, growing as the classes pull apart relative to their spread. Along
the depth of a classifier this score should rise, which is the geometric
restatement of "the network separates the classes".
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

Array = NDArray[Any]


def participation_ratio(activations: Array) -> float:
    """PR of the activation covariance, (sum lambda)^2 / sum lambda^2.

    ``activations`` has shape (n_samples, n_features). Returns a value in
    [1, n_features]; 0 for a degenerate all-constant cloud.
    """
    if activations.ndim != 2:
        msg = f"expected (n_samples, n_features), got {activations.ndim}-D"
        raise ValueError(msg)
    centered = activations - activations.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / max(len(centered) - 1, 1)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    total = float(eigenvalues.sum())
    if total == 0.0:
        return 0.0
    return float(total**2 / np.sum(eigenvalues**2))


def twonn_intrinsic_dimension(points: Array, rng: np.random.Generator | None = None) -> float:
    """TwoNN maximum-likelihood intrinsic dimension estimate.

    Uses every point's first and second nearest neighbours. Duplicate points
    (zero first-neighbour distance) are excluded from the estimate, as the
    ratio is undefined there. Requires at least 3 points.
    """
    if points.ndim != 2:
        msg = f"expected (n_samples, n_features), got {points.ndim}-D"
        raise ValueError(msg)
    n = len(points)
    if n < 3:
        msg = f"TwoNN needs at least 3 points, got {n}"
        raise ValueError(msg)

    # Pairwise Euclidean distances via the expansion |a-b|^2 = |a|^2 + |b|^2 - 2ab.
    sq = np.sum(points**2, axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (points @ points.T)
    np.fill_diagonal(d2, np.inf)
    d2 = np.clip(d2, 0.0, None)

    two_smallest = np.partition(d2, 1, axis=1)[:, :2]
    r1 = np.sqrt(two_smallest[:, 0])
    r2 = np.sqrt(two_smallest[:, 1])

    valid = r1 > 0
    if not np.any(valid):
        return 0.0
    mu = r2[valid] / r1[valid]
    log_mu = np.log(mu)
    denominator = float(log_mu.sum())
    if denominator == 0.0:
        return float("inf")
    return float(len(mu) / denominator)


def class_separability(activations: Array, labels: Array) -> float:
    """Fisher-style ratio trace(S_B) / trace(S_W).

    Returns inf for perfectly collapsed classes (zero within-class scatter
    with distinct means) and 0 when all class means coincide.
    """
    if activations.ndim != 2:
        msg = f"expected (n_samples, n_features), got {activations.ndim}-D"
        raise ValueError(msg)
    if len(labels) != len(activations):
        msg = f"got {len(activations)} activation rows but {len(labels)} labels"
        raise ValueError(msg)

    global_mean = activations.mean(axis=0)
    between = 0.0
    within = 0.0
    for c in np.unique(labels):
        rows = activations[labels == c]
        mean_c = rows.mean(axis=0)
        between += len(rows) * float(np.sum((mean_c - global_mean) ** 2))
        within += float(np.sum((rows - mean_c) ** 2))
    if within == 0.0:
        return float("inf") if between > 0.0 else 0.0
    return between / within
