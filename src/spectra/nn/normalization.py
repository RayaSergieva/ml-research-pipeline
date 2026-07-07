"""Embedding and normalization layers."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from spectra._backend import default_dtype
from spectra.nn.module import Module
from spectra.ops import EmbeddingLookup
from spectra.tensor import Tensor

Array = NDArray[Any]


class Embedding(Module):
    """A trainable lookup table mapping integer ids to vectors.

    The table is initialized N(0, 1) scaled by 1/sqrt(dim), keeping the
    embedding norm independent of the dimension, the common choice for
    token and position tables in language models.
    """

    def __init__(self, num_embeddings: int, dim: int, rng: np.random.Generator) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.dim = dim
        table = rng.standard_normal((num_embeddings, dim)) / np.sqrt(dim)
        self.weight = Tensor(table.astype(default_dtype()), requires_grad=True)

    def forward(self, indices: Array) -> Tensor:
        if np.any(indices < 0) or np.any(indices >= self.num_embeddings):
            msg = f"index out of range for embedding of size {self.num_embeddings}"
            raise ValueError(msg)
        return EmbeddingLookup.apply(self.weight, indices=indices)


class LayerNorm(Module):
    """Layer normalization over the last axis (Ba et al., 2016).

    y = gamma * (x - mean) / sqrt(var + eps) + beta, with mean and variance
    taken per sample over the feature axis. Unlike batch normalization the
    statistics involve no other samples, so behavior is identical in
    training and inference and independent of batch size. The whole layer is
    composed from the framework's differentiable primitives, so its backward
    pass is derived automatically by the autograd engine rather than coded
    by hand - a small demonstration that the primitive set is sufficient.
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.gamma = Tensor(np.ones(dim), requires_grad=True)
        self.beta = Tensor(np.zeros(dim), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(axis=-1, keepdims=True)
        centered = x - mean
        variance = (centered * centered).mean(axis=-1, keepdims=True)
        inv_std = (variance + self.eps) ** -0.5
        return self.gamma * (centered * inv_std) + self.beta
