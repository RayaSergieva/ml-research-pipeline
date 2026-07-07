"""The fully connected (linear, dense) layer."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from spectra import init
from spectra.nn.module import Module
from spectra.tensor import Tensor

Initializer = Callable[[tuple[int, int], np.random.Generator], NDArray[Any]]


class Linear(Module):
    """The affine map y = x W + b.

    Weights are stored with shape ``(in_features, out_features)`` so the
    forward pass is a plain right-multiplication of a batch of row vectors,
    with no transpose. The bias is a row vector broadcast over the batch.

    Parameters
    ----------
    in_features
        Dimensionality of each input row.
    out_features
        Dimensionality of each output row.
    rng
        Random generator used for weight initialization; passing it
        explicitly keeps every experiment reproducible.
    weight_init
        Initialization scheme for the weight matrix, one of the functions in
        :mod:`spectra.init`. Defaults to He normal, the appropriate choice
        for the ReLU networks this framework targets first.
    bias
        Whether to include the additive bias term.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rng: np.random.Generator,
        weight_init: Initializer = init.he_normal,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Tensor(
            weight_init((in_features, out_features), rng),
            requires_grad=True,
        )
        self.bias: Tensor | None = (
            Tensor(init.zeros((out_features,)), requires_grad=True) if bias else None
        )

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out
