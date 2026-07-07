"""Loss functions.

Only the loss needed for classification is implemented so far. It is written
as a single fused autograd node rather than a composition of exp, sum, and
log, for two reasons developed fully in the notebook. Numerically, the
log-sum-exp is stabilised by subtracting the row maximum, which the naive
composition does not do. And analytically, the fused backward rule collapses
to the famously simple expression softmax(z) - onehot(y), scaled by 1/N,
which is both faster and exactly what the derivation produces.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from spectra.autograd import Function
from spectra.tensor import Tensor

Array = NDArray[Any]


class SoftmaxCrossEntropy(Function):
    """Mean cross-entropy between logits and integer class targets.

    For a batch of N rows of logits z and target classes y, computes

        L = -(1/N) sum_i log softmax(z_i)[y_i]

    using the max-shifted log-sum-exp for numerical stability. The backward
    rule is (softmax(z) - onehot(y)) / N.
    """

    def forward(self, logits: Array, *, targets: Array) -> Array:  # type: ignore[override]
        if logits.ndim != 2:
            msg = f"expected 2-D logits (batch, classes), got {logits.ndim}-D"
            raise ValueError(msg)
        if targets.ndim != 1 or targets.shape[0] != logits.shape[0]:
            msg = (
                f"expected 1-D integer targets of length {logits.shape[0]}, "
                f"got shape {targets.shape}"
            )
            raise ValueError(msg)

        shifted = logits - logits.max(axis=1, keepdims=True)
        log_sum_exp = np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        log_probs = shifted - log_sum_exp

        n = logits.shape[0]
        softmax = np.exp(log_probs)
        self.save_for_backward(softmax, targets)
        picked = log_probs[np.arange(n), targets]
        return np.asarray(-picked.mean())

    def backward(self, grad_output: Array) -> tuple[Array | None, ...]:
        softmax, targets = self.saved
        n = softmax.shape[0]
        grad = softmax.copy()
        grad[np.arange(n), targets.astype(np.intp)] -= 1.0
        grad /= n
        return (grad * grad_output,)


def softmax_cross_entropy(logits: Tensor, targets: Array) -> Tensor:
    """Convenience wrapper applying :class:`SoftmaxCrossEntropy`.

    ``targets`` is a plain integer array, not a Tensor, because class labels
    are constants of the optimization, never differentiated through.
    """
    return SoftmaxCrossEntropy.apply(logits, targets=np.asarray(targets))
