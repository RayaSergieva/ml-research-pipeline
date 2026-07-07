"""Gradient-based optimizers.

Both optimizers derive from gradient descent on a differentiable objective.
Their update rules and the reasoning behind them (momentum as an exponential
moving average of gradients, Adam's bias-corrected first and second moments)
are derived in the notebook; the implementations here follow the original
papers exactly.

Updates are performed in place on each parameter's ``data`` buffer, outside
the autograd graph, which is the standard treatment of the training loop as
non-differentiable.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from spectra.tensor import Tensor

Array = NDArray[Any]


class Optimizer:
    """Base class holding the parameter list."""

    def __init__(self, parameters: Iterable[Tensor]) -> None:
        self.parameters = list(parameters)
        if not self.parameters:
            msg = "optimizer received an empty parameter list"
            raise ValueError(msg)

    def step(self) -> None:
        """Apply one update using the gradients currently stored."""
        raise NotImplementedError

    def zero_grad(self) -> None:
        """Reset the gradient of every managed parameter."""
        for p in self.parameters:
            p.zero_grad()


class SGD(Optimizer):
    """Stochastic gradient descent, optionally with classical momentum.

    Plain SGD updates theta <- theta - lr * g. With momentum m in (0, 1) a
    velocity buffer v accumulates an exponentially weighted sum of past
    gradients, v <- m v + g, and the parameter moves along the velocity,
    theta <- theta - lr v. Momentum damps oscillation across steep, narrow
    directions of the loss surface and accelerates along consistent ones.
    """

    def __init__(
        self,
        parameters: Iterable[Tensor],
        lr: float,
        momentum: float = 0.0,
    ) -> None:
        super().__init__(parameters)
        if lr <= 0:
            msg = f"learning rate must be positive, got {lr}"
            raise ValueError(msg)
        if not 0.0 <= momentum < 1.0:
            msg = f"momentum must be in [0, 1), got {momentum}"
            raise ValueError(msg)
        self.lr = lr
        self.momentum = momentum
        self._velocity: list[Array | None] = [None] * len(self.parameters)

    def step(self) -> None:
        for i, p in enumerate(self.parameters):
            if p.grad is None:
                continue
            if self.momentum == 0.0:
                p.data = p.data - self.lr * p.grad
                continue
            v = self._velocity[i]
            v = p.grad.copy() if v is None else self.momentum * v + p.grad
            self._velocity[i] = v
            p.data = p.data - self.lr * v


class Adam(Optimizer):
    """Adam (Kingma & Ba, 2015).

    Maintains exponential moving averages of the gradient (first moment m)
    and of its elementwise square (second moment v), corrects both for their
    initialization bias toward zero, and scales the step of each coordinate
    by 1 / (sqrt(v_hat) + eps). The result is an approximately
    scale-invariant per-coordinate step size.
    """

    def __init__(
        self,
        parameters: Iterable[Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ) -> None:
        super().__init__(parameters)
        if lr <= 0:
            msg = f"learning rate must be positive, got {lr}"
            raise ValueError(msg)
        beta1, beta2 = betas
        if not (0.0 <= beta1 < 1.0 and 0.0 <= beta2 < 1.0):
            msg = f"betas must each be in [0, 1), got {betas}"
            raise ValueError(msg)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self._step_count = 0
        self._m: list[Array | None] = [None] * len(self.parameters)
        self._v: list[Array | None] = [None] * len(self.parameters)

    def step(self) -> None:
        self._step_count += 1
        t = self._step_count
        for i, p in enumerate(self.parameters):
            if p.grad is None:
                continue
            g = p.grad
            m_prev = self._m[i]
            v_prev = self._v[i]
            m = (
                (1 - self.beta1) * g
                if m_prev is None
                else self.beta1 * m_prev + (1 - self.beta1) * g
            )
            v = (
                (1 - self.beta2) * g * g
                if v_prev is None
                else self.beta2 * v_prev + (1 - self.beta2) * g * g
            )
            self._m[i] = m
            self._v[i] = v
            m_hat = m / (1 - self.beta1**t)
            v_hat = v / (1 - self.beta2**t)
            p.data = p.data - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
