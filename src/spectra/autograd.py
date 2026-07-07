"""Reverse-mode automatic differentiation.

The engine follows the standard define-by-run design: every differentiable
operation is a :class:`Function`. Calling :meth:`Function.apply` executes the
forward computation on raw arrays, wraps the result in a new
:class:`~spectra.tensor.Tensor`, and records the node on that tensor so the
computational graph is built implicitly as expressions are evaluated.

The backward pass (:meth:`spectra.tensor.Tensor.backward`) traverses the graph
in reverse topological order and applies the chain rule at every node. Each
node maps the gradient of the loss with respect to its output to gradients
with respect to each of its inputs; gradients arriving at the same tensor from
different paths are summed, which is exactly the multivariate chain rule.

Reverse mode is the right choice for neural-network training: for a function
f: R^n -> R (many parameters, scalar loss) it computes the full gradient in a
single backward sweep, at a cost proportional to the forward pass, whereas
forward-mode differentiation would require n sweeps.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from numpy.typing import NDArray

if TYPE_CHECKING:
    from spectra.tensor import Tensor


class Function:
    """A node in the computational graph.

    Subclasses implement :meth:`forward` on raw numpy arrays and
    :meth:`backward`, which receives the gradient of the loss with respect to
    the node's output and returns a tuple of gradients with respect to each
    input (``None`` for inputs that do not require a gradient).

    Instances are created by :meth:`apply`, never directly.
    """

    def __init__(self, *parents: Tensor) -> None:
        self.parents: tuple[Tensor, ...] = parents
        self._saved: tuple[NDArray[Any], ...] = ()

    def save_for_backward(self, *arrays: NDArray[Any]) -> None:
        """Store arrays needed by :meth:`backward`."""
        self._saved = arrays

    @property
    def saved(self) -> tuple[NDArray[Any], ...]:
        """The arrays stored by :meth:`save_for_backward`."""
        return self._saved

    def forward(self, *args: NDArray[Any], **kwargs: Any) -> NDArray[Any]:
        """Compute the operation on raw arrays."""
        raise NotImplementedError

    def backward(self, grad_output: NDArray[Any]) -> tuple[NDArray[Any] | None, ...]:
        """Map the output gradient to input gradients (chain rule step)."""
        raise NotImplementedError

    @classmethod
    def apply(cls, *inputs: Tensor, **kwargs: Any) -> Tensor:
        """Run the forward pass and record the node on the output tensor.

        The output requires a gradient iff any input does; graph recording is
        skipped entirely otherwise, so inference-only code pays no overhead.
        """
        from spectra.tensor import Tensor

        node = cls(*inputs)
        out_data = node.forward(*(t.data for t in inputs), **kwargs)
        requires_grad = any(t.requires_grad for t in inputs)
        out = Tensor(out_data, dtype=out_data.dtype, requires_grad=requires_grad)
        if requires_grad:
            out._ctx = node
        return out
