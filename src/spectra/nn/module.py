"""The :class:`Module` base class.

A Module is a container for trainable parameters and child modules. Attribute
assignment is intercepted so that assigning a :class:`~spectra.tensor.Tensor`
with ``requires_grad=True`` registers it as a parameter, and assigning another
Module registers it as a child. :meth:`parameters` then walks the tree and
yields every trainable tensor, which is all an optimizer needs.

The design mirrors the module systems of the major frameworks, reduced to the
minimum that training requires. Buffers, hooks, and serialization are Horizon
2 concerns.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from spectra.tensor import Tensor


class Module:
    """Base class for all neural-network components.

    Subclasses must call ``super().__init__()`` before assigning any
    parameters or child modules, then implement :meth:`forward`. Calling the
    module invokes :meth:`forward`.
    """

    def __init__(self) -> None:
        object.__setattr__(self, "_parameters", {})
        object.__setattr__(self, "_modules", {})

    def __setattr__(self, name: str, value: Any) -> None:
        registry = self.__dict__
        if "_parameters" not in registry:
            msg = (
                f"cannot assign attributes to {type(self).__name__} before "
                "Module.__init__() has run; call super().__init__() first"
            )
            raise RuntimeError(msg)
        if isinstance(value, Tensor) and value.requires_grad:
            registry["_parameters"][name] = value
        elif isinstance(value, Module):
            registry["_modules"][name] = value
        object.__setattr__(self, name, value)

    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        """Compute the module's output. Subclasses must override."""
        raise NotImplementedError

    def __call__(self, *args: Any, **kwargs: Any) -> Tensor:
        return self.forward(*args, **kwargs)

    def parameters(self) -> Iterator[Tensor]:
        """Yield every trainable tensor in this module and its children."""
        params: dict[str, Tensor] = self.__dict__["_parameters"]
        modules: dict[str, Module] = self.__dict__["_modules"]
        yield from params.values()
        for module in modules.values():
            yield from module.parameters()

    def zero_grad(self) -> None:
        """Reset the gradient of every parameter."""
        for p in self.parameters():
            p.zero_grad()
