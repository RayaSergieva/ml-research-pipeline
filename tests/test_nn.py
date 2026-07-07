"""Tests for the module system, the Linear layer, and initialization."""

from __future__ import annotations

import numpy as np
import pytest

from spectra import Tensor, init
from spectra.nn import Linear, Module

RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Initialization schemes
# ---------------------------------------------------------------------------


def test_zeros() -> None:
    z = init.zeros((3, 4))
    assert z.shape == (3, 4)
    assert np.all(z == 0.0)


@pytest.mark.parametrize("scheme", [init.xavier_uniform, init.xavier_normal])
def test_xavier_variance(scheme: init.Initializer) -> None:  # type: ignore[name-defined]
    fan_in, fan_out = 300, 200
    w = scheme((fan_in, fan_out), np.random.default_rng(0))
    expected_var = 2.0 / (fan_in + fan_out)
    assert w.shape == (fan_in, fan_out)
    assert np.isclose(w.var(), expected_var, rtol=0.1)
    assert np.isclose(w.mean(), 0.0, atol=0.01)


@pytest.mark.parametrize("scheme", [init.he_uniform, init.he_normal])
def test_he_variance(scheme: init.Initializer) -> None:  # type: ignore[name-defined]
    fan_in, fan_out = 300, 200
    w = scheme((fan_in, fan_out), np.random.default_rng(0))
    expected_var = 2.0 / fan_in
    assert w.shape == (fan_in, fan_out)
    assert np.isclose(w.var(), expected_var, rtol=0.1)
    assert np.isclose(w.mean(), 0.0, atol=0.01)


def test_initializers_are_reproducible() -> None:
    a = init.he_normal((10, 10), np.random.default_rng(7))
    b = init.he_normal((10, 10), np.random.default_rng(7))
    np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------


def test_module_registers_parameters() -> None:
    class M(Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = Tensor([1.0], requires_grad=True)
            self.frozen = Tensor([2.0])  # not a parameter

    m = M()
    params = list(m.parameters())
    assert len(params) == 1
    assert params[0] is m.w


def test_module_registers_children_recursively() -> None:
    class Inner(Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = Tensor([1.0], requires_grad=True)

    class Outer(Module):
        def __init__(self) -> None:
            super().__init__()
            self.inner = Inner()
            self.v = Tensor([2.0], requires_grad=True)

    outer = Outer()
    params = list(outer.parameters())
    assert len(params) == 2


def test_module_zero_grad() -> None:
    class M(Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = Tensor([2.0], requires_grad=True)

    m = M()
    (m.w * m.w).sum().backward()
    assert m.w.grad is not None
    m.zero_grad()
    assert m.w.grad is None


def test_module_without_super_init_raises() -> None:
    class Bad(Module):
        def __init__(self) -> None:
            self.w = Tensor([1.0], requires_grad=True)  # no super().__init__()

    with pytest.raises(RuntimeError, match="super"):
        Bad()


def test_module_forward_must_be_overridden() -> None:
    m = Module()
    with pytest.raises(NotImplementedError):
        m(Tensor([1.0]))


# ---------------------------------------------------------------------------
# Linear
# ---------------------------------------------------------------------------


def test_linear_shapes() -> None:
    layer = Linear(4, 3, rng=np.random.default_rng(0))
    x = Tensor(RNG.standard_normal((5, 4)))
    out = layer(x)
    assert out.shape == (5, 3)


def test_linear_parameter_count() -> None:
    layer = Linear(4, 3, rng=np.random.default_rng(0))
    assert len(list(layer.parameters())) == 2  # weight and bias


def test_linear_without_bias() -> None:
    layer = Linear(4, 3, rng=np.random.default_rng(0), bias=False)
    assert len(list(layer.parameters())) == 1
    x = Tensor(RNG.standard_normal((5, 4)))
    assert layer(x).shape == (5, 3)


def test_linear_matches_manual_computation() -> None:
    layer = Linear(2, 2, rng=np.random.default_rng(0))
    x_data = RNG.standard_normal((3, 2))
    out = layer(Tensor(x_data))
    assert layer.bias is not None
    expected = x_data @ layer.weight.data + layer.bias.data
    np.testing.assert_allclose(out.data, expected)


def test_linear_gradients_flow_to_parameters() -> None:
    layer = Linear(4, 3, rng=np.random.default_rng(0))
    x = Tensor(RNG.standard_normal((5, 4)))
    layer(x).mean().backward()
    assert layer.weight.grad is not None
    assert layer.bias is not None
    assert layer.bias.grad is not None
    assert layer.weight.grad.shape == layer.weight.shape
    assert layer.bias.grad.shape == layer.bias.shape


def test_linear_custom_initializer() -> None:
    layer = Linear(4, 3, rng=np.random.default_rng(0), weight_init=init.xavier_uniform)
    bound = np.sqrt(6.0 / (4 + 3))
    assert np.all(np.abs(layer.weight.data) <= bound)


def test_two_layer_network_trains_one_step() -> None:
    """End to end sanity check that a step of gradient descent reduces loss."""

    class MLP(Module):
        def __init__(self, rng: np.random.Generator) -> None:
            super().__init__()
            self.fc1 = Linear(4, 8, rng=rng)
            self.fc2 = Linear(8, 1, rng=rng)

        def forward(self, x: Tensor) -> Tensor:
            return self.fc2(self.fc1(x).relu())

    rng = np.random.default_rng(0)
    model = MLP(rng)
    x = Tensor(rng.standard_normal((16, 4)))
    y = Tensor(rng.standard_normal((16, 1)))

    def compute_loss() -> Tensor:
        diff = model(x) - y
        return (diff * diff).mean()

    loss_before = compute_loss()
    loss_before.backward()
    lr = 0.05
    for p in model.parameters():
        assert p.grad is not None
        p.data = p.data - lr * p.grad
    model.zero_grad()
    loss_after = compute_loss()
    assert float(loss_after.data) < float(loss_before.data)
