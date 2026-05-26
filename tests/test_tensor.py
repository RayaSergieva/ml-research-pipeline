"""Tests for the :class:`spectra.Tensor` class."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from spectra import Tensor
from spectra._backend import default_dtype

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_from_python_scalar() -> None:
    t = Tensor(3.14)
    assert t.shape == ()
    assert t.ndim == 0
    assert t.size == 1
    assert t.dtype == default_dtype()


def test_from_python_list() -> None:
    t = Tensor([1.0, 2.0, 3.0])
    assert t.shape == (3,)
    assert t.dtype == default_dtype()


def test_from_nested_list() -> None:
    t = Tensor([[1.0, 2.0], [3.0, 4.0]])
    assert t.shape == (2, 2)
    assert t.ndim == 2
    assert t.size == 4


def test_from_numpy_array() -> None:
    arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    t = Tensor(arr)
    assert t.shape == (3, 4)
    assert (
        t.dtype == default_dtype()
    ), "floating inputs without an explicit dtype must be promoted to the framework default"


def test_from_numpy_array_with_explicit_dtype() -> None:
    arr = np.arange(6, dtype=np.float64).reshape(2, 3)
    t = Tensor(arr, dtype=np.float32)
    assert t.dtype == np.float32
    assert t.shape == (2, 3)


def test_integer_input_keeps_integer_dtype() -> None:
    t = Tensor([1, 2, 3])
    assert np.issubdtype(
        t.dtype, np.integer
    ), "integer inputs without an explicit dtype must keep their integer dtype"


def test_from_tensor_copies_metadata() -> None:
    source = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    copy = Tensor(source)
    assert copy.shape == source.shape
    assert copy.dtype == source.dtype
    # requires_grad does NOT propagate; it must be passed explicitly.
    assert copy.requires_grad is False


def test_requires_grad_defaults_to_false() -> None:
    assert Tensor([1.0]).requires_grad is False


def test_requires_grad_can_be_set() -> None:
    assert Tensor([1.0], requires_grad=True).requires_grad is True


def test_grad_defaults_to_none() -> None:
    assert Tensor([1.0]).grad is None


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("data", "expected_shape", "expected_ndim", "expected_size"),
    [
        (5.0, (), 0, 1),
        ([1.0], (1,), 1, 1),
        ([1.0, 2.0, 3.0], (3,), 1, 3),
        ([[1.0, 2.0], [3.0, 4.0]], (2, 2), 2, 4),
        ([[[1.0]]], (1, 1, 1), 3, 1),
    ],
)
def test_properties_match_data(
    data: object,
    expected_shape: tuple[int, ...],
    expected_ndim: int,
    expected_size: int,
) -> None:
    t = Tensor(data)  # type: ignore[arg-type]
    assert t.shape == expected_shape
    assert t.ndim == expected_ndim
    assert t.size == expected_size


def test_data_property_returns_underlying_array() -> None:
    arr = np.array([1.0, 2.0, 3.0])
    t = Tensor(arr)
    assert isinstance(t.data, np.ndarray)


def test_numpy_returns_a_copy() -> None:
    t = Tensor([1.0, 2.0, 3.0])
    a = t.numpy()
    a[0] = 999.0
    assert t.data[0] == 1.0, "Tensor.numpy() must not alias the underlying array"


# ---------------------------------------------------------------------------
# Mutability
# ---------------------------------------------------------------------------


def test_data_can_be_overwritten_with_same_shape() -> None:
    t = Tensor([1.0, 2.0, 3.0])
    t.data = np.array([4.0, 5.0, 6.0])
    np.testing.assert_array_equal(t.data, [4.0, 5.0, 6.0])


def test_data_overwrite_with_different_shape_raises() -> None:
    t = Tensor([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="shape"):
        t.data = np.array([1.0, 2.0])


def test_data_overwrite_preserves_dtype() -> None:
    t = Tensor([1.0, 2.0, 3.0], dtype=np.float32)
    t.data = np.array([4.0, 5.0, 6.0], dtype=np.float64)
    assert t.dtype == np.float32, "overwriting data must not change the tensor's dtype"


# ---------------------------------------------------------------------------
# Equality and hashing
# ---------------------------------------------------------------------------


def test_equality_for_identical_tensors() -> None:
    assert Tensor([1.0, 2.0]) == Tensor([1.0, 2.0])


def test_inequality_for_different_data() -> None:
    assert Tensor([1.0, 2.0]) != Tensor([1.0, 3.0])


def test_inequality_for_different_shape() -> None:
    assert Tensor([1.0, 2.0]) != Tensor([[1.0, 2.0]])


def test_inequality_for_different_dtype() -> None:
    assert Tensor([1.0, 2.0], dtype=np.float64) != Tensor([1.0, 2.0], dtype=np.float32)


def test_inequality_for_different_requires_grad() -> None:
    assert Tensor([1.0], requires_grad=True) != Tensor([1.0], requires_grad=False)


def test_nan_tensors_are_structurally_equal() -> None:
    """NaN equals NaN under structural equality, deviating from IEEE 754.

    This matches :func:`numpy.array_equal`'s ``equal_nan=True`` behaviour.
    """
    assert Tensor([float("nan")]) == Tensor([float("nan")])


def test_nan_position_matters_for_equality() -> None:
    assert Tensor([float("nan"), 1.0]) != Tensor([1.0, float("nan")])


def test_integer_tensor_equality() -> None:
    assert Tensor([1, 2, 3]) == Tensor([1, 2, 3])
    assert Tensor([1, 2, 3]) != Tensor([1, 2, 4])


def test_equality_with_non_tensor_returns_notimplemented() -> None:
    assert (Tensor([1.0]) == [1.0]) is False


def test_tensor_is_unhashable() -> None:
    with pytest.raises(TypeError):
        hash(Tensor([1.0]))


# ---------------------------------------------------------------------------
# Repr
# ---------------------------------------------------------------------------


def test_repr_includes_class_name() -> None:
    assert repr(Tensor([1.0])).startswith("Tensor(")


def test_repr_includes_dtype() -> None:
    assert "dtype=" in repr(Tensor([1.0]))


def test_repr_includes_requires_grad_when_true() -> None:
    assert "requires_grad=True" in repr(Tensor([1.0], requires_grad=True))


def test_repr_omits_requires_grad_when_false() -> None:
    assert "requires_grad" not in repr(Tensor([1.0]))


def test_repr_truncates_large_arrays() -> None:
    t = Tensor(list(range(1000)))
    r = repr(t)
    # numpy's truncation marker is an ellipsis
    assert "..." in r


# ---------------------------------------------------------------------------
# Property-based tests
# ---------------------------------------------------------------------------


@given(
    hnp.arrays(
        dtype=hnp.floating_dtypes(sizes=(32, 64)),
        shape=hnp.array_shapes(min_dims=0, max_dims=4, min_side=0, max_side=5),
    )
)
def test_construction_preserves_shape(arr: np.ndarray) -> None:
    """For any numpy array of any floating dtype and shape, Tensor preserves shape."""
    t = Tensor(arr)
    assert t.shape == arr.shape
    assert t.size == arr.size
    assert t.ndim == arr.ndim


@given(
    hnp.arrays(
        dtype=hnp.floating_dtypes(sizes=(32, 64)),
        shape=hnp.array_shapes(min_dims=1, max_dims=3, min_side=1, max_side=4),
    )
)
def test_numpy_roundtrip_preserves_values(arr: np.ndarray) -> None:
    """Tensor(arr).numpy() yields data equal to arr (modulo dtype promotion)."""
    t = Tensor(arr)
    np.testing.assert_array_equal(t.numpy(), arr.astype(t.dtype))


@given(st.integers(min_value=-(2**31), max_value=2**31 - 1))
def test_python_integer_scalars_remain_integer(n: int) -> None:
    t = Tensor(n)
    assert np.issubdtype(t.dtype, np.integer)
    assert int(t.data) == n


@given(
    hnp.arrays(
        dtype=np.float64,
        shape=hnp.array_shapes(min_dims=1, max_dims=3, min_side=1, max_side=4),
    )
)
def test_equality_is_reflexive(arr: np.ndarray) -> None:
    t = Tensor(arr)
    assert t == Tensor(arr)
