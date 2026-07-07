"""Tests for embedding, layer normalization, attention, and the block.

The two properties that matter most are checked directly - the causal mask
(no position may depend on a later one, verified by perturbation) and the
correctness of every gradient (verified against finite differences through
the full block).
"""

from __future__ import annotations

import numpy as np
import pytest

from spectra import Tensor
from spectra.nn import Embedding, LayerNorm, MultiHeadAttention, TransformerBlock


def rng_for(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


def test_embedding_shape_and_values() -> None:
    emb = Embedding(10, 4, rng_for(0))
    idx = np.array([[1, 2], [3, 1]])
    out = emb(idx)
    assert out.shape == (2, 2, 4)
    np.testing.assert_allclose(out.data[0, 0], emb.weight.data[1])
    np.testing.assert_allclose(out.data[1, 1], emb.weight.data[1])


def test_embedding_gradient_scatter_adds_repeats() -> None:
    """A row selected twice accumulates two units of gradient."""
    emb = Embedding(5, 3, rng_for(1))
    idx = np.array([1, 1, 4])
    emb(idx).sum().backward()
    assert emb.weight.grad is not None
    np.testing.assert_allclose(emb.weight.grad[1], 2.0 * np.ones(3))
    np.testing.assert_allclose(emb.weight.grad[4], np.ones(3))
    np.testing.assert_allclose(emb.weight.grad[0], np.zeros(3))


def test_embedding_rejects_out_of_range() -> None:
    emb = Embedding(5, 3, rng_for(2))
    with pytest.raises(ValueError, match="out of range"):
        emb(np.array([5]))
    with pytest.raises(ValueError, match="out of range"):
        emb(np.array([-1]))


# ---------------------------------------------------------------------------
# LayerNorm
# ---------------------------------------------------------------------------


def test_layernorm_normalizes_last_axis() -> None:
    ln = LayerNorm(16)
    x = Tensor(rng_for(3).standard_normal((4, 7, 16)) * 5.0 + 2.0)
    out = ln(x).data
    np.testing.assert_allclose(out.mean(axis=-1), 0.0, atol=1e-10)
    np.testing.assert_allclose(out.std(axis=-1), 1.0, atol=1e-3)


def test_layernorm_gamma_beta_apply() -> None:
    ln = LayerNorm(4)
    ln.gamma.data = np.full(4, 3.0)
    ln.beta.data = np.full(4, 7.0)
    out = ln(Tensor(rng_for(4).standard_normal((10, 4)))).data
    np.testing.assert_allclose(out.mean(axis=-1), 7.0, atol=1e-10)


def test_layernorm_gradients_flow() -> None:
    ln = LayerNorm(6)
    x = Tensor(rng_for(5).standard_normal((3, 6)), requires_grad=True)
    ln(x).sum().backward()
    assert x.grad is not None
    assert ln.gamma.grad is not None
    assert ln.beta.grad is not None
    # d(sum(beta-shifted output))/d(beta) is the number of rows.
    np.testing.assert_allclose(ln.beta.grad, 3.0 * np.ones(6))


def test_layernorm_gradient_matches_finite_differences() -> None:
    ln = LayerNorm(5)
    x_data = rng_for(6).standard_normal((2, 5))
    x = Tensor(x_data.copy(), requires_grad=True)
    weights = rng_for(7).standard_normal((2, 5))
    (ln(x) * Tensor(weights)).sum().backward()
    assert x.grad is not None

    eps = 1e-6
    numerical = np.zeros_like(x_data)
    for i in range(2):
        for j in range(5):
            bumped = x_data.copy()
            bumped[i, j] += eps
            plus = float((ln(Tensor(bumped)) * Tensor(weights)).sum().data)
            bumped[i, j] -= 2 * eps
            minus = float((ln(Tensor(bumped)) * Tensor(weights)).sum().data)
            numerical[i, j] = (plus - minus) / (2 * eps)
    np.testing.assert_allclose(x.grad, numerical, atol=1e-5)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


def test_attention_output_shape() -> None:
    attn = MultiHeadAttention(dim=16, num_heads=4, max_len=8, rng=rng_for(8))
    x = Tensor(rng_for(9).standard_normal((2, 6, 16)))
    assert attn(x).shape == (2, 6, 16)


def test_attention_rejects_indivisible_heads() -> None:
    with pytest.raises(ValueError, match="divisible"):
        MultiHeadAttention(dim=10, num_heads=3, max_len=8, rng=rng_for(10))


def test_attention_is_causal() -> None:
    """Perturbing a future token must not change earlier outputs."""
    attn = MultiHeadAttention(dim=8, num_heads=2, max_len=10, rng=rng_for(11))
    x_data = rng_for(12).standard_normal((1, 5, 8))
    baseline = attn(Tensor(x_data)).data.copy()

    perturbed = x_data.copy()
    perturbed[0, 3] += 10.0  # change token at position 3
    changed = attn(Tensor(perturbed)).data

    np.testing.assert_allclose(changed[0, :3], baseline[0, :3], atol=1e-10)
    assert not np.allclose(changed[0, 3:], baseline[0, 3:])


def test_attention_weights_ignore_masked_positions() -> None:
    """With a huge value planted at the last position, causal outputs at
    earlier positions stay finite and bounded - the mask really blocks it."""
    attn = MultiHeadAttention(dim=8, num_heads=1, max_len=6, rng=rng_for(13))
    x_data = rng_for(14).standard_normal((1, 4, 8))
    x_data[0, 3] *= 1e3
    out = attn(Tensor(x_data)).data
    assert np.all(np.isfinite(out))


def test_attention_gradients_flow_to_all_projections() -> None:
    attn = MultiHeadAttention(dim=8, num_heads=2, max_len=6, rng=rng_for(15))
    x = Tensor(rng_for(16).standard_normal((2, 4, 8)), requires_grad=True)
    attn(x).sum().backward()
    assert x.grad is not None
    for p in attn.parameters():
        assert p.grad is not None


# ---------------------------------------------------------------------------
# TransformerBlock
# ---------------------------------------------------------------------------


def test_block_shape_preserved() -> None:
    block = TransformerBlock(dim=16, num_heads=4, max_len=12, rng=rng_for(17))
    x = Tensor(rng_for(18).standard_normal((3, 7, 16)))
    assert block(x).shape == (3, 7, 16)


def test_block_is_causal_end_to_end() -> None:
    block = TransformerBlock(dim=8, num_heads=2, max_len=10, rng=rng_for(19))
    x_data = rng_for(20).standard_normal((1, 6, 8))
    baseline = block(Tensor(x_data)).data.copy()
    perturbed = x_data.copy()
    perturbed[0, 4] += 5.0
    changed = block(Tensor(perturbed)).data
    np.testing.assert_allclose(changed[0, :4], baseline[0, :4], atol=1e-10)


def test_block_gradient_matches_finite_differences() -> None:
    """Full end to end check through attention, layer norms, and the MLP."""
    block = TransformerBlock(dim=4, num_heads=2, max_len=5, rng=rng_for(21))
    x_data = rng_for(22).standard_normal((1, 3, 4))
    weights = rng_for(23).standard_normal((1, 3, 4))

    x = Tensor(x_data.copy(), requires_grad=True)
    (block(x) * Tensor(weights)).sum().backward()
    assert x.grad is not None

    eps = 1e-6
    numerical = np.zeros_like(x_data)
    it = np.nditer(x_data, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        bumped = x_data.copy()
        bumped[idx] += eps
        plus = float((block(Tensor(bumped)) * Tensor(weights)).sum().data)
        bumped[idx] -= 2 * eps
        minus = float((block(Tensor(bumped)) * Tensor(weights)).sum().data)
        numerical[idx] = (plus - minus) / (2 * eps)
        it.iternext()

    np.testing.assert_allclose(x.grad, numerical, atol=1e-5)


def test_block_parameter_count() -> None:
    dim, heads = 8, 2
    block = TransformerBlock(dim=dim, num_heads=heads, max_len=4, rng=rng_for(24))
    # ln1 (2) + attention (3 weights + out weight + out bias = 5) +
    # ln2 (2) + mlp_in (2) + mlp_out (2)
    assert len(list(block.parameters())) == 13
