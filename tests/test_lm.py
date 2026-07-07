"""Tests for the decoder language model."""

from __future__ import annotations

import numpy as np
import pytest

from spectra.nn.transformer import DecoderLM


def rng_for(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def tiny_model(seed: int = 0, vocab: int = 7, max_len: int = 8) -> DecoderLM:
    return DecoderLM(
        vocab_size=vocab,
        max_len=max_len,
        dim=8,
        num_heads=2,
        num_layers=2,
        rng=rng_for(seed),
    )


def test_logits_shape() -> None:
    model = tiny_model()
    tokens = rng_for(1).integers(0, 7, size=(3, 5))
    assert model(tokens).shape == (3, 5, 7)


def test_rejects_sequences_beyond_max_len() -> None:
    model = tiny_model(max_len=4)
    with pytest.raises(ValueError, match="max_len"):
        model(np.zeros((1, 5), dtype=np.int64))


def test_model_is_causal() -> None:
    model = tiny_model()
    tokens = rng_for(2).integers(0, 7, size=(1, 6))
    baseline = model(tokens).data.copy()
    perturbed = tokens.copy()
    perturbed[0, 4] = (perturbed[0, 4] + 1) % 7
    changed = model(perturbed).data
    np.testing.assert_allclose(changed[0, :4], baseline[0, :4], atol=1e-10)


def test_initial_loss_near_uniform_entropy() -> None:
    """Before training the model should be roughly a uniform guesser, so the
    loss starts near log(vocab)."""
    model = tiny_model(vocab=7)
    x = rng_for(3).integers(0, 7, size=(8, 6))
    y = rng_for(4).integers(0, 7, size=(8, 6))
    loss = float(model.loss(x, y).data)
    assert loss == pytest.approx(np.log(7), rel=0.5)


def test_loss_decreases_with_training() -> None:
    """A few Adam steps on one fixed batch must reduce the loss - the full
    stack (embeddings, blocks, head, loss, optimizer) working together."""
    from spectra.optim import Adam

    model = tiny_model()
    rng = rng_for(5)
    x = rng.integers(0, 7, size=(8, 6))
    y = np.roll(x, -1, axis=1)
    optimizer = Adam(model.parameters(), lr=1e-2)

    first = float(model.loss(x, y).data)
    for _ in range(30):
        optimizer.zero_grad()
        model.loss(x, y).backward()
        optimizer.step()
    final = float(model.loss(x, y).data)
    assert final < first / 2


def test_generate_length_and_range() -> None:
    model = tiny_model()
    prompt = np.array([1, 2, 3])
    out = model.generate(prompt, num_tokens=10, rng=rng_for(6))
    assert out.shape == (13,)
    np.testing.assert_array_equal(out[:3], prompt)
    assert np.all((out >= 0) & (out < 7))


def test_generate_greedy_is_deterministic() -> None:
    model = tiny_model()
    prompt = np.array([1, 2])
    a = model.generate(prompt, 8, rng_for(7), temperature=0.0)
    b = model.generate(prompt, 8, rng_for(8), temperature=0.0)
    np.testing.assert_array_equal(a, b)


def test_generate_respects_context_window() -> None:
    """Prompts longer than max_len must still work via truncation."""
    model = tiny_model(max_len=4)
    prompt = np.array([1, 2, 3, 4, 5, 6]) % 7
    out = model.generate(prompt, 3, rng_for(9))
    assert out.shape == (9,)
