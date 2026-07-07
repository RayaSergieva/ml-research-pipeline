"""The decoder-only transformer language model.

The model is the GPT recipe at miniature scale - token embeddings plus
learned position embeddings feed a stack of pre-norm transformer blocks, a
final layer normalization, and a linear head producing next-token logits.
Autoregressive training minimizes the cross-entropy of position t's logits
against token t+1, so one forward pass over a window of length T yields T
prediction problems at once; the causal mask inside attention is what makes
those T problems honest, letting position t see only positions <= t.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from spectra.losses import softmax_cross_entropy
from spectra.nn.attention import TransformerBlock
from spectra.nn.linear import Linear
from spectra.nn.module import Module
from spectra.nn.normalization import Embedding, LayerNorm
from spectra.tensor import Tensor

Array = NDArray[Any]


class DecoderLM(Module):
    """A small autoregressive transformer language model.

    Parameters
    ----------
    vocab_size
        Number of distinct tokens.
    max_len
        Longest context window the model supports.
    dim
        Model width.
    num_heads
        Attention heads per block.
    num_layers
        Number of transformer blocks.
    rng
        Generator for all initializations.
    """

    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        dim: int,
        num_heads: int,
        num_layers: int,
        rng: np.random.Generator,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.token_embedding = Embedding(vocab_size, dim, rng)
        self.position_embedding = Embedding(max_len, dim, rng)
        for i in range(num_layers):
            setattr(self, f"block{i}", TransformerBlock(dim, num_heads, max_len, rng))
        self.num_layers = num_layers
        self.final_norm = LayerNorm(dim)
        self.head = Linear(dim, vocab_size, rng=rng, bias=False)

    def forward(self, tokens: Array) -> Tensor:
        """Logits of shape (batch, seq_len, vocab_size) for integer tokens."""
        _, seq_len = tokens.shape
        if seq_len > self.max_len:
            msg = f"sequence length {seq_len} exceeds max_len {self.max_len}"
            raise ValueError(msg)
        positions = np.arange(seq_len)
        h = self.token_embedding(tokens) + self.position_embedding(positions)
        for i in range(self.num_layers):
            block: TransformerBlock = getattr(self, f"block{i}")
            h = block(h)
        return self.head(self.final_norm(h))

    def loss(self, tokens: Array, targets: Array) -> Tensor:
        """Mean next-token cross-entropy over every position in the batch."""
        logits = self.forward(tokens)
        batch, seq_len, vocab = logits.shape
        return softmax_cross_entropy(logits.reshape(batch * seq_len, vocab), targets.reshape(-1))

    def generate(
        self,
        prompt: Array,
        num_tokens: int,
        rng: np.random.Generator,
        temperature: float = 1.0,
    ) -> Array:
        """Sample a continuation of an integer prompt vector.

        At each step the context is truncated to the last ``max_len`` tokens,
        logits for the final position are divided by the temperature, and the
        next token is drawn from the resulting softmax. Temperature 0 or very
        small approaches greedy decoding.
        """
        tokens = list(prompt.tolist())
        for _ in range(num_tokens):
            context = np.array(tokens[-self.max_len :])[None, :]
            logits = self.forward(context).data[0, -1]
            if temperature <= 0:
                tokens.append(int(np.argmax(logits)))
                continue
            scaled = logits / temperature
            scaled -= scaled.max()
            probs = np.exp(scaled)
            probs /= probs.sum()
            tokens.append(int(rng.choice(len(probs), p=probs)))
        return np.array(tokens)
