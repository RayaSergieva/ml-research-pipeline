"""Multi-head self-attention and the decoder transformer block.

Attention is the densest piece of linear algebra in modern deep learning,
and it decomposes entirely into operations the framework already
differentiates - projections, batched matrix products, a scaled softmax, and
reshapes. Writing S = Q K^T / sqrt(d_k), the layer computes

    Attention(Q, K, V) = softmax(S + M) V

per head, where M is the causal mask, 0 on and below the diagonal and a
large negative constant above it, so each position attends only to itself
and the past. The 1/sqrt(d_k) factor keeps the entries of S at unit variance
when Q and K have unit-variance entries - without it the softmax saturates
as d_k grows and gradients vanish, the same variance argument as
initialization theory applied inside the layer.

Head splitting is a reshape to (batch, heads, time, head_dim), so the h
heads are h independent attention maps computed in one batched matmul.
"""

from __future__ import annotations

import numpy as np

from spectra.nn.linear import Linear
from spectra.nn.module import Module
from spectra.nn.normalization import LayerNorm
from spectra.tensor import Tensor


class MultiHeadAttention(Module):
    """Causal multi-head self-attention.

    Parameters
    ----------
    dim
        Model width; must be divisible by ``num_heads``.
    num_heads
        Number of attention heads; each works in ``dim // num_heads``
        dimensions.
    max_len
        Longest sequence the layer will see; sets the causal mask size.
    rng
        Generator for the projection initializations.
    """

    def __init__(self, dim: int, num_heads: int, max_len: int, rng: np.random.Generator) -> None:
        super().__init__()
        if dim % num_heads != 0:
            msg = f"dim {dim} is not divisible by num_heads {num_heads}"
            raise ValueError(msg)
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = Linear(dim, dim, rng=rng, bias=False)
        self.k_proj = Linear(dim, dim, rng=rng, bias=False)
        self.v_proj = Linear(dim, dim, rng=rng, bias=False)
        self.out_proj = Linear(dim, dim, rng=rng)
        mask = np.triu(np.full((max_len, max_len), -1e9), k=1)
        self._mask = mask  # additive causal mask, not a parameter

    def forward(self, x: Tensor) -> Tensor:
        batch, seq_len, _ = x.shape

        def split_heads(t: Tensor) -> Tensor:
            return t.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        q = split_heads(self.q_proj(x))
        k = split_heads(self.k_proj(x))
        v = split_heads(self.v_proj(x))

        scale = 1.0 / np.sqrt(self.head_dim)
        scores = q.bmm(k.transpose(0, 1, 3, 2)) * scale
        scores = scores + self._mask[:seq_len, :seq_len]
        weights = scores.softmax(axis=-1)

        context = weights.bmm(v)
        merged = context.transpose(0, 2, 1, 3).reshape(batch, seq_len, self.dim)
        return self.out_proj(merged)


class TransformerBlock(Module):
    """Pre-norm decoder block, x + Attn(LN(x)) then x + MLP(LN(x)).

    The pre-norm arrangement (normalize before each sublayer, residual
    around it) keeps the residual stream an identity path, which is the
    variant that trains stably without warmup at these scales. The MLP
    expands by a factor of 4 and uses GELU, the GPT-2 recipe.
    """

    def __init__(self, dim: int, num_heads: int, max_len: int, rng: np.random.Generator) -> None:
        super().__init__()
        self.ln1 = LayerNorm(dim)
        self.attention = MultiHeadAttention(dim, num_heads, max_len, rng)
        self.ln2 = LayerNorm(dim)
        self.mlp_in = Linear(dim, 4 * dim, rng=rng)
        self.mlp_out = Linear(4 * dim, dim, rng=rng)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attention(self.ln1(x))
        return x + self.mlp_out(self.mlp_in(self.ln2(x)).gelu())
