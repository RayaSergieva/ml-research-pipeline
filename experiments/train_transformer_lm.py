"""Train a character-level transformer language model with spectra.

Run from the repository root:

    uv run python experiments/train_transformer_lm.py

The script downloads the tiny-shakespeare corpus (about 1 MB) on first use,
trains a small decoder-only transformer at character level, logs every
measurement to a JSON-lines file under runs/, captures the spectra of all
weight matrices during training, and prints a sample of generated text at
the end of every epoch. Character-level modeling keeps the vocabulary tiny
(about 65 symbols) so the whole experiment runs on a laptop CPU in tens of
minutes with the default settings.
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import numpy as np

from spectra.analysis import SpectralTracker
from spectra.nn.transformer import DecoderLM
from spectra.optim import Adam
from spectra.utils import RunLogger

_CORPUS_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


def load_corpus(cache_dir: str | Path = "data/shakespeare") -> str:
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / "input.txt"
    if not target.exists():
        print(f"downloading {_CORPUS_URL}")
        urllib.request.urlretrieve(_CORPUS_URL, target)
    return target.read_text(encoding="utf-8")


class CharTokenizer:
    """Bijective character-to-integer codec built from the corpus."""

    def __init__(self, text: str) -> None:
        self.chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}

    def __len__(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> np.ndarray:
        return np.array([self.stoi[c] for c in text], dtype=np.int64)

    def decode(self, tokens: np.ndarray) -> str:
        return "".join(self.chars[int(t)] for t in tokens)


def sample_batch(
    data: np.ndarray,
    batch_size: int,
    seq_len: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Random windows (x, y) with y the one-step shift of x."""
    starts = rng.integers(0, len(data) - seq_len - 1, size=batch_size)
    x = np.stack([data[s : s + seq_len] for s in starts])
    y = np.stack([data[s + 1 : s + seq_len + 1] for s in starts])
    return x, y


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample-every", type=int, default=500)
    parser.add_argument("--spectral-every", type=int, default=100)
    parser.add_argument("--log-file", type=str, default="runs/transformer_lm.jsonl")
    parser.add_argument("--spectra-file", type=str, default="runs/transformer_lm_spectra.npz")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    text = load_corpus()
    tokenizer = CharTokenizer(text)
    data = tokenizer.encode(text)
    split = int(0.9 * len(data))
    train_data, val_data = data[:split], data[split:]
    print(f"corpus {len(data)} chars, vocab {len(tokenizer)}")

    model = DecoderLM(
        vocab_size=len(tokenizer),
        max_len=args.seq_len,
        dim=args.dim,
        num_heads=args.heads,
        num_layers=args.layers,
        rng=rng,
    )
    optimizer = Adam(model.parameters(), lr=args.lr)
    tracker = SpectralTracker(model)

    with RunLogger(args.log_file) as logger:
        logger.log(0, event="config", vocab=len(tokenizer), **vars(args))
        for step in range(1, args.steps + 1):
            x, y = sample_batch(train_data, args.batch_size, args.seq_len, rng)
            optimizer.zero_grad()
            loss = model.loss(x, y)
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                logger.log(step, event="train", loss=float(loss.data))
            if args.spectral_every and step % args.spectral_every == 0:
                for snap in tracker.capture(step):
                    logger.log(step, event="spectrum", name=snap.name, **snap.summary())
            if step % args.sample_every == 0 or step == args.steps:
                vx, vy = sample_batch(val_data, args.batch_size, args.seq_len, rng)
                val_loss = float(model.loss(vx, vy).data)
                logger.log(step, event="val", val_loss=val_loss)
                prompt = tokenizer.encode("\n")
                sample = model.generate(prompt, 200, rng, temperature=0.8)
                print(f"step {step}  train loss {float(loss.data):.4f}  val loss {val_loss:.4f}")
                print("--- sample ---")
                print(tokenizer.decode(sample))
                print("---")

    arrays = {f"{s.name}_step{s.step}": s.singular_values for s in tracker.snapshots}
    np.savez_compressed(args.spectra_file, **arrays)
    print(f"saved {len(tracker.snapshots)} spectral snapshots to {args.spectra_file}")


if __name__ == "__main__":
    main()
