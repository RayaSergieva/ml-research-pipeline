"""Train an MLP classifier on MNIST with the spectra framework.

Run from the repository root:

    uv run python experiments/train_mlp_mnist.py

The script downloads MNIST on first use (about 11 MB), trains a two-hidden-
layer ReLU network with Adam, prints per-epoch metrics, and appends every
measurement to a JSON-lines file under runs/ for later analysis.

A held-out test set is evaluated once per epoch. With the default settings
the network reaches roughly 97-98 percent test accuracy in a few minutes on
a laptop CPU.
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
from numpy.typing import NDArray

from spectra import Tensor
from spectra.losses import softmax_cross_entropy
from spectra.nn import Linear, Module
from spectra.optim import Adam
from spectra.utils import RunLogger, iterate_minibatches, load_mnist

Array = NDArray[Any]


class MLP(Module):
    """784 -> 256 -> 128 -> 10 ReLU network."""

    def __init__(self, rng: np.random.Generator) -> None:
        super().__init__()
        self.fc1 = Linear(784, 256, rng=rng)
        self.fc2 = Linear(256, 128, rng=rng)
        self.fc3 = Linear(128, 10, rng=rng)

    def forward(self, x: Tensor) -> Tensor:
        h = self.fc1(x).relu()
        h = self.fc2(h).relu()
        return self.fc3(h)


def evaluate(model: MLP, x: Array, y: Array, batch_size: int = 1000) -> tuple[float, float]:
    """Return (mean loss, accuracy) over a dataset, in inference mode."""
    losses: list[float] = []
    correct = 0
    for start in range(0, len(x), batch_size):
        xb = x[start : start + batch_size]
        yb = y[start : start + batch_size]
        logits = model(Tensor(xb))
        losses.append(float(softmax_cross_entropy(logits, yb).data) * len(xb))
        correct += int((logits.data.argmax(axis=1) == yb).sum())
    return sum(losses) / len(x), correct / len(x)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-file", type=str, default="runs/mlp_mnist.jsonl")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    x_train, y_train, x_test, y_test = load_mnist()
    model = MLP(rng)
    optimizer = Adam(model.parameters(), lr=args.lr)

    step = 0
    with RunLogger(args.log_file) as logger:
        logger.log(step, event="config", **vars(args))
        for epoch in range(1, args.epochs + 1):
            for xb, yb in iterate_minibatches(x_train, y_train, args.batch_size, rng):
                optimizer.zero_grad()
                loss = softmax_cross_entropy(model(Tensor(xb)), yb)
                loss.backward()
                optimizer.step()
                step += 1
                if step % 100 == 0:
                    logger.log(step, event="train", loss=float(loss.data))
            test_loss, test_acc = evaluate(model, x_test, y_test)
            logger.log(step, event="epoch", epoch=epoch, test_loss=test_loss, test_acc=test_acc)
            print(f"epoch {epoch}  test loss {test_loss:.4f}  test accuracy {test_acc:.4f}")


if __name__ == "__main__":
    main()
