"""Controlled comparison of initialization schemes on MNIST.

Run from the repository root:

    uv run python experiments/benchmark_init.py

The spectral scheme is swept over five concentration levels alpha in
{0, 0.25, 0.5, 0.75, 1.0} alongside the He and Xavier baselines, and the
same MLP is trained on MNIST for a fixed budget across five seeds per
configuration, giving a dose-response curve in alpha rather than isolated
points. Everything except the initial weight matrices is held constant -
architecture, optimizer, learning rate, batch order (seeded per run), and
epochs - so any difference in outcome is attributable to initialization.

Reported per run - the test accuracy and loss after each epoch, plus the
training loss after the first 100 steps as an early-progress measure. All
records land in a JSON-lines file for the notebook to aggregate; the
notebook reports means and standard deviations across seeds and draws its
conclusions from those, including negative ones. Comparing schemes with
matched energy (see spectra.init.spectral) means the study isolates the
*distribution* of the initial spectrum as the only manipulated variable.
"""

from __future__ import annotations

import argparse
from functools import partial

import numpy as np

from spectra import Tensor, init
from spectra.losses import softmax_cross_entropy
from spectra.nn import Linear, Module
from spectra.optim import Adam
from spectra.utils import RunLogger, iterate_minibatches, load_mnist


class MLP(Module):
    """The same 784-256-128-10 architecture as the main MNIST experiment."""

    def __init__(self, rng: np.random.Generator, weight_init) -> None:  # type: ignore[no-untyped-def]
        super().__init__()
        self.fc1 = Linear(784, 256, rng=rng, weight_init=weight_init)
        self.fc2 = Linear(256, 128, rng=rng, weight_init=weight_init)
        self.fc3 = Linear(128, 10, rng=rng, weight_init=weight_init)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc3(self.fc2(self.fc1(x).relu()).relu())


def evaluate(model: MLP, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    losses = 0.0
    correct = 0
    for start in range(0, len(x), 1000):
        xb, yb = x[start : start + 1000], y[start : start + 1000]
        logits = model(Tensor(xb))
        losses += float(softmax_cross_entropy(logits, yb).data) * len(xb)
        correct += int((logits.data.argmax(axis=1) == yb).sum())
    return losses / len(x), correct / len(x)


SCHEMES = {
    "he_normal": init.he_normal,
    "xavier_normal": init.xavier_normal,
    "spectral_a000": partial(init.spectral, alpha=0.0),
    "spectral_a025": partial(init.spectral, alpha=0.25),
    "spectral_a050": partial(init.spectral, alpha=0.5),
    "spectral_a075": partial(init.spectral, alpha=0.75),
    "spectral_a100": partial(init.spectral, alpha=1.0),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log-file", type=str, default="runs/benchmark_init.jsonl")
    args = parser.parse_args()

    x_train, y_train, x_test, y_test = load_mnist()

    with RunLogger(args.log_file) as logger:
        logger.log(0, event="config", schemes=list(SCHEMES), **vars(args))
        for scheme_name, weight_init in SCHEMES.items():
            for seed in args.seeds:
                rng = np.random.default_rng(seed)

                def scheme(shape: tuple[int, int], generator: np.random.Generator) -> np.ndarray:
                    return weight_init(shape, generator)  # noqa: B023

                model = MLP(rng, scheme)
                optimizer = Adam(model.parameters(), lr=args.lr)
                step = 0
                for epoch in range(1, args.epochs + 1):
                    for xb, yb in iterate_minibatches(x_train, y_train, args.batch_size, rng):
                        optimizer.zero_grad()
                        loss = softmax_cross_entropy(model(Tensor(xb)), yb)
                        loss.backward()
                        optimizer.step()
                        step += 1
                        if step == 100:
                            logger.log(
                                step,
                                event="early",
                                scheme=scheme_name,
                                seed=seed,
                                loss=float(loss.data),
                            )
                    test_loss, test_acc = evaluate(model, x_test, y_test)
                    logger.log(
                        step,
                        event="epoch",
                        scheme=scheme_name,
                        seed=seed,
                        epoch=epoch,
                        test_loss=test_loss,
                        test_acc=test_acc,
                    )
                    print(
                        f"{scheme_name} seed {seed} epoch {epoch}  "
                        f"test loss {test_loss:.4f}  test acc {test_acc:.4f}"
                    )
                step = 0

    print(f"done, results in {args.log_file}")


if __name__ == "__main__":
    main()
