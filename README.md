# spectra

A linear-algebra-first deep learning framework, written from scratch in
NumPy, with built-in spectral and geometric analysis instruments and a
research notebook deriving every component from first principles.

The framework trains real models - a multilayer perceptron reaches 97.9%
test accuracy on MNIST and a small decoder-only transformer learns
character-level structure from raw Shakespeare text - while every gradient
it computes is validated against finite differences and every design
decision is derived mathematically in the accompanying notebook.

## What is inside

**The framework** (`src/spectra/`). A `Tensor` type wrapping NumPy arrays, a
reverse-mode automatic differentiation engine built on vector-Jacobian
products, neural network modules (linear, embedding, layer normalization,
causal multi-head attention, transformer blocks, a full decoder LM),
softmax cross-entropy with the stabilized log-sum-exp, SGD with momentum
and Adam, and initialization schemes including Xavier, He, and an original
spectrum-aware scheme.

**The instruments** (`src/spectra/analysis/`). A spectral tracker recording
the full singular value decomposition of every weight matrix during
training, a loss landscape probe measuring the top Hessian eigenvalue by
power iteration on finite-difference Hessian-vector products plus
filter-normalized 2D loss slices, and a representation geometry suite with
participation ratio, TwoNN intrinsic dimension, and Fisher-style class
separability.

**The research notebook** (`notebooks/spectra.ipynb`). The full report -
matrix calculus and the derivation of reverse-mode differentiation,
variance-propagation initialization theory, optimization on quadratics and
the role of conditioning, the MNIST experiment watched through its spectra,
and a controlled, energy-matched study of spectrum-aware initialization
with honestly reported mixed results.

**The experiments** (`experiments/`). Reproducible scripts for MLP training
on MNIST with spectral tracking, character-level transformer training on
tiny-shakespeare, and the initialization benchmark. All experiment
measurements stream to JSON-lines files under `runs/` via a zero-dependency
logger.

## Findings so far

Tracking the SVD of every weight matrix during MNIST training shows three
consistent signatures. Dominant singular values grow steadily while the
loss falls, stable rank decreases as layers concentrate their energy into
few directions (most strongly in the final classification layer), and
conditioning deteriorates moderately as the learned anisotropy emerges.

The follow-up hypothesis - that initializing with a concentrated spectrum
helps - was tested with total energy matched to the He baseline so only the
spectrum's shape varied. Mild concentration performs on par with He with a
positive hint; heavy concentration consistently slows early training and
finishes worse. Details, plots, and limitations are in section 7 of the
notebook.

## Installation

The project uses [`uv`](https://docs.astral.sh/uv/) for environment and
dependency management, with Python 3.12.

```bash
git clone https://github.com/RayaSergieva/ml-research-pipeline.git
cd ml-research-pipeline
uv sync --all-extras
```

## Quickstart

```python
import numpy as np
from spectra import Tensor
from spectra.nn import Linear, Module
from spectra.optim import Adam
from spectra.losses import softmax_cross_entropy

rng = np.random.default_rng(0)

class Net(Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = Linear(784, 128, rng=rng)
        self.fc2 = Linear(128, 10, rng=rng)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.fc1(x).relu())

model = Net()
optimizer = Adam(model.parameters(), lr=1e-3)

logits = model(Tensor(rng.standard_normal((32, 784))))
loss = softmax_cross_entropy(logits, rng.integers(0, 10, 32))
loss.backward()
optimizer.step()
```

## Reproducing the experiments

```bash
uv run python experiments/train_mlp_mnist.py        # MNIST MLP with spectral tracking
uv run python experiments/train_transformer_lm.py  # character-level Shakespeare LM
uv run python experiments/benchmark_init.py        # initialization study
```

Datasets download automatically on first use (MNIST about 11 MB,
tiny-shakespeare about 1 MB) into `data/`.

## Development

Tests, type-checking, and linting run on every commit via `pre-commit` and
on every push via GitHub Actions. The test suite has over 200 tests; its
core is finite-difference gradient checking of every differentiable
operation, including end to end through a full transformer block.

```bash
uv run pytest -n auto       # tests with coverage
uv run mypy                 # strict type check
uv run ruff check           # lint
uv run ruff format --check  # formatting
uv run pre-commit install   # enable the hooks locally
```

## Project layout

```
src/spectra/           Core framework
src/spectra/nn/        Layers, attention, transformer, module system
src/spectra/analysis/  Spectral, landscape, and geometry instruments
src/spectra/utils/     MNIST loader, minibatching, JSON-lines logger
experiments/           Reproducible training and benchmark scripts
notebooks/             The research notebook (main deliverable)
tests/                 Test suite with gradient checks
```

## Road map

Ongoing work extends the framework with convolutional and state-space
architectures, higher-order automatic differentiation for exact Hessian
analysis, further instruments (neural tangent kernel, information flow,
topological descriptors), a deeper sweep of the spectrum-aware
initialization study across architectures and training budgets, an
interactive visualization lab, and a paper-style writeup.

## License

MIT. See [`LICENSE`](LICENSE).
