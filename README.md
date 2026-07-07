# spectra

![ci](https://github.com/RayaSergieva/ml-research-pipeline/actions/workflows/ci.yml/badge.svg)
![python](https://img.shields.io/badge/python-3.12-blue)
![license](https://img.shields.io/badge/license-MIT-green)
![typing](https://img.shields.io/badge/mypy-strict-informational)
![style](https://img.shields.io/badge/style-ruff-orange)
![tests](https://img.shields.io/badge/tests-219%20passing-brightgreen)

I wanted to know what actually happens inside a neural network while it
trains. Not at the level of "the loss goes down", but at the level of the
linear algebra - what the weight matrices do to the space passing through
them, and how that changes as the network learns. Frameworks like PyTorch
hide all of this behind very good engineering, so I built my own from
nothing but NumPy, derived every formula it relies on by hand, and then
pointed its own analysis instruments at the training process.

The result is a working deep learning framework, a set of spectral and
geometric measurement tools, and a research notebook that develops the
mathematics from the chain rule up and ends with an original,
honestly-reported experiment on initialization.

## Results at a glance

| Experiment | Model | Result |
|---|---|---|
| MNIST classification | 784-256-128-10 MLP, Adam | **97.9%** test accuracy in 5 epochs |
| Character-level language modeling | 2-layer decoder transformer | val loss 4.17 to **1.88** on tiny-shakespeare |
| Gradient correctness | every operation + full transformer block | matches finite differences to 1e-5 |
| Initialization study | 7 configurations x 5 seeds | dose-response curve, see below |

## What is inside

| Layer of the project | Contents | Where |
|---|---|---|
| Framework | `Tensor`, reverse-mode autograd, linear / embedding / layer norm / causal multi-head attention / transformer blocks, SGD + momentum, Adam, stabilized softmax cross-entropy | `src/spectra/` |
| Instruments | SVD tracker for weight spectra, Hessian top-eigenvalue probe via power iteration, filter-normalized loss slices, participation ratio, TwoNN intrinsic dimension, class separability | `src/spectra/analysis/` |
| Research notebook | full mathematical derivations, all experiments, all plots, references | `notebooks/spectra.ipynb` |
| Experiments | reproducible scripts with JSON-lines logging | `experiments/` |
| Tests | 219 tests, the core being finite-difference gradient checks | `tests/` |

## What I found

Watching the singular value spectra of the weight matrices during MNIST
training shows three consistent signatures. The largest singular values
grow steadily while the loss falls, the stable rank drops as each layer
concentrates its energy into a few directions (most sharply in the final
classification layer), and conditioning slowly deteriorates as the learned
anisotropy emerges. In short, training does not move weight matrices
around randomly - it pumps energy into a small set of directions that
matter for the task.

That observation raised a question I could actually test. If trained
networks end up with concentrated spectra, does *starting* concentrated
help? I built an initialization scheme that draws random orthogonal
singular vectors and imposes an explicit power-law spectrum with
concentration knob alpha, energy-matched to the He baseline so the
spectrum's shape is the only thing that varies.

| Scheme | Final test accuracy (mean of 5 seeds) |
|---|---|
| He normal (baseline) | 0.9744 |
| Xavier normal | 0.9741 |
| spectral, alpha = 0 (flat) | 0.9754 |
| spectral, alpha = 0.25 | **0.9757** |
| spectral, alpha = 0.5 | 0.9752 |
| spectral, alpha = 0.75 | 0.9731 |
| spectral, alpha = 1.0 | 0.9709 |

The curve peaks at mild concentration and falls monotonically after, and
the flat-spectrum variant already beats the i.i.d. baseline, which points
at the perfect initial conditioning rather than concentration as the
active ingredient (this connects to orthogonal initialization theory). The
effects are small at this scale and I report them as such - the full
discussion, error bars, and limitations are in section 7 of the notebook.

## Installation

The project uses [uv](https://docs.astral.sh/uv/) with Python 3.12.

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

Datasets download automatically on first use into `data/`.

| Command | What it does | Rough runtime on a laptop CPU |
|---|---|---|
| `uv run python experiments/train_mlp_mnist.py` | MNIST MLP with spectral tracking | a few minutes |
| `uv run python experiments/train_transformer_lm.py --steps 6000` | character-level Shakespeare LM | around an hour |
| `uv run python experiments/benchmark_init.py` | the initialization sweep, 7 configs x 5 seeds | 40-60 minutes |

Every run streams its measurements to a JSON-lines file under `runs/`,
written by the project's own zero-dependency logger, and the notebook
aggregates those files into the plots.

## Development

| Check | Command |
|---|---|
| Tests with coverage | `uv run pytest -n auto` |
| Strict type check | `uv run mypy` |
| Lint | `uv run ruff check` |
| Formatting | `uv run ruff format --check` |
| Enable git hooks | `uv run pre-commit install` |

All of these also run in CI on every push. The heart of the test suite is
gradient checking - every backward rule in the framework is compared
against central finite differences, including end to end through a
complete transformer block, because an autograd engine you have not
checked numerically is an autograd engine you should not trust.

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
topological descriptors), an ablation separating the orthogonality effect
from the concentration effect in the initialization study, an interactive
visualization lab, and a paper-style writeup.

## License

MIT. See [LICENSE](LICENSE).
