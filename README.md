# spectra

A linear-algebra-first deep learning framework, written from scratch in NumPy,
with built-in spectral and geometric analysis tools.

The framework is the substrate for an ongoing research project on how neural
networks reorganize themselves during training, viewed through the singular
value decomposition of their weight matrices and the eigenstructure of their
loss surfaces. The first concrete output of that investigation is a benchmark
study of spectrum-aware weight initialization on MLP and Transformer training.

## Status

Pre-alpha. The public API is unstable and will change without notice until
`v0.1.0`.

## Installation

The project uses [`uv`](https://docs.astral.sh/uv/) for environment and
dependency management.

```bash
git clone https://github.com/RayaSergieva/ml-research-pipeline.git
cd ml-research-pipeline
uv sync --all-extras
```

To activate the virtual environment in your shell:

```bash
source .venv/bin/activate
```

Or run commands without activating, using `uv run`:

```bash
uv run pytest
```

## Project layout

```
src/spectra/      Core framework (Tensor, autograd, nn modules, optimizers)
analysis/         Spectral, loss-landscape, and representation-geometry tools
experiments/      Reproducible training and benchmark scripts
notebooks/        The research notebook (main exam deliverable)
tests/            Test suite, including gradient checks
```

## Development

Tests, type-checking, and linting are enforced on every commit via
`pre-commit`, and on every push via GitHub Actions.

```bash
uv run pytest            # tests with coverage
uv run mypy              # type check
uv run ruff check        # lint
uv run ruff format       # format
```

To enable the pre-commit hooks locally:

```bash
uv run pre-commit install
```

## Roadmap

The project has two delivery horizons.

**Horizon 1** (the exam submission) ships an MLP and a small decoder-only
Transformer trained end to end on top of the framework, three analytical
instruments (SVD evolution, top-Hessian eigenvalue tracking, representation
geometry), and a controlled benchmark of spectrum-aware initialization against
the Xavier and He baselines.

**Horizon 2** extends the framework with higher-order automatic
differentiation, convolutional and state-space architectures, additional
analytical instruments (neural tangent kernel, information flow, topological
descriptors), and further methodological contributions arising from the
analysis. An interactive web demo and a written paper draft are part of this
horizon.

## License

MIT. See [`LICENSE`](LICENSE).
