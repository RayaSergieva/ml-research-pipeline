"""Dataset loading utilities.

MNIST is distributed in the IDX binary format described on the original
dataset page. Images are stored as a magic number, the item count, the two
image dimensions, then raw unsigned bytes; labels the same without the
dimensions. The parser below reads that format directly, avoiding any
dataset-library dependency, in keeping with the from-scratch character of
the project.

Files are downloaded once into a local cache directory and reused.
"""

from __future__ import annotations

import gzip
import struct
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

Array = NDArray[Any]

_MNIST_BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
_MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

_IMAGE_MAGIC = 2051
_LABEL_MAGIC = 2049


def parse_idx_images(raw: bytes) -> Array:
    """Parse an IDX3 image file into a float64 array of shape (n, rows*cols).

    Pixel bytes in [0, 255] are scaled to [0, 1]; each image is flattened to
    a row vector, the layout the MLP consumes.
    """
    magic, count, rows, cols = struct.unpack(">IIII", raw[:16])
    if magic != _IMAGE_MAGIC:
        msg = f"bad IDX image magic number {magic}, expected {_IMAGE_MAGIC}"
        raise ValueError(msg)
    pixels = np.frombuffer(raw, dtype=np.uint8, offset=16)
    if pixels.size != count * rows * cols:
        msg = f"IDX image payload has {pixels.size} bytes, expected {count * rows * cols}"
        raise ValueError(msg)
    return pixels.reshape(count, rows * cols).astype(np.float64) / 255.0


def parse_idx_labels(raw: bytes) -> Array:
    """Parse an IDX1 label file into an int64 vector."""
    magic, count = struct.unpack(">II", raw[:8])
    if magic != _LABEL_MAGIC:
        msg = f"bad IDX label magic number {magic}, expected {_LABEL_MAGIC}"
        raise ValueError(msg)
    labels = np.frombuffer(raw, dtype=np.uint8, offset=8)
    if labels.size != count:
        msg = f"IDX label payload has {labels.size} entries, expected {count}"
        raise ValueError(msg)
    return labels.astype(np.int64)


def _fetch(filename: str, cache_dir: Path) -> bytes:
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / filename
    if not target.exists():
        url = _MNIST_BASE_URL + filename
        print(f"downloading {url}")
        urllib.request.urlretrieve(url, target)
    return gzip.decompress(target.read_bytes())


def load_mnist(cache_dir: str | Path = "data/mnist") -> tuple[Array, Array, Array, Array]:
    """Return (train_images, train_labels, test_images, test_labels).

    Images have shape (n, 784) with values in [0, 1]; labels are int64
    vectors. Downloads on first use into ``cache_dir``.
    """
    cache = Path(cache_dir)
    x_train = parse_idx_images(_fetch(_MNIST_FILES["train_images"], cache))
    y_train = parse_idx_labels(_fetch(_MNIST_FILES["train_labels"], cache))
    x_test = parse_idx_images(_fetch(_MNIST_FILES["test_images"], cache))
    y_test = parse_idx_labels(_fetch(_MNIST_FILES["test_labels"], cache))
    return x_train, y_train, x_test, y_test


def iterate_minibatches(
    x: Array,
    y: Array,
    batch_size: int,
    rng: np.random.Generator,
) -> list[tuple[Array, Array]]:
    """Split (x, y) into shuffled minibatches for one epoch.

    A fresh permutation is drawn per call, so calling once per epoch gives
    the standard without-replacement sampling scheme of SGD.
    """
    if len(x) != len(y):
        msg = f"x and y have mismatched lengths {len(x)} and {len(y)}"
        raise ValueError(msg)
    if batch_size <= 0:
        msg = f"batch size must be positive, got {batch_size}"
        raise ValueError(msg)
    order = rng.permutation(len(x))
    return [
        (x[order[i : i + batch_size]], y[order[i : i + batch_size]])
        for i in range(0, len(x), batch_size)
    ]
