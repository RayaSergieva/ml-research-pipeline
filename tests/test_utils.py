"""Tests for the data utilities and the JSON-lines logger."""

from __future__ import annotations

import gzip
import struct
from pathlib import Path

import numpy as np
import pytest

from spectra.utils import RunLogger, iterate_minibatches, read_run
from spectra.utils.data import parse_idx_images, parse_idx_labels

RNG = np.random.default_rng(9)


# ---------------------------------------------------------------------------
# IDX parsing
# ---------------------------------------------------------------------------


def make_idx_images(images: np.ndarray) -> bytes:
    n, rows, cols = images.shape
    header = struct.pack(">IIII", 2051, n, rows, cols)
    return header + images.astype(np.uint8).tobytes()


def make_idx_labels(labels: np.ndarray) -> bytes:
    header = struct.pack(">II", 2049, len(labels))
    return header + labels.astype(np.uint8).tobytes()


def test_parse_idx_images_shape_and_scaling() -> None:
    imgs = RNG.integers(0, 256, size=(5, 4, 3), dtype=np.uint8)
    parsed = parse_idx_images(make_idx_images(imgs))
    assert parsed.shape == (5, 12)
    assert parsed.dtype == np.float64
    assert parsed.min() >= 0.0
    assert parsed.max() <= 1.0
    np.testing.assert_allclose(parsed, imgs.reshape(5, 12) / 255.0)


def test_parse_idx_labels_values() -> None:
    labels = np.array([0, 3, 9, 1], dtype=np.uint8)
    parsed = parse_idx_labels(make_idx_labels(labels))
    assert parsed.dtype == np.int64
    np.testing.assert_array_equal(parsed, labels)


def test_parse_idx_images_bad_magic_raises() -> None:
    imgs = RNG.integers(0, 256, size=(2, 2, 2), dtype=np.uint8)
    raw = bytearray(make_idx_images(imgs))
    raw[3] = 99
    with pytest.raises(ValueError, match="magic"):
        parse_idx_images(bytes(raw))


def test_parse_idx_labels_bad_magic_raises() -> None:
    raw = bytearray(make_idx_labels(np.array([1], dtype=np.uint8)))
    raw[3] = 99
    with pytest.raises(ValueError, match="magic"):
        parse_idx_labels(bytes(raw))


def test_parse_idx_images_truncated_payload_raises() -> None:
    imgs = RNG.integers(0, 256, size=(2, 2, 2), dtype=np.uint8)
    raw = make_idx_images(imgs)[:-3]
    with pytest.raises(ValueError, match="payload"):
        parse_idx_images(raw)


def test_gzip_roundtrip_matches_mnist_distribution_format() -> None:
    """The real files arrive gzipped; confirm the parse survives that path."""
    imgs = RNG.integers(0, 256, size=(3, 2, 2), dtype=np.uint8)
    compressed = gzip.compress(make_idx_images(imgs))
    parsed = parse_idx_images(gzip.decompress(compressed))
    assert parsed.shape == (3, 4)


# ---------------------------------------------------------------------------
# Minibatch iteration
# ---------------------------------------------------------------------------


def test_minibatches_cover_all_rows_exactly_once() -> None:
    x = np.arange(10).reshape(10, 1).astype(np.float64)
    y = np.arange(10)
    batches = iterate_minibatches(x, y, batch_size=3, rng=np.random.default_rng(0))
    seen = np.concatenate([yb for _, yb in batches])
    assert sorted(seen.tolist()) == list(range(10))
    assert [len(b[0]) for b in batches] == [3, 3, 3, 1]


def test_minibatches_keep_x_y_aligned() -> None:
    x = np.arange(20).reshape(20, 1).astype(np.float64)
    y = np.arange(20)
    for xb, yb in iterate_minibatches(x, y, batch_size=7, rng=np.random.default_rng(1)):
        np.testing.assert_array_equal(xb[:, 0].astype(np.int64), yb)


def test_minibatches_shuffle_differs_between_epochs() -> None:
    x = np.arange(50).reshape(50, 1).astype(np.float64)
    y = np.arange(50)
    rng = np.random.default_rng(2)
    first = np.concatenate([yb for _, yb in iterate_minibatches(x, y, 10, rng)])
    second = np.concatenate([yb for _, yb in iterate_minibatches(x, y, 10, rng)])
    assert not np.array_equal(first, second)


def test_minibatches_validate_inputs() -> None:
    x = np.zeros((4, 1))
    with pytest.raises(ValueError, match="mismatched"):
        iterate_minibatches(x, np.zeros(3), 2, np.random.default_rng(0))
    with pytest.raises(ValueError, match="batch size"):
        iterate_minibatches(x, np.zeros(4), 0, np.random.default_rng(0))


# ---------------------------------------------------------------------------
# RunLogger
# ---------------------------------------------------------------------------


def test_logger_writes_and_reads_back(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    with RunLogger(path) as logger:
        logger.log(0, event="config", lr=0.001)
        logger.log(1, event="train", loss=2.3)
        logger.log(2, event="train", loss=1.9)
    records = read_run(path)
    assert len(records) == 3
    assert records[0]["event"] == "config"
    assert records[2]["loss"] == 1.9
    assert all("elapsed_s" in r for r in records)


def test_logger_appends_across_sessions(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    with RunLogger(path) as logger:
        logger.log(0, a=1)
    with RunLogger(path) as logger:
        logger.log(1, a=2)
    assert len(read_run(path)) == 2


def test_logger_creates_parent_directories(tmp_path: Path) -> None:
    path = tmp_path / "deep" / "nested" / "run.jsonl"
    with RunLogger(path) as logger:
        logger.log(0, ok=True)
    assert path.exists()
