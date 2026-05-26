"""Smoke tests that verify the package imports and exposes its version."""

from __future__ import annotations

import spectra


def test_package_has_version() -> None:
    assert isinstance(spectra.__version__, str)
    assert spectra.__version__.count(".") == 2


def test_version_is_pre_alpha() -> None:
    major, minor, patch = (int(p) for p in spectra.__version__.split("."))
    assert major == 0
    assert minor == 0
    assert patch >= 1
