"""Pytest fixtures scoped to ``tests/video_compilation/`` only.

Why not root ``tests/conftest.py``: heavy GPU-backend imports and mock wiring
would run during collection for the entire suite and slow discovery. Keep
compilation-specific fixtures here so only tests under this package load them.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock

import pytest

from helpers.video_compilation import (
    fake_create_decoder_context,
    gpu_backend_for_compilation_tests,
    gpu_backend_for_compilation_tests_amd,
    gpu_backend_for_compilation_tests_intel,
)


@pytest.fixture
def make_fake_decoder_context() -> Callable[..., Any]:
    """Factory for ``fake_create_decoder_context`` (frame_count, height, width)."""

    def _make(
        frame_count: int = 200,
        height: int = 480,
        width: int = 640,
    ) -> Any:
        return fake_create_decoder_context(
            frame_count=frame_count, height=height, width=width
        )

    return _make


@pytest.fixture
def make_nvidia_compilation_gpu_backend() -> Callable[..., MagicMock]:
    """Returns :func:`gpu_backend_for_compilation_tests` for ``**create_decoder_kw``."""
    return gpu_backend_for_compilation_tests


@pytest.fixture
def make_intel_compilation_gpu_backend() -> Callable[..., MagicMock]:
    """Returns :func:`gpu_backend_for_compilation_tests_intel`."""
    return gpu_backend_for_compilation_tests_intel


@pytest.fixture
def make_amd_compilation_gpu_backend() -> Callable[..., MagicMock]:
    """Returns :func:`gpu_backend_for_compilation_tests_amd`."""
    return gpu_backend_for_compilation_tests_amd
