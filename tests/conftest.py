"""
Pytest conftest. Optional mocks for tests that run without GPU or optional deps.
GPU decode is mocked per-test (e.g. nvidia.decoder._create_simple_decoder,
VideoService._decoder_context, or get_gpu_backend return values).
"""

from __future__ import annotations

import os

import pytest


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Skip ``amd_gpu`` tests unless RUN_AMD_GPU_TESTS=1 (default CI has no ROCm)."""
    if os.environ.get("RUN_AMD_GPU_TESTS") == "1":
        return
    skip_amd = pytest.mark.skip(
        reason="set RUN_AMD_GPU_TESTS=1 for amd_gpu tests (ROCm host only)",
    )
    for item in items:
        if "amd_gpu" in item.keywords:
            item.add_marker(skip_amd)
