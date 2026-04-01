"""ROCm-only checks; skipped in default CI (``RUN_AMD_GPU_TESTS=1`` to run)."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.amd_gpu


def test_rocm_torch_cuda_and_hip() -> None:
    """ROCm wheels use the CUDA API and set ``torch.version.hip``."""
    import torch

    assert torch.cuda.is_available()
    hip = getattr(torch.version, "hip", None)
    assert hip is not None
