"""Test helpers: video compilation GPU mocks and fake decoder contexts."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock


def gpu_backend_for_compilation_tests(**create_decoder_kw: Any) -> MagicMock:
    """Real NVENC argv + runtime; ``create_decoder`` mocked per test kwargs."""
    from frigate_buffer.services.gpu_backends.nvidia.ffmpeg_encode import (
        nvidia_ffmpeg_compilation_encode,
    )
    from frigate_buffer.services.gpu_backends.nvidia.runtime import nvidia_runtime

    b = MagicMock()
    b.create_decoder = MagicMock(**create_decoder_kw)
    b.ffmpeg_compilation_encode = nvidia_ffmpeg_compilation_encode
    b.runtime = nvidia_runtime
    return b


def gpu_backend_for_compilation_tests_intel(**create_decoder_kw: Any) -> MagicMock:
    """Real QSV argv + Intel runtime; ``create_decoder`` mocked per test kwargs."""
    from frigate_buffer.services.gpu_backends.intel.ffmpeg_encode import (
        intel_ffmpeg_compilation_encode,
    )
    from frigate_buffer.services.gpu_backends.intel.runtime import intel_runtime

    b = MagicMock()
    b.create_decoder = MagicMock(**create_decoder_kw)
    b.ffmpeg_compilation_encode = intel_ffmpeg_compilation_encode
    b.runtime = intel_runtime
    return b


def gpu_backend_for_compilation_tests_amd(**create_decoder_kw: Any) -> MagicMock:
    """Real h264_amf argv + AMD runtime; ``create_decoder`` mocked per test kwargs."""
    from frigate_buffer.services.gpu_backends.amd.ffmpeg_encode import (
        amd_ffmpeg_compilation_encode,
    )
    from frigate_buffer.services.gpu_backends.amd.runtime import amd_runtime

    b = MagicMock()
    b.create_decoder = MagicMock(**create_decoder_kw)
    b.ffmpeg_compilation_encode = amd_ffmpeg_compilation_encode
    b.runtime = amd_runtime
    return b


def fake_create_decoder_context(
    frame_count: int = 200, height: int = 480, width: int = 640
) -> Any:
    """Build a mock DecoderContext for _run_pynv_compilation tests."""
    try:
        import torch
    except ImportError:
        return None
    mock_ctx = MagicMock()
    mock_ctx.__len__ = lambda self: frame_count
    mock_ctx.get_index_from_time_in_seconds = lambda t: min(
        max(0, int(t * 20)), max(0, frame_count - 1)
    )
    mock_ctx.get_frames = lambda indices: torch.zeros(
        (len(indices), 3, height, width), dtype=torch.uint8
    )
    return mock_ctx
