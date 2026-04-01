"""Intel DecoderContext with mocked frigate_intel_decode (no native .so required)."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest
import torch

from frigate_buffer.services.gpu_backends.registry import clear_gpu_backend_cache


@pytest.fixture
def fake_native_module() -> types.ModuleType:
    """Minimal fake matching IntelDecoderSession API used by DecoderContext."""

    class FakeSession:
        def __init__(self, path: str, gpu_id: int = 0) -> None:
            self._path = path
            self._gpu_id = gpu_id
            self._nb = 120

        def frame_count(self) -> int:
            return self._nb

        def __len__(self) -> int:
            return self._nb

        def get_frames(self, indices: list[int]) -> torch.Tensor:
            if not indices:
                return torch.empty((0, 3, 0, 0), dtype=torch.uint8)
            return torch.zeros((len(indices), 3, 64, 64), dtype=torch.uint8)

        def get_frame_at_index(self, index: int) -> torch.Tensor:
            return torch.zeros((1, 3, 64, 64), dtype=torch.uint8)

        def get_batch_frames(self, count: int) -> list[torch.Tensor]:
            return [
                torch.zeros((1, 3, 64, 64), dtype=torch.uint8) for _ in range(count)
            ]

        def seek_to_index(self, index: int) -> None:
            pass

        def get_index_from_time_in_seconds(self, t_sec: float) -> int:
            return int(t_sec * 30.0)

    m = types.ModuleType("frigate_intel_decode")
    m.IntelDecoderSession = FakeSession
    m.version = lambda: "0.2.0-mock"
    m.decode_first_frame_bchw_rgb = MagicMock()
    return m


def test_intel_decoder_context_get_frames_empty_device_cpu(
    fake_native_module: types.ModuleType,
) -> None:
    sys.modules["frigate_intel_decode"] = fake_native_module
    try:
        import importlib

        import frigate_buffer.services.gpu_backends.intel.decoder as dec

        importlib.reload(dec)
        ctx = dec.DecoderContext(
            fake_native_module.IntelDecoderSession("/x.mp4", 0), gpu_id=0
        )
        t = ctx.get_frames([])
        assert t.shape == (0, 3, 0, 0)
        assert str(t.device) == "cpu"
    finally:
        sys.modules.pop("frigate_intel_decode", None)
        import importlib

        import frigate_buffer.services.gpu_backends.intel.decoder as dec

        importlib.reload(dec)
        clear_gpu_backend_cache()


def test_intel_decoder_context_len_and_time_index(
    fake_native_module: types.ModuleType,
) -> None:
    sys.modules["frigate_intel_decode"] = fake_native_module
    try:
        import importlib

        import frigate_buffer.services.gpu_backends.intel.decoder as dec

        importlib.reload(dec)
        ctx = dec.DecoderContext(fake_native_module.IntelDecoderSession("/clip.mp4", 0))
        assert len(ctx) == 120
        assert ctx.frame_count == 120
        assert ctx.get_index_from_time_in_seconds(1.0) == 30
    finally:
        sys.modules.pop("frigate_intel_decode", None)
        import importlib

        import frigate_buffer.services.gpu_backends.intel.decoder as dec

        importlib.reload(dec)
        clear_gpu_backend_cache()
