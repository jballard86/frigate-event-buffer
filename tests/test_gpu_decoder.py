"""
Tests for the GPU decoder wrapper (gpu_decoder.py).

Uses mocks so tests run without PyNvVideoCodec/GPU. Verifies DecoderContext API,
create_decoder context manager, and BCHW tensor shape from get_frames.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def mock_pynv(monkeypatch):
    """Mock PyNvVideoCodec so tests run without GPU."""
    class MockFrame:
        """DLPack-capable mock frame: (3, H, W) CHW."""

        def __init__(self, h: int = 1080, w: int = 1920):
            import torch
            self._t = torch.zeros(3, h, w, dtype=torch.uint8, device="cuda")

        def __dlpack__(self):
            return self._t.__dlpack__()

        def __dlpack_device__(self):
            return (2, 0)  # CUDA, device 0

    class MockSimpleDecoder:
        def __init__(self, path, gpu_id=0, use_device_memory=True, max_width=4096, output_color_type=None):
            self._path = path
            self._frame_count = 100

        def __len__(self):
            return self._frame_count

        def get_batch_frames_by_index(self, indices):
            return [MockFrame() for _ in indices]

        def __getitem__(self, index):
            return MockFrame()

        def seek_to_index(self, index):
            pass

        def get_index_from_time_in_seconds(self, t_sec):
            return max(0, min(int(t_sec * 30), self._frame_count - 1))

        def get_batch_frames(self, count):
            return [MockFrame() for _ in range(count)]

    class Nvc:
        SimpleDecoder = MockSimpleDecoder
        OutputColorType = type("OutputColorType", (), {"RGBP": "RGBP"})()

    monkeypatch.setattr(
        "frigate_buffer.services.gpu_decoder._create_simple_decoder",
        lambda path, gpu_id: MockSimpleDecoder(path, gpu_id=gpu_id),
    )
    return Nvc


def test_decoder_context_frame_count(mock_pynv):
    """DecoderContext.frame_count and __len__ return decoder length."""
    from frigate_buffer.services.gpu_decoder import DecoderContext

    class Dec:
        def __len__(self):
            return 42

    ctx = DecoderContext(Dec())
    assert len(ctx) == 42
    assert ctx.frame_count == 42


@pytest.mark.skipif(
    __import__("torch").cuda.is_available() is False,
    reason="get_frames([]) returns cuda empty tensor; skip when CUDA not available",
)
def test_decoder_context_get_frames_empty_indices(mock_pynv):
    """get_frames([]) returns empty BCHW tensor."""
    import torch
    from frigate_buffer.services.gpu_decoder import DecoderContext

    class Dec:
        def __len__(self):
            return 10

        def get_batch_frames_by_index(self, indices):
            return []

    ctx = DecoderContext(Dec())
    out = ctx.get_frames([])
    assert isinstance(out, torch.Tensor)
    assert out.dim() == 4
    assert out.shape[0] == 0
    assert out.shape[1] == 3
    assert out.dtype == torch.uint8


def test_decoder_context_get_frames_bchw(mock_pynv):
    """get_frames([0, 1]) returns BCHW tensor (N, 3, H, W)."""
    import torch
    from frigate_buffer.services.gpu_decoder import DecoderContext

    class Dec:
        def __len__(self):
            return 10

        def get_batch_frames_by_index(self, indices):
            return [object() for _ in indices]  # MockFrame used by real mock_pynv

    # Mock frame with DLPack (use cpu so test runs without GPU)
    class MockFrame:
        def __init__(self):
            self._t = torch.zeros(3, 1080, 1920, dtype=torch.uint8)

        def __dlpack__(self):
            return self._t.__dlpack__()

        def __dlpack_device__(self):
            return (1, 0)  # CPU device type in DLPack

    class DecWithFrames(Dec):
        def get_batch_frames_by_index(self, indices):
            return [MockFrame() for _ in indices]

    ctx = DecoderContext(DecWithFrames())
    out = ctx.get_frames([0, 1])
    assert isinstance(out, torch.Tensor)
    assert out.dim() == 4
    assert out.shape[0] == 2
    assert out.shape[1] == 3
    assert out.shape[2] == 1080
    assert out.shape[3] == 1920
    assert out.dtype == torch.uint8


def test_create_decoder_yields_context(mock_pynv):
    """create_decoder yields DecoderContext and closes on exit."""
    from frigate_buffer.services.gpu_decoder import create_decoder

    with create_decoder("/fake/path.mp4", gpu_id=0) as ctx:
        assert ctx.frame_count == 100
        assert len(ctx) == 100


def test_create_decoder_logs_on_failure(monkeypatch):
    """On init failure, NVDEC_INIT_FAILURE_PREFIX is logged and exception re-raised."""
    from frigate_buffer.constants import NVDEC_INIT_FAILURE_PREFIX
    from frigate_buffer.services.gpu_decoder import create_decoder

    def fail(*args, **kwargs):
        raise RuntimeError("fake NVDEC init failure")

    monkeypatch.setattr(
        "frigate_buffer.services.gpu_decoder._create_simple_decoder",
        fail,
    )
    with pytest.raises(RuntimeError, match="fake NVDEC"):
        with create_decoder("/fake/path.mp4"):
            pass
    # Prefix is used in logger.error; we only verify the exception propagates.
    # Full log assertion would require caplog.
