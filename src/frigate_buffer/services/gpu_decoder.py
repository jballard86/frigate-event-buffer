"""
GPU-native video decoder service using PyNvVideoCodec SimpleDecoder.

This is the only module that imports PyNvVideoCodec. It provides a zero-copy pipeline:
decode to device memory, DLPack handover to PyTorch, output BCHW RGB tensors.
Callers (video.py, multi_clip_extractor.py, video_compilation.py) use this wrapper
so the CPU never touches decoded pixels until encode/export boundaries.

Decoder is created with use_device_memory=True, max_width=4096 (hardware width cap),
and OutputColorType.RGBP (planar CHW) for direct BCHW stacking. All decode and
frame access must be serialized by the app-wide GPU_LOCK in video.py.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from frigate_buffer.constants import NVDEC_INIT_FAILURE_PREFIX

logger = logging.getLogger("frigate-buffer")

# Maximum decoded width enforced in hardware.
DECODER_MAX_WIDTH = 4096


class DecoderContext:
    """
    Thin wrapper around PyNvVideoCodec SimpleDecoder exposing frame count and
    batch get in BCHW RGB (uint8) for the rest of the pipeline.
    """

    def __init__(self, decoder: Any) -> None:
        self._decoder = decoder

    def __len__(self) -> int:
        """Total number of frames in the video (from decoder metadata/scan)."""
        return len(self._decoder)

    @property
    def frame_count(self) -> int:
        """Alias for len(self) for clarity at call sites."""
        return len(self)

    def get_frames(self, indices: list[int]) -> Any:
        """
        Decode frames at the given indices and return a single BCHW tensor (uint8 RGB).

        Uses DLPack for zero-copy handover to PyTorch. Each frame from the decoder
        is CHW (3, H, W); stacking produces (N, 3, H, W) = BCHW. Caller must hold
        GPU_LOCK when calling this. The returned batch is cloned so the decoder
        context may be closed or reconfigured without invalidating the tensor.
        """
        import torch

        if not indices:
            # Return empty tensor with 4 dims for BCHW (device from decoder).
            return torch.empty((0, 3, 0, 0), dtype=torch.uint8, device="cuda")
        # PyNvVideoCodec emits UserWarning when duplicate indices are passed
        # (e.g. stutter in source).
        # Re-log at DEBUG and suppress default WARNING so logs stay clean at INFO.
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always", UserWarning)
            old_showwarning = warnings.showwarning

            def _showwarning(message, category, filename, lineno, file=None, line=None):
                if category is UserWarning and "Duplicates" in str(message):
                    logger.debug("PyNvVideoCodec: %s", message)
                    return
                old_showwarning(message, category, filename, lineno, file, line)

            warnings.showwarning = _showwarning
            try:
                raw_frames = self._decoder.get_batch_frames_by_index(indices)
            finally:
                warnings.showwarning = old_showwarning
        # Each frame supports __dlpack__; RGBP gives (3, H, W).
        # torch.from_dlpack exists at runtime but is not in type stubs; use getattr.
        from_dlpack = getattr(torch, "from_dlpack")
        tensors = [from_dlpack(f) for f in raw_frames]
        # Stack to BCHW. Cloning avoids sharing memory with decoder's internal buffer
        # so decoder can be closed or reconfigured without invalidating tensors.
        batch = torch.stack(tensors, dim=0).clone()
        return batch

    def get_frame_at_index(self, index: int) -> Any:
        """
        Decode a single frame at index; returns (1, 3, H, W) BCHW uint8 tensor.

        Caller must hold GPU_LOCK. Convenience for single-frame extraction.
        """
        import torch

        # torch.from_dlpack exists at runtime but is not in type stubs; use getattr.
        from_dlpack = getattr(torch, "from_dlpack")
        frame = self._decoder[index]
        t = from_dlpack(frame).unsqueeze(0).clone()
        return t

    def get_batch_frames(self, count: int) -> list[Any]:
        """
        Sequential batch: next `count` frames from current position. Returns list of
        DLPack-capable frame objects (caller can from_dlpack each). Used for
        PTS/sequential compilation path. Caller must hold GPU_LOCK.
        """
        return self._decoder.get_batch_frames(count)

    def seek_to_index(self, index: int) -> None:
        """Seek decoder to frame index (e.g. segment start). Caller must hold
        GPU_LOCK."""
        self._decoder.seek_to_index(index)

    def get_index_from_time_in_seconds(self, t_sec: float) -> int:
        """Map time in seconds to frame index (for segment/sample time mapping)."""
        return self._decoder.get_index_from_time_in_seconds(t_sec)


def _create_simple_decoder(clip_path: str, gpu_id: int) -> Any:
    """
    Instantiate PyNvVideoCodec SimpleDecoder with project defaults.

    use_device_memory=True (VRAM decode), max_width=4096 (hardware cap),
    output_color_type=RGBP (planar CHW for BCHW). Raises on init failure.
    """
    import PyNvVideoCodec as nvc

    # OutputColorType exists at runtime but is not in type stubs; use getattr.
    output_color_type = getattr(nvc, "OutputColorType").RGBP
    return nvc.SimpleDecoder(
        clip_path,
        gpu_id=gpu_id,
        use_device_memory=True,
        max_width=DECODER_MAX_WIDTH,
        output_color_type=output_color_type,
    )


@contextmanager
def create_decoder(clip_path: str, gpu_id: int = 0) -> Iterator[DecoderContext]:
    """
    Create a GPU decoder for the given clip and yield a DecoderContext.

    Decoder uses use_device_memory=True, max_width=4096, RGBP output. Caller must
    hold the app-wide GPU_LOCK when entering this context and when calling
    any DecoderContext methods. On failure logs NVDEC_INIT_FAILURE_PREFIX and
    re-raises (no silent fallback).
    """
    decoder = None
    try:
        decoder = _create_simple_decoder(clip_path, gpu_id)
        yield DecoderContext(decoder)
    except Exception as e:
        # PyNvVideoCodec may raise library-specific exceptions; we catch all and
        # log with NVDEC_INIT_FAILURE_PREFIX so crash-loop and support searches
        # remain effective.
        logger.error(
            "%s (PyNvVideoCodec decoder init failed). path=%s error=%s "
            "Check GPU, drivers, and container NVDEC access.",
            NVDEC_INIT_FAILURE_PREFIX,
            clip_path,
            e,
            exc_info=True,
        )
        raise
    finally:
        if decoder is not None:
            try:
                del decoder
            except Exception:
                pass
