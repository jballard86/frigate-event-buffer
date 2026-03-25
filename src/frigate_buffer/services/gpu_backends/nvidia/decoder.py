"""
NVIDIA decode: PyNvVideoCodec SimpleDecoder → DLPack → torch BCHW uint8.

All PyNvVideoCodec imports stay in this package. Callers must hold
``GPU_LOCK`` from ``frigate_buffer.services.gpu_backends.lock`` during
decoder use.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from frigate_buffer.constants import NVDEC_INIT_FAILURE_PREFIX
from frigate_buffer.services.gpu_backends.nvidia import runtime as nv_runtime

logger = logging.getLogger("frigate-buffer")

DECODER_MAX_WIDTH = 4096


class DecoderContext:
    """
    Thin wrapper around PyNvVideoCodec SimpleDecoder exposing frame count and
    batch get in BCHW RGB (uint8) for the rest of the pipeline.
    """

    def __init__(self, decoder: Any, gpu_id: int = 0) -> None:
        self._decoder = decoder
        self._gpu_id = gpu_id

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
            dev = torch.device(nv_runtime.tensor_device_for_decode(self._gpu_id))
            return torch.empty((0, 3, 0, 0), dtype=torch.uint8, device=dev)
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
        from_dlpack = torch.from_dlpack
        tensors = [from_dlpack(f) for f in raw_frames]
        batch = torch.stack(tensors, dim=0).clone()
        return batch

    def get_frame_at_index(self, index: int) -> Any:
        """
        Decode a single frame at index; returns (1, 3, H, W) BCHW uint8 tensor.

        Caller must hold GPU_LOCK. Convenience for single-frame extraction.
        """
        import torch

        from_dlpack = torch.from_dlpack
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

    output_color_type = nvc.OutputColorType.RGBP
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
        yield DecoderContext(decoder, gpu_id=gpu_id)
    except Exception as e:
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
