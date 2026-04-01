"""
Intel decode: native ``frigate_intel_decode.IntelDecoderSession`` → BCHW uint8.

Holds ``GPU_LOCK`` during use (same contract as NVIDIA). Native module must be
built (see ``native/intel_decode/README.md``).
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from frigate_buffer.constants import GPU_DECODE_INIT_FAILURE_PREFIX
from frigate_buffer.services.gpu_backends.intel.runtime import intel_runtime

logger = logging.getLogger("frigate-buffer")

DECODER_MAX_WIDTH = 4096


def _to_decode_device(batch: Any, dev_s: str) -> Any:
    """Move tensor to ``dev_s`` only when it is not already on that device."""
    import torch

    if dev_s == "cpu":
        return batch.cpu() if getattr(batch, "is_cuda", False) else batch
    dev = torch.device(dev_s)
    if batch.device.type == dev.type and batch.device.index == dev.index:
        return batch
    return batch.to(device=dev, non_blocking=True)


class DecoderContext:
    """Adapter implementing :class:`DecoderContextProto` over the native session."""

    def __init__(self, session: Any, gpu_id: int = 0) -> None:
        self._session = session
        self._gpu_id = gpu_id

    def __len__(self) -> int:
        return int(self._session.frame_count())

    @property
    def frame_count(self) -> int:
        return int(self._session.frame_count())

    def get_frames(self, indices: list[int]) -> Any:
        import torch

        if not indices:
            dev = torch.device(intel_runtime.tensor_device_for_decode(self._gpu_id))
            return torch.empty((0, 3, 0, 0), dtype=torch.uint8, device=dev)
        conv = [int(i) for i in indices]
        batch_cpu = self._session.get_frames(conv)
        dev_s = intel_runtime.tensor_device_for_decode(self._gpu_id)
        return _to_decode_device(batch_cpu, dev_s)

    def get_frame_at_index(self, index: int) -> Any:
        t_cpu = self._session.get_frame_at_index(int(index))
        dev_s = intel_runtime.tensor_device_for_decode(self._gpu_id)
        return _to_decode_device(t_cpu, dev_s)

    def get_batch_frames(self, count: int) -> list[Any]:
        raw = self._session.get_batch_frames(int(count))
        dev_s = intel_runtime.tensor_device_for_decode(self._gpu_id)
        return [_to_decode_device(t_cpu, dev_s) for t_cpu in raw]

    def seek_to_index(self, index: int) -> None:
        self._session.seek_to_index(int(index))

    def get_index_from_time_in_seconds(self, t_sec: float) -> int:
        return int(self._session.get_index_from_time_in_seconds(float(t_sec)))


@contextmanager
def create_decoder(clip_path: str, gpu_id: int = 0) -> Iterator[DecoderContext]:
    """Open native session; on failure log GPU decode init prefix + ``[intel]``."""
    import frigate_intel_decode as native

    session = None
    try:
        session = native.IntelDecoderSession(clip_path, gpu_id)
        yield DecoderContext(session, gpu_id=gpu_id)
    except Exception as e:
        logger.error(
            "%s [intel] (native decoder init or decode failed). path=%s error=%s",
            GPU_DECODE_INIT_FAILURE_PREFIX,
            clip_path,
            e,
            exc_info=True,
        )
        raise
    finally:
        del session
