"""
AMD decode: native ``frigate_amd_decode.AmdDecoderSession`` → BCHW ``uint8``.

Native builds from ``native/amd_decode/`` (VAAPI/SW FFmpeg → CPU tensors); when
``amd_runtime.tensor_device_for_decode`` is ``cuda:N`` (ROCm torch), frames are
moved to the GPU in Python. Missing ``import`` raises a chained error with build
hints. Callers hold ``GPU_LOCK`` (same contract as NVIDIA / Intel).
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from frigate_buffer.constants import GPU_DECODE_INIT_FAILURE_PREFIX
from frigate_buffer.services.gpu_backends.amd.runtime import amd_runtime

logger = logging.getLogger("frigate-buffer")

_AMD_DECODE_IMPORT_HINT = (
    "Build and install the frigate_amd_decode extension (see native/amd_decode/ "
    "and docs/Multi_GPU_Support_Integration_Plan/gpu-03-amd-rocm.md)."
)


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
            dev = torch.device(amd_runtime.tensor_device_for_decode(self._gpu_id))
            return torch.empty((0, 3, 0, 0), dtype=torch.uint8, device=dev)
        conv = [int(i) for i in indices]
        batch = self._session.get_frames(conv)
        dev_s = amd_runtime.tensor_device_for_decode(self._gpu_id)
        if dev_s == "cpu":
            return batch
        return batch.to(device=torch.device(dev_s), non_blocking=True)

    def get_frame_at_index(self, index: int) -> Any:
        import torch

        one = self._session.get_frame_at_index(int(index))
        dev_s = amd_runtime.tensor_device_for_decode(self._gpu_id)
        if dev_s == "cpu":
            return one
        return one.to(device=torch.device(dev_s), non_blocking=True)

    def get_batch_frames(self, count: int) -> list[Any]:
        import torch

        raw = self._session.get_batch_frames(int(count))
        dev_s = amd_runtime.tensor_device_for_decode(self._gpu_id)
        dev = torch.device(dev_s)
        out: list[Any] = []
        for t in raw:
            if dev_s == "cpu":
                out.append(t)
            else:
                out.append(t.to(device=dev, non_blocking=True))
        return out

    def seek_to_index(self, index: int) -> None:
        self._session.seek_to_index(int(index))

    def get_index_from_time_in_seconds(self, t_sec: float) -> int:
        return int(self._session.get_index_from_time_in_seconds(float(t_sec)))


@contextmanager
def create_decoder(clip_path: str, gpu_id: int = 0) -> Iterator[DecoderContext]:
    """Open native AMD session; on failure log ``[amd]`` and preserve import errors."""
    try:
        import frigate_amd_decode as native
    except ImportError as e:
        raise ImportError(_AMD_DECODE_IMPORT_HINT) from e

    session = None
    try:
        session = native.AmdDecoderSession(clip_path, gpu_id)
        yield DecoderContext(session, gpu_id=gpu_id)
    except Exception as e:
        logger.error(
            "%s [amd] (native decoder init or decode failed). path=%s error=%s",
            GPU_DECODE_INIT_FAILURE_PREFIX,
            clip_path,
            e,
            exc_info=True,
        )
        raise
    finally:
        del session
