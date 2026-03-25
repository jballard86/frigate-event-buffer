"""Narrow protocols for pluggable GPU backends (Intel/AMD satisfy same surface)."""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class DecoderContextProto(Protocol):
    """Frame access contract matching NVDEC → BCHW uint8 path."""

    def __len__(self) -> int: ...

    @property
    def frame_count(self) -> int: ...

    def get_frames(self, indices: list[int]) -> Any: ...

    def get_frame_at_index(self, index: int) -> Any: ...

    def get_batch_frames(self, count: int) -> list[Any]: ...

    def seek_to_index(self, index: int) -> None: ...

    def get_index_from_time_in_seconds(self, t_sec: float) -> int: ...


class CreateDecoderFn(Protocol):
    """Shape of create_decoder(clip_path, gpu_id=0) context manager factory."""

    def __call__(
        self, clip_path: str, gpu_id: int = 0
    ) -> AbstractContextManager[DecoderContextProto]: ...


@runtime_checkable
class GpuRuntimeProto(Protocol):
    """VRAM / device helpers and startup diagnostics."""

    def log_gpu_status(self) -> None: ...

    def empty_cache(self) -> None: ...

    def memory_summary(self, *, abbreviated: bool = False) -> str | None: ...

    def tensor_device_for_decode(self, gpu_id: int = 0) -> str: ...

    def default_detection_device(self, config: dict) -> str | None: ...


@runtime_checkable
class FfmpegCompilationEncodeProto(Protocol):
    """Argv builders for compilation encode (stdin rawvideo → NVENC/QSV/AMF)."""

    def compilation_ffmpeg_cmd_and_log_path(
        self, tmp_output_path: str, target_w: int, target_h: int
    ) -> tuple[list[str], str]: ...


@runtime_checkable
class GifFfmpegProto(Protocol):
    """FFmpeg argv + filter_complex for hardware GIF preview."""

    def gif_filter_complex(self, fps: int, preview_width: int) -> str: ...

    def gif_ffmpeg_argv(
        self,
        clip_path: str,
        output_path: str,
        fps: int,
        duration_sec: float,
        preview_width: int,
    ) -> list[str]: ...
