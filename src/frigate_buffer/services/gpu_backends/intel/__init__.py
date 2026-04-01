"""Intel Arc backend: native decode + QSV FFmpeg helpers."""

from __future__ import annotations

from frigate_buffer.services.gpu_backends.types import GpuBackend


def build_intel_backend() -> GpuBackend:
    """Build Intel :class:`GpuBackend` for :func:`get_gpu_backend`."""
    from frigate_buffer.services.gpu_backends.intel.decoder import create_decoder
    from frigate_buffer.services.gpu_backends.intel.ffmpeg_encode import (
        intel_ffmpeg_compilation_encode,
    )
    from frigate_buffer.services.gpu_backends.intel.gif_ffmpeg import intel_gif_ffmpeg
    from frigate_buffer.services.gpu_backends.intel.runtime import intel_runtime

    return GpuBackend(
        create_decoder=create_decoder,
        runtime=intel_runtime,
        ffmpeg_compilation_encode=intel_ffmpeg_compilation_encode,
        gif_ffmpeg=intel_gif_ffmpeg,
    )


__all__ = ["build_intel_backend"]
