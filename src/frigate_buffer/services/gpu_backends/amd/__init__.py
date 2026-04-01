"""AMD ROCm backend: native decode (planned) + AMF/VAAPI FFmpeg helpers."""

from __future__ import annotations

from frigate_buffer.services.gpu_backends.types import GpuBackend


def build_amd_backend() -> GpuBackend:
    """Build AMD :class:`GpuBackend` for :func:`get_gpu_backend`."""
    from frigate_buffer.services.gpu_backends.amd.decoder import create_decoder
    from frigate_buffer.services.gpu_backends.amd.ffmpeg_encode import (
        amd_ffmpeg_compilation_encode,
    )
    from frigate_buffer.services.gpu_backends.amd.gif_ffmpeg import amd_gif_ffmpeg
    from frigate_buffer.services.gpu_backends.amd.runtime import amd_runtime

    return GpuBackend(
        create_decoder=create_decoder,
        runtime=amd_runtime,
        ffmpeg_compilation_encode=amd_ffmpeg_compilation_encode,
        gif_ffmpeg=amd_gif_ffmpeg,
    )


__all__ = ["build_amd_backend"]
