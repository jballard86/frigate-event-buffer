"""Bundled backend handles (filled by registry in gpu-01 Phase 3)."""

from __future__ import annotations

from dataclasses import dataclass

from frigate_buffer.services.gpu_backends.protocols import (
    CreateDecoderFn,
    FfmpegCompilationEncodeProto,
    GifFfmpegProto,
    GpuRuntimeProto,
)


@dataclass(frozen=True)
class GpuBackend:
    """Vendor implementation bundle for decode, runtime, FFmpeg encode, and GIF."""

    create_decoder: CreateDecoderFn
    runtime: GpuRuntimeProto
    ffmpeg_compilation_encode: FfmpegCompilationEncodeProto
    gif_ffmpeg: GifFfmpegProto
