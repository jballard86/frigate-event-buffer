"""NVIDIA backend: PyNvVideoCodec decode, NVENC compilation, CUDA GIF FFmpeg."""

from frigate_buffer.services.gpu_backends.nvidia.decoder import (
    DECODER_MAX_WIDTH,
    DecoderContext,
    _create_simple_decoder,
    create_decoder,
)
from frigate_buffer.services.gpu_backends.nvidia.ffmpeg_encode import (
    COMPILATION_OUTPUT_FPS,
    NvidiaFfmpegCompilationEncode,
    compilation_ffmpeg_cmd_and_log_path,
    nvidia_ffmpeg_compilation_encode,
)
from frigate_buffer.services.gpu_backends.nvidia.gif_ffmpeg import (
    NvidiaGifFfmpeg,
    gif_ffmpeg_argv,
    gif_filter_complex,
    nvidia_gif_ffmpeg,
)
from frigate_buffer.services.gpu_backends.nvidia.runtime import (
    NvidiaRuntime,
    default_detection_device,
    empty_cache,
    log_gpu_status,
    memory_summary,
    nvidia_runtime,
    tensor_device_for_decode,
)
from frigate_buffer.services.gpu_backends.types import GpuBackend


def build_nvidia_backend() -> GpuBackend:
    """Build NVIDIA :class:`GpuBackend` for :func:`get_gpu_backend`."""
    return GpuBackend(
        create_decoder=create_decoder,
        runtime=nvidia_runtime,
        ffmpeg_compilation_encode=nvidia_ffmpeg_compilation_encode,
        gif_ffmpeg=nvidia_gif_ffmpeg,
    )


__all__ = [
    "COMPILATION_OUTPUT_FPS",
    "DECODER_MAX_WIDTH",
    "DecoderContext",
    "NvidiaFfmpegCompilationEncode",
    "NvidiaGifFfmpeg",
    "NvidiaRuntime",
    "_create_simple_decoder",
    "build_nvidia_backend",
    "compilation_ffmpeg_cmd_and_log_path",
    "create_decoder",
    "default_detection_device",
    "empty_cache",
    "gif_ffmpeg_argv",
    "gif_filter_complex",
    "log_gpu_status",
    "memory_summary",
    "nvidia_ffmpeg_compilation_encode",
    "nvidia_gif_ffmpeg",
    "nvidia_runtime",
    "tensor_device_for_decode",
]
