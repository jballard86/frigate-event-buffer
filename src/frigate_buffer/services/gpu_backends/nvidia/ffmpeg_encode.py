"""FFmpeg h264_nvenc argv builder for compilation encode (rawvideo stdin)."""

from __future__ import annotations

from frigate_buffer.services.gpu_backends.compilation_argv_common import (
    compilation_log_path,
    rawvideo_stdin_argv_fragment,
)


def compilation_ffmpeg_cmd_and_log_path(
    tmp_output_path: str,
    target_w: int,
    target_h: int,
    *,
    config: dict | None = None,
) -> tuple[list[str], str]:
    """
    Build FFmpeg h264_nvenc command and log path for compilation encode.

    Shared by streaming and batch encode paths; behavior unchanged from
    pre-refactor video_compilation. ``config`` is accepted for protocol parity
    with the Intel backend (NVENC argv does not read it).
    """
    _ = config
    log_file_path = compilation_log_path(tmp_output_path)
    cmd = [
        "ffmpeg",
        "-y",
        *rawvideo_stdin_argv_fragment(target_w, target_h),
        "-c:v",
        "h264_nvenc",
        "-preset",
        "p1",
        "-tune",
        "hq",
        "-rc",
        "vbr",
        "-cq",
        "24",
        "-an",
        "-pix_fmt",
        "yuv420p",
        tmp_output_path,
    ]
    return cmd, log_file_path


NVENC_COMPILE_NOT_FOUND_USER_MSG = (
    "Compilation encode failed: ffmpeg not found. "
    "Compilation requires GPU encoding (h264_nvenc). No CPU fallback. "
    "Ensure FFmpeg is installed and on PATH with NVENC support."
)

NVENC_COMPILE_NOT_FOUND_RUNTIME_MSG = (
    "ffmpeg not found; compilation encoding is GPU-only (h264_nvenc), no CPU fallback"
)


class NvidiaFfmpegCompilationEncode:
    """Concrete :class:`FfmpegCompilationEncodeProto` for NVENC compilation."""

    def compilation_ffmpeg_cmd_and_log_path(
        self,
        tmp_output_path: str,
        target_w: int,
        target_h: int,
        *,
        config: dict | None = None,
    ) -> tuple[list[str], str]:
        return compilation_ffmpeg_cmd_and_log_path(
            tmp_output_path, target_w, target_h, config=config
        )


nvidia_ffmpeg_compilation_encode = NvidiaFfmpegCompilationEncode()
