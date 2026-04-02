"""FFmpeg h264_amf argv builder for compilation encode (rawvideo stdin)."""

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
    """Build FFmpeg h264_amf command and log path (AMD AMF encode).

    ``config`` is reserved for future ``multi_cam.amd`` rate-control options
    (gpu-03). Why AMF here: matches multi-vendor plan for compilation on AMD
    deployments; requires FFmpeg built with AMF (typical on Windows, some Linux).
    """
    _ = config
    log_file_path = compilation_log_path(tmp_output_path)
    cmd = [
        "ffmpeg",
        "-y",
        *rawvideo_stdin_argv_fragment(target_w, target_h),
        "-c:v",
        "h264_amf",
        "-quality",
        "balanced",
        "-an",
        "-pix_fmt",
        "yuv420p",
        tmp_output_path,
    ]
    return cmd, log_file_path


class AmdFfmpegCompilationEncode:
    """Concrete :class:`FfmpegCompilationEncodeProto` for AMF compilation."""

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


amd_ffmpeg_compilation_encode = AmdFfmpegCompilationEncode()
