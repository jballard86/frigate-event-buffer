"""FFmpeg h264_qsv argv builder for compilation encode (rawvideo stdin)."""

from __future__ import annotations

from frigate_buffer.services.gpu_backends.compilation_argv_common import (
    compilation_log_path,
    rawvideo_stdin_argv_fragment,
)


def intel_qsv_encode_preset(config: dict | None) -> str:
    """Resolve ``-preset`` for h264_qsv from merged app config (YAML / env).

    Why: QSV quality/speed tradeoff is deployment-specific; defaults match prior
    hard-coded ``medium`` so behavior is unchanged when unset.
    """
    default = "medium"
    if not config:
        return default
    raw = config.get("INTEL_QSV_ENCODE_PRESET", default)
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return default


def intel_qsv_encode_global_quality(config: dict | None) -> int:
    """Clamp ``-global_quality`` for QSV (typical usable range)."""
    default = 24
    if not config:
        return default
    try:
        q = int(config.get("INTEL_QSV_ENCODE_GLOBAL_QUALITY", default))
    except (TypeError, ValueError):
        return default
    return max(1, min(51, q))


def compilation_ffmpeg_cmd_and_log_path(
    tmp_output_path: str,
    target_w: int,
    target_h: int,
    *,
    config: dict | None = None,
) -> tuple[list[str], str]:
    """Build FFmpeg h264_qsv command and log path (Intel Arc / QSV encode)."""
    log_file_path = compilation_log_path(tmp_output_path)
    preset = intel_qsv_encode_preset(config)
    quality = intel_qsv_encode_global_quality(config)
    cmd = [
        "ffmpeg",
        "-y",
        *rawvideo_stdin_argv_fragment(target_w, target_h),
        "-c:v",
        "h264_qsv",
        "-preset",
        preset,
        "-global_quality",
        str(quality),
        "-an",
        "-pix_fmt",
        "yuv420p",
        tmp_output_path,
    ]
    return cmd, log_file_path


QSV_COMPILE_NOT_FOUND_USER_MSG = (
    "Compilation encode failed: ffmpeg not found. "
    "Intel path expects h264_qsv. Install FFmpeg with QSV support."
)


class IntelFfmpegCompilationEncode:
    """Concrete :class:`FfmpegCompilationEncodeProto` for QSV compilation."""

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


intel_ffmpeg_compilation_encode = IntelFfmpegCompilationEncode()
