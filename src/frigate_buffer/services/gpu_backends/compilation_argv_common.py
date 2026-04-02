"""Shared FFmpeg argv fragments for compilation encode (rawvideo on stdin)."""

from __future__ import annotations

import os

from frigate_buffer.constants import COMPILATION_OUTPUT_FPS


def compilation_log_path(tmp_output_path: str) -> str:
    """Path for FFmpeg stderr log next to the compilation output file."""
    return os.path.join(os.path.dirname(tmp_output_path), "ffmpeg_compile.log")


def rawvideo_stdin_argv_fragment(
    target_w: int,
    target_h: int,
    *,
    output_fps: int | None = None,
) -> list[str]:
    """
    FFmpeg input argv from raw RGB24 stdin through ``pipe:0``.

    Why: NVIDIA NVENC, Intel QSV, and AMD AMF compilation all use the same
    rawvideo graph before the codec-specific ``-c:v`` block; centralizing this
    avoids argv drift when one vendor is fixed.
    """
    fps = output_fps if output_fps is not None else COMPILATION_OUTPUT_FPS
    return [
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{target_w}x{target_h}",
        "-r",
        str(fps),
        "-thread_queue_size",
        "512",
        "-i",
        "pipe:0",
    ]
