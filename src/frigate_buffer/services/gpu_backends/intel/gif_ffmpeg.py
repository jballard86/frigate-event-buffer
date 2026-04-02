"""FFmpeg QSV hwaccel argv + filter_complex for preview GIF generation."""

from __future__ import annotations

from frigate_buffer.services.gpu_backends.gif_common import (
    build_gif_ffmpeg_argv,
    cpu_palette_gif_filter_complex,
)


def gif_filter_complex(fps: int, preview_width: int) -> str:
    """scale + fps + palette after QSV decode (hw frames downloaded in graph)."""
    return cpu_palette_gif_filter_complex(fps, preview_width)


def gif_ffmpeg_argv(
    clip_path: str,
    output_path: str,
    fps: int,
    duration_sec: float,
    preview_width: int,
) -> list[str]:
    """QSV hwaccel decode; CPU scale and palette for GIF."""
    return build_gif_ffmpeg_argv(
        clip_path,
        output_path,
        duration_sec,
        gif_filter_complex(fps, preview_width),
        ["-hwaccel", "qsv"],
    )


class IntelGifFfmpeg:
    """Concrete :class:`GifFfmpegProto` for QSV GIF pipeline."""

    def gif_filter_complex(self, fps: int, preview_width: int) -> str:
        return gif_filter_complex(fps, preview_width)

    def gif_ffmpeg_argv(
        self,
        clip_path: str,
        output_path: str,
        fps: int,
        duration_sec: float,
        preview_width: int,
    ) -> list[str]:
        return gif_ffmpeg_argv(clip_path, output_path, fps, duration_sec, preview_width)


intel_gif_ffmpeg = IntelGifFfmpeg()
