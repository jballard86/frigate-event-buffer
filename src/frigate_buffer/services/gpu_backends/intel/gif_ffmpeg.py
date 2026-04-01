"""FFmpeg QSV hwaccel argv + filter_complex for preview GIF generation."""

from __future__ import annotations


def gif_filter_complex(fps: int, preview_width: int) -> str:
    """scale + fps + palette after QSV decode (hw frames downloaded in graph)."""
    return (
        f"scale={preview_width}:-1:flags=bilinear,"
        f"fps={fps},split[a][b];"
        f"[a]palettegen=stats_mode=single[p];"
        f"[b][p]paletteuse=dither=bayer"
    )


def gif_ffmpeg_argv(
    clip_path: str,
    output_path: str,
    fps: int,
    duration_sec: float,
    preview_width: int,
) -> list[str]:
    """QSV hwaccel decode; CPU scale and palette for GIF."""
    filter_str = gif_filter_complex(fps, preview_width)
    return [
        "ffmpeg",
        "-y",
        "-hwaccel",
        "qsv",
        "-i",
        clip_path,
        "-t",
        str(duration_sec),
        "-filter_complex",
        filter_str,
        output_path,
    ]


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
