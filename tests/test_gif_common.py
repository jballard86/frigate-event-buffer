"""Unit tests for shared GPU preview-GIF FFmpeg argv helpers."""

from __future__ import annotations

from frigate_buffer.services.gpu_backends.gif_common import (
    build_gif_ffmpeg_argv,
    cpu_palette_gif_filter_complex,
)


def test_cpu_palette_gif_filter_complex_matches_expected_shape() -> None:
    fc = cpu_palette_gif_filter_complex(fps=10, preview_width=400)
    assert "scale=400:-1:flags=bilinear" in fc
    assert "fps=10" in fc
    assert "palettegen=stats_mode=single" in fc
    assert "paletteuse=dither=bayer" in fc


def test_build_gif_ffmpeg_argv_ordering() -> None:
    argv = build_gif_ffmpeg_argv(
        "in.mp4",
        "out.gif",
        2.5,
        "fc_here",
        ["-hwaccel", "qsv"],
    )
    assert argv == [
        "ffmpeg",
        "-y",
        "-hwaccel",
        "qsv",
        "-i",
        "in.mp4",
        "-t",
        "2.5",
        "-filter_complex",
        "fc_here",
        "out.gif",
    ]
