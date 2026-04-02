"""Shared FFmpeg argv fragments for preview GIF generation (multi-vendor DRY)."""

from __future__ import annotations


def cpu_palette_gif_filter_complex(fps: int, preview_width: int) -> str:
    """
    CPU scale (bilinear) + fps + palette after hardware decode.

    Why: Intel QSV and AMD VAAPI GIF paths use identical filter graphs once frames
    are in system memory; NVIDIA uses a separate CUDA scale_cuda graph instead.
    """
    return (
        f"scale={preview_width}:-1:flags=bilinear,"
        f"fps={fps},split[a][b];"
        f"[a]palettegen=stats_mode=single[p];"
        f"[b][p]paletteuse=dither=bayer"
    )


def build_gif_ffmpeg_argv(
    clip_path: str,
    output_path: str,
    duration_sec: float,
    filter_complex: str,
    hwaccel_args: list[str],
) -> list[str]:
    """
    Assemble ``ffmpeg`` argv: ``-y``, hardware-accel options, input, duration,
    ``filter_complex``, and output path.

    Why: Keeps Intel/AMD/NVIDIA branches aligned on ordering and flags that are
    common to all preview-GIF subprocesses; vendors only supply ``hwaccel_args``
    and the appropriate ``filter_complex`` string.
    """
    return [
        "ffmpeg",
        "-y",
        *hwaccel_args,
        "-i",
        clip_path,
        "-t",
        str(duration_sec),
        "-filter_complex",
        filter_complex,
        output_path,
    ]
