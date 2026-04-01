"""FFmpeg VAAPI hwaccel argv + filter_complex for preview GIF (Linux-first)."""

from __future__ import annotations

# Default DRM render node; override per host if decode fails (e.g. renderD129).
DEFAULT_VAAPI_RENDER_NODE = "/dev/dri/renderD128"


def gif_filter_complex(fps: int, preview_width: int) -> str:
    """CPU scale and palette after VAAPI decode (same structure as Intel QSV GIF)."""
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
    """VAAPI hwaccel decode; CPU scale and palette for GIF (gpu-03 Linux-first)."""
    filter_str = gif_filter_complex(fps, preview_width)
    return [
        "ffmpeg",
        "-y",
        "-hwaccel",
        "vaapi",
        "-hwaccel_device",
        DEFAULT_VAAPI_RENDER_NODE,
        "-i",
        clip_path,
        "-t",
        str(duration_sec),
        "-filter_complex",
        filter_str,
        output_path,
    ]


class AmdGifFfmpeg:
    """Concrete :class:`GifFfmpegProto` for VAAPI GIF pipeline."""

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


amd_gif_ffmpeg = AmdGifFfmpeg()
