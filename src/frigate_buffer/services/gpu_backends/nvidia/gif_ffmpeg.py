"""FFmpeg CUDA hwaccel argv + filter_complex for preview GIF generation."""

from __future__ import annotations

from frigate_buffer.services.gpu_backends.gif_common import build_gif_ffmpeg_argv


def gif_filter_complex(fps: int, preview_width: int) -> str:
    """
    filter_complex for NVDEC + scale_cuda, then palette for GIF.

    palettegen/paletteuse stay CPU-bound after hwdownload; matches prior video.py
    behavior.
    """
    return (
        f"scale_cuda={preview_width}:-1,"
        f"hwdownload,format=nv12,format=rgb24,"
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
    """Full argv for ffmpeg subprocess (CUDA decode/scale, no CPU fallback)."""
    return build_gif_ffmpeg_argv(
        clip_path,
        output_path,
        duration_sec,
        gif_filter_complex(fps, preview_width),
        ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"],
    )


class NvidiaGifFfmpeg:
    """Concrete :class:`GifFfmpegProto` for CUDA GIF pipeline."""

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


nvidia_gif_ffmpeg = NvidiaGifFfmpeg()
