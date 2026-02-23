"""
GPU-accelerated video sanitizer for NeLux.

Re-encodes corrupted or non-compliant clips to clean H.264 entirely on GPU (scale_cuda
caps width at 4096px for panoramic; h264_nvenc); frames stay in VRAM. NeLux never sees bad frames or HEVC. Uses a context manager
to guarantee strict cleanup of temporary files (RAM disk when available).
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from contextlib import contextmanager
from typing import Generator

logger = logging.getLogger("frigate-buffer")

# Prefer Linux RAM disk to avoid SSD wear in Docker/Unraid; fallback to system temp.
_SANITIZER_TEMP_DIR: str | None = "/dev/shm" if os.path.exists("/dev/shm") else None


@contextmanager
def sanitize_for_nelux(clip_path: str) -> Generator[str, None, None]:
    """
    Yield a path safe for NeLux VideoReader: either a sanitized temp file or the original.

    Re-encodes the clip with FFmpeg (CUDA decode + scale_cuda, h264_nvenc profile high, GOP=1, -b:v 30M)
    into a temp file on RAM disk when possible. 100% GPU; All-I-frame output so the downstream CPU reader can decode frames without NVDEC. On FFmpeg failure, logs stderr and yields clip_path
    so callers can attempt NeLux on the original and fail in a controlled way.
    Temp file is always removed in finally.
    """
    fd, temp_path = tempfile.mkstemp(
        suffix=".mp4", prefix="sanitized_", dir=_SANITIZER_TEMP_DIR
    )
    os.close(fd)
    try:
        try:
            subprocess.run([
                "ffmpeg", "-y",
                "-hwaccel", "cuda",
                "-hwaccel_output_format", "cuda",
                "-i", clip_path,
                "-vf", "scale_cuda=w='min(iw,4096)':h=-2",
                "-c:v", "h264_nvenc",
                "-preset", "p1",
                "-tune", "hq",
                "-profile:v", "high",
                "-g", "1",
                "-b:v", "30M",
                "-an",
                temp_path
            ], capture_output=True, text=True, check=True)
            yield temp_path
        except subprocess.CalledProcessError as e:
            logger.warning(
                "Video sanitizer FFmpeg failed for %s: %s",
                clip_path,
                e.stderr or e.stdout or str(e),
            )
            yield clip_path
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError as cleanup_err:
                logger.debug(
                    "Sanitizer cleanup failed for %s: %s",
                    temp_path,
                    cleanup_err,
                )
