"""Unit tests for shared compilation FFmpeg argv helpers."""

from __future__ import annotations

import os

from frigate_buffer.constants import COMPILATION_OUTPUT_FPS
from frigate_buffer.services.gpu_backends.compilation_argv_common import (
    compilation_log_path,
    rawvideo_stdin_argv_fragment,
)


def test_compilation_log_path_beside_output() -> None:
    nested = os.path.join("tmpdir", "nested", "clip.mp4")
    assert compilation_log_path(nested) == os.path.join(
        os.path.dirname(nested), "ffmpeg_compile.log"
    )
    assert os.path.basename(compilation_log_path("clip.mp4")) == "ffmpeg_compile.log"


def test_rawvideo_stdin_argv_fragment_default_fps() -> None:
    frag = rawvideo_stdin_argv_fragment(640, 480)
    assert frag == [
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        "640x480",
        "-r",
        str(COMPILATION_OUTPUT_FPS),
        "-thread_queue_size",
        "512",
        "-i",
        "pipe:0",
    ]


def test_rawvideo_stdin_argv_fragment_explicit_fps() -> None:
    frag = rawvideo_stdin_argv_fragment(1280, 720, output_fps=15)
    assert frag[5] == "1280x720"
    idx_r = frag.index("-r")
    assert frag[idx_r + 1] == "15"
