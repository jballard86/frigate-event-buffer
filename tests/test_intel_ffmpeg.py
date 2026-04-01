"""Argv snapshot tests for Intel QSV compilation encode and GIF FFmpeg (gpu-02 §8)."""

from __future__ import annotations

import os

from frigate_buffer.services.gpu_backends.intel.ffmpeg_encode import (
    IntelFfmpegCompilationEncode,
    compilation_ffmpeg_cmd_and_log_path,
    intel_ffmpeg_compilation_encode,
    intel_qsv_encode_global_quality,
    intel_qsv_encode_preset,
)
from frigate_buffer.services.gpu_backends.intel.gif_ffmpeg import (
    IntelGifFfmpeg,
    gif_ffmpeg_argv,
    gif_filter_complex,
    intel_gif_ffmpeg,
)
from frigate_buffer.constants import COMPILATION_OUTPUT_FPS


def test_compilation_cmd_uses_h264_qsv_not_libx264() -> None:
    cmd, _log = compilation_ffmpeg_cmd_and_log_path("/tmp/out/x.mp4", 1280, 720)
    joined = " ".join(cmd)
    assert "h264_qsv" in joined
    assert "libx264" not in joined
    assert "-c:v" in cmd
    idx = cmd.index("-c:v")
    assert cmd[idx + 1] == "h264_qsv"


def test_compilation_cmd_rawvideo_rgb24_dimensions_and_fps() -> None:
    cmd, _log = compilation_ffmpeg_cmd_and_log_path("/tmp/out/x.mp4", 800, 600)
    assert "-f" in cmd
    assert cmd[cmd.index("-f") + 1] == "rawvideo"
    assert "-pix_fmt" in cmd
    assert cmd[cmd.index("-pix_fmt") + 1] == "rgb24"
    assert "-s" in cmd
    assert cmd[cmd.index("-s") + 1] == "800x600"
    assert str(COMPILATION_OUTPUT_FPS) in cmd
    assert "-thread_queue_size" in cmd
    assert cmd[cmd.index("-thread_queue_size") + 1] == "512"


def test_compilation_log_path_same_dir_as_output() -> None:
    out = os.path.join("var", "tmp", "clip", "out.mp4")
    _cmd, log = compilation_ffmpeg_cmd_and_log_path(out, 640, 480)
    assert log.endswith("ffmpeg_compile.log")
    assert os.path.dirname(os.path.abspath(log)) == os.path.dirname(
        os.path.abspath(out)
    )


def test_intel_ffmpeg_compilation_encode_singleton_matches_free_function() -> None:
    tmp = "/tmp/z/out.mp4"
    a, la = intel_ffmpeg_compilation_encode.compilation_ffmpeg_cmd_and_log_path(
        tmp, 1920, 1080, config=None
    )
    b, lb = compilation_ffmpeg_cmd_and_log_path(tmp, 1920, 1080, config=None)
    assert a == b
    assert la == lb


def test_intel_ffmpeg_class_is_proto_shaped() -> None:
    enc = IntelFfmpegCompilationEncode()
    cmd, _ = enc.compilation_ffmpeg_cmd_and_log_path("o.mp4", 100, 100)
    assert "h264_qsv" in cmd


def test_intel_qsv_preset_and_quality_helpers() -> None:
    assert intel_qsv_encode_preset(None) == "medium"
    assert intel_qsv_encode_global_quality(None) == 24
    assert intel_qsv_encode_preset({"INTEL_QSV_ENCODE_PRESET": "  slow  "}) == "slow"
    assert (
        intel_qsv_encode_global_quality({"INTEL_QSV_ENCODE_GLOBAL_QUALITY": 30}) == 30
    )
    assert (
        intel_qsv_encode_global_quality({"INTEL_QSV_ENCODE_GLOBAL_QUALITY": 99}) == 51
    )
    assert intel_qsv_encode_global_quality({"INTEL_QSV_ENCODE_GLOBAL_QUALITY": 0}) == 1


def test_compilation_cmd_respects_config_qsv_options() -> None:
    cfg = {
        "INTEL_QSV_ENCODE_PRESET": "veryslow",
        "INTEL_QSV_ENCODE_GLOBAL_QUALITY": 28,
    }
    cmd, _log = compilation_ffmpeg_cmd_and_log_path(
        "/tmp/out/x.mp4", 640, 480, config=cfg
    )
    assert cmd[cmd.index("-preset") + 1] == "veryslow"
    assert cmd[cmd.index("-global_quality") + 1] == "28"


def test_gif_argv_hwaccel_qsv_input_and_duration() -> None:
    argv = gif_ffmpeg_argv(
        clip_path="/clips/a.mp4",
        output_path="/out/g.gif",
        fps=8,
        duration_sec=3.5,
        preview_width=320,
    )
    assert argv[0] == "ffmpeg"
    assert "-hwaccel" in argv
    assert argv[argv.index("-hwaccel") + 1] == "qsv"
    assert "/clips/a.mp4" in argv
    assert "-t" in argv
    assert argv[argv.index("-t") + 1] == "3.5"
    assert "-filter_complex" in argv
    assert argv[-1] == "/out/g.gif"


def test_gif_filter_complex_scale_fps_palette() -> None:
    fc = gif_filter_complex(fps=10, preview_width=400)
    assert "scale=400:-1:flags=bilinear" in fc
    assert "fps=10" in fc
    assert "palettegen" in fc
    assert "paletteuse" in fc


def test_intel_gif_singleton_matches_helpers() -> None:
    assert intel_gif_ffmpeg.gif_filter_complex(5, 200) == gif_filter_complex(5, 200)
    a = intel_gif_ffmpeg.gif_ffmpeg_argv("a.mp4", "b.gif", 6, 2.0, 240)
    b = gif_ffmpeg_argv("a.mp4", "b.gif", 6, 2.0, 240)
    assert a == b


def test_intel_gif_class_is_proto_shaped() -> None:
    g = IntelGifFfmpeg()
    assert "qsv" in " ".join(g.gif_ffmpeg_argv("in.mp4", "out.gif", 4, 1.0, 100))
