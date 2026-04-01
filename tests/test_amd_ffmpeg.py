"""Argv snapshot tests for AMD h264_amf compilation encode and VAAPI GIF (gpu-03 §6)."""

from __future__ import annotations

import os

from frigate_buffer.constants import COMPILATION_OUTPUT_FPS
from frigate_buffer.services.gpu_backends.amd.ffmpeg_encode import (
    AmdFfmpegCompilationEncode,
    amd_ffmpeg_compilation_encode,
    compilation_ffmpeg_cmd_and_log_path,
)
from frigate_buffer.services.gpu_backends.amd.gif_ffmpeg import (
    DEFAULT_VAAPI_RENDER_NODE,
    AmdGifFfmpeg,
    amd_gif_ffmpeg,
    gif_ffmpeg_argv,
    gif_filter_complex,
)


def test_compilation_cmd_uses_h264_amf_not_libx264() -> None:
    cmd, _log = compilation_ffmpeg_cmd_and_log_path("/tmp/out/x.mp4", 1280, 720)
    joined = " ".join(cmd)
    assert "h264_amf" in joined
    assert "libx264" not in joined
    assert "-c:v" in cmd
    idx = cmd.index("-c:v")
    assert cmd[idx + 1] == "h264_amf"


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


def test_compilation_cmd_amf_quality_balanced() -> None:
    cmd, _log = compilation_ffmpeg_cmd_and_log_path("/tmp/out/x.mp4", 640, 480)
    assert "-quality" in cmd
    assert cmd[cmd.index("-quality") + 1] == "balanced"


def test_compilation_cmd_config_reserved_no_op_for_now() -> None:
    cmd_none, _ = compilation_ffmpeg_cmd_and_log_path("/tmp/a.mp4", 320, 240)
    cmd_cfg, _ = compilation_ffmpeg_cmd_and_log_path(
        "/tmp/a.mp4", 320, 240, config={"AMD_PLACEHOLDER": 1}
    )
    assert cmd_none == cmd_cfg


def test_compilation_log_path_same_dir_as_output() -> None:
    out = os.path.join("var", "tmp", "clip", "out.mp4")
    _cmd, log = compilation_ffmpeg_cmd_and_log_path(out, 640, 480)
    assert log.endswith("ffmpeg_compile.log")
    assert os.path.dirname(os.path.abspath(log)) == os.path.dirname(
        os.path.abspath(out)
    )


def test_amd_ffmpeg_compilation_encode_singleton_matches_free_function() -> None:
    tmp = "/tmp/z/out.mp4"
    a, la = amd_ffmpeg_compilation_encode.compilation_ffmpeg_cmd_and_log_path(
        tmp, 1920, 1080, config=None
    )
    b, lb = compilation_ffmpeg_cmd_and_log_path(tmp, 1920, 1080, config=None)
    assert a == b
    assert la == lb


def test_amd_ffmpeg_class_is_proto_shaped() -> None:
    enc = AmdFfmpegCompilationEncode()
    cmd, _ = enc.compilation_ffmpeg_cmd_and_log_path("o.mp4", 100, 100)
    assert "h264_amf" in cmd


def test_gif_argv_hwaccel_vaapi_device_input_and_duration() -> None:
    argv = gif_ffmpeg_argv(
        clip_path="/clips/a.mp4",
        output_path="/out/g.gif",
        fps=8,
        duration_sec=3.5,
        preview_width=320,
    )
    assert argv[0] == "ffmpeg"
    assert "-hwaccel" in argv
    assert argv[argv.index("-hwaccel") + 1] == "vaapi"
    assert "-hwaccel_device" in argv
    assert argv[argv.index("-hwaccel_device") + 1] == DEFAULT_VAAPI_RENDER_NODE
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


def test_amd_gif_singleton_matches_helpers() -> None:
    assert amd_gif_ffmpeg.gif_filter_complex(5, 200) == gif_filter_complex(5, 200)
    a = amd_gif_ffmpeg.gif_ffmpeg_argv("a.mp4", "b.gif", 6, 2.0, 240)
    b = gif_ffmpeg_argv("a.mp4", "b.gif", 6, 2.0, 240)
    assert a == b


def test_amd_gif_class_is_proto_shaped() -> None:
    g = AmdGifFfmpeg()
    joined = " ".join(g.gif_ffmpeg_argv("in.mp4", "out.gif", 4, 1.0, 100))
    assert "vaapi" in joined
    assert DEFAULT_VAAPI_RENDER_NODE in joined
