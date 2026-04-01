"""Optional smoke tests for native frigate_amd_decode (gpu-03 Phase 3).

Built on Linux with FFmpeg + libtorch; CI without the module skips via importorskip.
"""

from __future__ import annotations

import importlib.util

import pytest

frigate_amd_decode = pytest.importorskip(
    "frigate_amd_decode",
    reason="frigate_amd_decode not built (see native/amd_decode/README.md)",
)


def test_version_string() -> None:
    v = frigate_amd_decode.version()
    assert isinstance(v, str)
    assert "phase3" in v or "0.1.0" in v


def test_decode_first_frame_requires_existing_file() -> None:
    with pytest.raises(RuntimeError, match="avformat_open_input"):
        frigate_amd_decode.decode_first_frame_bchw_rgb("/nonexistent/clip.mp4")


def test_module_spec_when_imported() -> None:
    assert importlib.util.find_spec("frigate_amd_decode") is not None
