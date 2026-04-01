"""Optional smoke tests for native frigate_intel_decode (gpu-02 Phase 1).

The extension is built only on Linux with FFmpeg + libtorch; CI without a build
skips these tests via importorskip.
"""

from __future__ import annotations

import importlib.util

import pytest

frigate_intel_decode = pytest.importorskip(
    "frigate_intel_decode",
    reason="frigate_intel_decode not built (see native/intel_decode/README.md)",
)


def test_version_string() -> None:
    v = frigate_intel_decode.version()
    assert isinstance(v, str)
    assert "phase2" in v or "spike" in v


def test_decode_first_frame_requires_existing_file() -> None:
    with pytest.raises(RuntimeError, match="avformat_open_input"):
        frigate_intel_decode.decode_first_frame_bchw_rgb("/nonexistent/clip.mp4")


def test_module_spec_when_imported() -> None:
    assert importlib.util.find_spec("frigate_intel_decode") is not None
