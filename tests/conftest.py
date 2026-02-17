"""
Pytest conftest: inject a mock ffmpegcv module so app code can be imported without
requiring ffmpeg (and GPU) on the host. ffmpegcv runs _check() at import time and
raises if ffmpeg is not installed; tests patch VideoCaptureNV/VideoWriterNV
return values, so we only need the name to exist.
"""
import sys
from unittest.mock import MagicMock

if "ffmpegcv" not in sys.modules:
    _mock_ffmpegcv = MagicMock()
    _mock_ffmpegcv.VideoCaptureNV = MagicMock()
    _mock_ffmpegcv.VideoWriterNV = MagicMock()
    sys.modules["ffmpegcv"] = _mock_ffmpegcv
