"""
Temporary benchmarks for post-download, pre-API segment (sidecar → extract → overlay → write).
Run from project root: pytest scripts/bench_post_download_pre_api.py -v -s
Use BENCH_RUNS=5 to set repeat count (default 3). Results print mean ± stdev in seconds.
Phase 6: extract uses NeLux VideoReader; bench mocks nelux.VideoReader so no GPU required.
"""

import json
import os
import shutil
import sys
import time
import unittest
from unittest.mock import MagicMock, patch

# Ensure src is on path when run from scripts/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import numpy as np

# Fake nelux so extract_target_centric_frames can be called with mocked VideoReader
if "nelux" not in sys.modules:
    sys.modules["nelux"] = MagicMock()

# Fixture and segment helpers
from frigate_buffer.services.multi_clip_extractor import DETECTION_SIDECAR_FILENAME, extract_target_centric_frames
from frigate_buffer.managers.file import write_ai_frame_analysis_multi_cam
from frigate_buffer.models import ExtractedFrame

BENCH_RUNS = int(os.environ.get("BENCH_RUNS", 3))


def make_fixture_ce_folder(tmp_path: str, cameras: int = 2, num_sidecar_entries: int = 11) -> str:
    """Create a CE folder with camera subdirs, placeholder clips, and detection.json sidecars."""
    for i in range(cameras):
        cam_name = f"cam{i+1}"
        sub = os.path.join(tmp_path, cam_name)
        os.makedirs(sub, exist_ok=True)
        clip_path = os.path.join(sub, f"{cam_name}-00001.mp4")
        with open(clip_path, "wb"):
            pass
        # New dict format with entries + native_width/height
        sidecar = {
            "native_width": 1280,
            "native_height": 720,
            "entries": [
                {"timestamp_sec": float(t), "detections": [{"label": "person", "area": 100}]}
                for t in range(num_sidecar_entries)
            ],
        }
        with open(os.path.join(sub, DETECTION_SIDECAR_FILENAME), "w", encoding="utf-8") as f:
            json.dump(sidecar, f, separators=(",", ":"))
    return tmp_path


def run_extraction_only(ce_folder: str, config: dict | None = None) -> float:
    """Run extract_target_centric_frames only; return wall time in seconds."""
    config = config or {}
    t0 = time.perf_counter()
    extract_target_centric_frames(
        ce_folder,
        max_frames_sec=1.0,
        max_frames_min=10,
        config=config,
    )
    return time.perf_counter() - t0


def run_write_zip_only(ce_folder: str, num_frames: int = 10) -> float:
    """Run write_ai_frame_analysis_multi_cam (frames + zip); return wall time."""
    frames_data = [
        ExtractedFrame(frame=np.zeros((240, 320, 3), dtype=np.uint8), timestamp_sec=float(i), camera="cam1", metadata={})
        for i in range(num_frames)
    ]
    t0 = time.perf_counter()
    write_ai_frame_analysis_multi_cam(
        ce_folder,
        frames_data,
        write_manifest=True,
        create_zip_flag=True,
        save_frames=True,
    )
    return time.perf_counter() - t0


def run_segment_extract_and_write(ce_folder: str, config: dict | None = None) -> float:
    """Run extract_target_centric_frames + overlay + write_ai_frame_analysis_multi_cam (no API)."""
    config = config or {}
    t0 = time.perf_counter()
    frames_raw = extract_target_centric_frames(
        ce_folder,
        max_frames_sec=1.0,
        max_frames_min=10,
        config=config,
    )
    if not frames_raw:
        return time.perf_counter() - t0
    write_ai_frame_analysis_multi_cam(
        ce_folder,
        frames_raw,
        write_manifest=True,
        create_zip_flag=True,
        save_frames=True,
    )
    return time.perf_counter() - t0


def _mean_stdev(times: list[float]) -> tuple[float, float]:
    n = len(times)
    if n == 0:
        return 0.0, 0.0
    mean = sum(times) / n
    if n < 2:
        return mean, 0.0
    variance = sum((t - mean) ** 2 for t in times) / (n - 1)
    return mean, variance ** 0.5


def _make_nelux_reader_mock():
    """Mock NeLux VideoReader: fps, __len__, get_batch(indices) -> BCHW uint8 tensor."""
    try:
        import torch
    except ImportError:
        return None
    mock_reader = MagicMock()
    mock_reader.fps = 1.0
    mock_reader.__len__ = MagicMock(return_value=10)

    def get_batch(indices):
        return torch.zeros((len(indices), 3, 480, 640), dtype=torch.uint8)

    mock_reader.get_batch.side_effect = get_batch
    return mock_reader


class BenchPostDownloadPreApi(unittest.TestCase):
    """Benchmarks for post-download pre-API segment. Use mocks so no GPU required."""

    def setUp(self) -> None:
        self.tmp = None

    def tearDown(self) -> None:
        if self.tmp and os.path.isdir(self.tmp):
            shutil.rmtree(self.tmp, ignore_errors=True)

    @patch.object(sys.modules["nelux"], "VideoReader")
    def test_bench_extraction_only_baseline(self, mock_video_reader_cls):
        """Baseline: extract_target_centric_frames only (mocked NeLux decode)."""
        mock_reader = _make_nelux_reader_mock()
        if mock_reader is None:
            self.skipTest("torch not available")
        mock_video_reader_cls.return_value = mock_reader
        self.tmp = os.path.join(os.environ.get("TEMP", "/tmp"), "bench_ce_" + str(time.time()).replace(".", "_"))
        os.makedirs(self.tmp, exist_ok=True)
        make_fixture_ce_folder(self.tmp, cameras=2, num_sidecar_entries=11)
        times: list[float] = []
        for _ in range(BENCH_RUNS):
            times.append(run_extraction_only(self.tmp))
        mean, stdev = _mean_stdev(times)
        print(f"\n[bench] extraction_only_baseline: {mean:.3f} ± {stdev:.3f} s (n={BENCH_RUNS})")
        self.assertGreater(mean, 0)

    @patch.object(sys.modules["nelux"], "VideoReader")
    def test_bench_segment_extract_and_write_baseline(self, mock_video_reader_cls):
        """Baseline: extract + write_ai_frame_analysis_multi_cam (no API)."""
        mock_reader = _make_nelux_reader_mock()
        if mock_reader is None:
            self.skipTest("torch not available")
        mock_video_reader_cls.return_value = mock_reader
        self.tmp = os.path.join(os.environ.get("TEMP", "/tmp"), "bench_ce_" + str(time.time()).replace(".", "_"))
        os.makedirs(self.tmp, exist_ok=True)
        make_fixture_ce_folder(self.tmp, cameras=2, num_sidecar_entries=11)
        times: list[float] = []
        for _ in range(BENCH_RUNS):
            times.append(run_segment_extract_and_write(self.tmp))
        mean, stdev = _mean_stdev(times)
        print(f"\n[bench] segment_extract_and_write_baseline: {mean:.3f} ± {stdev:.3f} s (n={BENCH_RUNS})")
        self.assertGreater(mean, 0)

    def test_bench_write_zip_only(self):
        """Write + zip only (no decode)."""
        self.tmp = os.path.join(os.environ.get("TEMP", "/tmp"), "bench_write_" + str(time.time()).replace(".", "_"))
        os.makedirs(self.tmp, exist_ok=True)
        times: list[float] = []
        for _ in range(BENCH_RUNS):
            times.append(run_write_zip_only(self.tmp, num_frames=10))
        mean, stdev = _mean_stdev(times)
        print(f"\n[bench] write_zip_only: {mean:.3f} ± {stdev:.3f} s (n={BENCH_RUNS})")
        self.assertGreater(mean, 0)
