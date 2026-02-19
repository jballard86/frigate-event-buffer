"""Tests for multi-clip target-centric frame extractor."""

import json
import unittest
import os
import tempfile
import shutil
import numpy as np
from unittest.mock import MagicMock, patch

from frigate_buffer.services.multi_clip_extractor import (
    extract_target_centric_frames,
    _get_fps_and_duration,
    _nearest_sidecar_entry,
    _person_area_from_detections,
    _person_area_at_time,
    _load_sidecar_for_camera,
    _detection_timestamps_with_person,
    _subsample_with_min_gap,
    DETECTION_SIDECAR_FILENAME,
)


class TestMultiClipExtractorHelpers(unittest.TestCase):
    """Tests for sidecar helpers (person area, load)."""

    def test_person_area_from_detections_empty(self):
        """Empty or no detections returns 0."""
        self.assertEqual(_person_area_from_detections([]), 0.0)
        self.assertEqual(_person_area_from_detections(None), 0.0)

    def test_person_area_from_detections_sum_preferred_labels(self):
        """Sums area only for person/people/pedestrian (case-insensitive)."""
        dets = [
            {"label": "person", "area": 1000},
            {"label": "Person", "area": 500},
            {"label": "car", "area": 8000},
        ]
        self.assertEqual(_person_area_from_detections(dets), 1500.0)

    def test_person_area_from_detections_new_sidecar_format(self):
        """New sidecar format (label, bbox, centerpoint, area) is summed correctly."""
        dets = [
            {"label": "person", "bbox": [0, 0, 10, 20], "centerpoint": [5, 10], "area": 200},
            {"label": "person", "bbox": [10, 0, 30, 15], "centerpoint": [20, 7.5], "area": 300},
        ]
        self.assertEqual(_person_area_from_detections(dets), 500.0)

    def test_nearest_sidecar_entry_returns_none_for_empty(self):
        """Empty sidecar entries returns None."""
        self.assertIsNone(_nearest_sidecar_entry([], 0.5))

    def test_nearest_sidecar_entry_returns_entry_closest_to_t(self):
        """Returns entry with timestamp_sec closest to t_sec."""
        entries = [
            {"timestamp_sec": 0.0, "frame_number": 0, "detections": []},
            {"timestamp_sec": 1.0, "frame_number": 30, "detections": []},
            {"timestamp_sec": 2.0, "frame_number": 60, "detections": []},
        ]
        self.assertEqual(_nearest_sidecar_entry(entries, 0.6)["timestamp_sec"], 1.0)
        self.assertEqual(_nearest_sidecar_entry(entries, 0.0)["timestamp_sec"], 0.0)

    def test_person_area_at_time_empty_returns_zero(self):
        """Empty sidecar or missing detections defaults to 0."""
        self.assertEqual(_person_area_at_time([], 0.5), 0.0)
        self.assertEqual(_person_area_at_time([{"timestamp_sec": 0.0}], 0.0), 0.0)

    def test_person_area_at_time_nearest(self):
        """Returns person area from nearest entry by timestamp_sec."""
        entries = [
            {"timestamp_sec": 0.0, "detections": [{"label": "person", "area": 100}]},
            {"timestamp_sec": 1.0, "detections": [{"label": "person", "area": 200}]},
            {"timestamp_sec": 2.0, "detections": [{"label": "person", "area": 50}]},
        ]
        # t=0.6: nearest is 1.0 (distance 0.4), not 0.0 (0.6)
        self.assertEqual(_person_area_at_time(entries, 0.6), 200.0)
        self.assertEqual(_person_area_at_time(entries, 0.0), 100.0)

    def test_load_sidecar_missing_returns_none(self):
        """Missing file returns None."""
        test_dir = os.path.dirname(os.path.abspath(__file__))
        d = tempfile.mkdtemp(prefix="frigate_sidecar_", dir=test_dir)
        try:
            self.assertIsNone(_load_sidecar_for_camera(d))
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_load_sidecar_valid_returns_list(self):
        """Valid detection.json returns list of entries."""
        test_dir = os.path.dirname(os.path.abspath(__file__))
        d = tempfile.mkdtemp(prefix="frigate_sidecar_", dir=test_dir)
        try:
            path = os.path.join(d, DETECTION_SIDECAR_FILENAME)
            data = [{"timestamp_sec": 0.0, "detections": [{"label": "person", "area": 100}]}]
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f)
            result = _load_sidecar_for_camera(d)
            self.assertIsNotNone(result)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["timestamp_sec"], 0.0)
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_detection_timestamps_with_person_empty_sidecar_safe(self):
        """One camera has entries with person area, other is None/empty; no exception."""
        sidecars = {
            "cam1": [{"timestamp_sec": 0.0, "detections": [{"label": "person", "area": 100}]}],
            "cam2": None,
        }
        result = _detection_timestamps_with_person(sidecars, global_end=10.0)
        self.assertEqual(result, [0.0])
        sidecars["cam2"] = []
        result = _detection_timestamps_with_person(sidecars, global_end=10.0)
        self.assertEqual(result, [0.0])

    def test_detection_timestamps_with_person_filters_zero_area(self):
        """Only entries with person area > 0 are included."""
        sidecars = {
            "cam1": [
                {"timestamp_sec": 0.0, "detections": [{"label": "person", "area": 50}]},
                {"timestamp_sec": 1.0, "detections": []},
                {"timestamp_sec": 2.0, "detections": [{"label": "car", "area": 1000}]},
            ],
        }
        result = _detection_timestamps_with_person(sidecars, global_end=10.0)
        self.assertEqual(result, [0.0])

    def test_subsample_with_min_gap_enforces_step(self):
        """Selected timestamps have at least step_sec between them; not all from start."""
        timestamps = [0.0, 0.1, 0.2, 1.0, 1.05, 2.0, 2.1, 3.0]
        result = _subsample_with_min_gap(timestamps, step_sec=1.0, max_count=5)
        self.assertEqual(result, [0.0, 1.0, 2.0, 3.0])
        self.assertLessEqual(len(result), 5)

    def test_subsample_with_min_gap_respects_max_count(self):
        """Subsample stops at max_count."""
        timestamps = [float(i) for i in range(20)]
        result = _subsample_with_min_gap(timestamps, step_sec=1.0, max_count=3)
        self.assertEqual(len(result), 3)
        self.assertEqual(result, [0.0, 1.0, 2.0])


class TestMultiClipExtractor(unittest.TestCase):
    """Tests for extract_target_centric_frames."""

    def setUp(self):
        # Use a temp dir under the project so sandbox/Windows can access it
        self._test_dir = os.path.dirname(os.path.abspath(__file__))
        self.tmp = tempfile.mkdtemp(prefix="frigate_extractor_test_", dir=self._test_dir)

    def tearDown(self):
        if os.path.exists(self.tmp):
            shutil.rmtree(self.tmp, ignore_errors=True)

    def test_empty_ce_folder_returns_empty(self):
        """When CE folder has no camera subdirs with clips, returns empty list."""
        result = extract_target_centric_frames(self.tmp, max_frames_sec=1, max_frames_min=60)
        self.assertEqual(result, [])

    def test_no_clips_in_subdirs_returns_empty(self):
        """When subdirs exist but no clip.mp4, returns empty list."""
        os.makedirs(os.path.join(self.tmp, "Camera1"))
        os.makedirs(os.path.join(self.tmp, "Camera2"))
        result = extract_target_centric_frames(self.tmp, max_frames_sec=1, max_frames_min=60)
        self.assertEqual(result, [])

    def test_get_fps_and_duration_from_len_and_fps(self):
        """_get_fps_and_duration returns (fps, duration, False) when cap has fps and __len__ (no consume)."""
        cap = MagicMock()
        cap.fps = 10.0
        cap.__len__ = MagicMock(return_value=100)
        fps, duration, used = _get_fps_and_duration(cap, "/fake/path")
        self.assertEqual(fps, 10.0)
        self.assertEqual(duration, 10.0)
        self.assertFalse(used)
        cap.read.assert_not_called()

    def test_get_fps_and_duration_ffmpegcv_no_get(self):
        """_get_fps_and_duration works with FFmpegReaderNV-like mock: .fps and __len__, no .get()."""
        # FFmpegReaderNV has .fps, __len__ (via .count), but NO .get() - must never call cap.get()
        class FFmpegReaderNVMock:
            fps = 12.0
            count = 120

            def __len__(self):
                return self.count

            def read(self):
                return False, None

        cap = FFmpegReaderNVMock()
        fps, duration, used = _get_fps_and_duration(cap, "/fake/path")
        self.assertEqual(fps, 12.0)
        self.assertEqual(duration, 10.0)
        self.assertFalse(used)

    def test_get_fps_and_duration_read_to_eof_when_no_count(self):
        """_get_fps_and_duration uses read-to-EOF when cap has no __len__/count; returns used_read_to_eof=True."""
        # Use simple object: fps, read, no __len__, no count (ffmpegcv-style when metadata unknown)
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        reads = iter([(True, frame), (True, frame), (False, None)])

        class CapNoCount:
            fps = 1.0

            def read(self):
                return next(reads)

        cap = CapNoCount()
        fps, duration, used = _get_fps_and_duration(cap, "/fake/path")
        self.assertEqual(fps, 1.0)
        self.assertEqual(duration, 2.0)
        self.assertTrue(used)

    @patch("frigate_buffer.services.multi_clip_extractor.ffmpegcv.VideoCapture")
    @patch("frigate_buffer.services.multi_clip_extractor.ffmpegcv.VideoCaptureNV")
    def test_extract_with_ffmpegcv_like_mock_sequential_read(self, mock_video_capture_nv, mock_video_capture):
        """extract_target_centric_frames works with ffmpegcv-like readers (fps, read, no get/set)."""
        os.makedirs(os.path.join(self.tmp, "cam1"))
        os.makedirs(os.path.join(self.tmp, "cam2"))
        clip1 = os.path.join(self.tmp, "cam1", "clip.mp4")
        clip2 = os.path.join(self.tmp, "cam2", "clip.mp4")
        with open(clip1, "wb"):
            pass
        with open(clip2, "wb"):
            pass
        # Sidecars so we pick by person area (no HOG on mock frame); gives area > 0 so we collect frames
        for cam_name in ("cam1", "cam2"):
            sidecar_path = os.path.join(self.tmp, cam_name, DETECTION_SIDECAR_FILENAME)
            with open(sidecar_path, "w", encoding="utf-8") as f:
                json.dump(
                    [{"timestamp_sec": t, "detections": [{"label": "person", "area": 100}]} for t in range(11)],
                    f,
                )

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        # Each mock: fps=1, __len__=10 so duration=10, no .get/.set. read yields 10 frames then EOF.
        def make_mock():
            m = MagicMock()
            m.fps = 1.0
            m.__len__ = MagicMock(return_value=10)
            m.isOpened.return_value = True
            m.read.side_effect = [(True, frame)] * 10 + [(False, None)]
            m.release = MagicMock()
            return m

        mock_video_capture_nv.side_effect = Exception("no nv")
        mock_video_capture.side_effect = lambda path: make_mock()

        result = extract_target_centric_frames(
            self.tmp, max_frames_sec=1.0, max_frames_min=5
        )

        self.assertGreaterEqual(len(result), 1)
        self.assertLessEqual(len(result), 5)
        for item in result:
            self.assertEqual(len(item), 3)
            frame_out, t_sec, cam = item
            self.assertIsNotNone(frame_out)
            self.assertIn(cam, ("cam1", "cam2"))
        mock_video_capture.assert_called()

    @patch("frigate_buffer.services.multi_clip_extractor.ffmpegcv.VideoCapture")
    @patch("frigate_buffer.services.multi_clip_extractor.ffmpegcv.VideoCaptureNV")
    def test_extract_drops_camera_when_read_raises_closed_file(self, mock_video_capture_nv, mock_video_capture):
        """When one camera's cap.read() raises ValueError (e.g. read of closed file), that camera is dropped and extraction continues with the other(s)."""
        os.makedirs(os.path.join(self.tmp, "cam1"))
        os.makedirs(os.path.join(self.tmp, "cam2"))
        clip1 = os.path.join(self.tmp, "cam1", "clip.mp4")
        clip2 = os.path.join(self.tmp, "cam2", "clip.mp4")
        with open(clip1, "wb"):
            pass
        with open(clip2, "wb"):
            pass
        for cam_name in ("cam1", "cam2"):
            sidecar_path = os.path.join(self.tmp, cam_name, DETECTION_SIDECAR_FILENAME)
            with open(sidecar_path, "w", encoding="utf-8") as f:
                json.dump(
                    [{"timestamp_sec": t, "detections": [{"label": "person", "area": 100}]} for t in range(11)],
                    f,
                )

        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        def make_good_mock():
            m = MagicMock()
            m.fps = 1.0
            m.__len__ = MagicMock(return_value=10)
            m.isOpened.return_value = True
            m.read.side_effect = [(True, frame)] * 10 + [(False, None)]
            m.release = MagicMock()
            return m

        def make_bad_mock():
            m = MagicMock()
            m.fps = 1.0
            m.__len__ = MagicMock(return_value=10)
            m.isOpened.return_value = True
            m.read.side_effect = ValueError("read of closed file")
            m.release = MagicMock()
            return m

        def open_cap(path):
            if "cam2" in path:
                return make_bad_mock()
            return make_good_mock()

        mock_video_capture_nv.side_effect = Exception("no nv")
        mock_video_capture.side_effect = open_cap

        result = extract_target_centric_frames(
            self.tmp, max_frames_sec=1.0, max_frames_min=5
        )

        self.assertIsInstance(result, list)
        # Only cam1 should contribute (cam2 is dropped on first read failure).
        for item in result:
            self.assertEqual(len(item), 3)
            frame_out, t_sec, cam = item
            self.assertIsNotNone(frame_out)
            self.assertEqual(cam, "cam1", "Failed camera should be dropped; only cam1 should appear")

    @patch("frigate_buffer.services.multi_clip_extractor.ffmpegcv.VideoCapture")
    @patch("frigate_buffer.services.multi_clip_extractor.ffmpegcv.VideoCaptureNV")
    def test_decode_second_camera_cpu_only_logs_mixed_backends(self, mock_video_capture_nv, mock_video_capture):
        """When decode_second_camera_cpu_only is True and two cameras are used, log_callback receives mixed GPU/CPU message."""
        os.makedirs(os.path.join(self.tmp, "cam1"))
        os.makedirs(os.path.join(self.tmp, "cam2"))
        with open(os.path.join(self.tmp, "cam1", "clip.mp4"), "wb"):
            pass
        with open(os.path.join(self.tmp, "cam2", "clip.mp4"), "wb"):
            pass
        for cam_name in ("cam1", "cam2"):
            sidecar_path = os.path.join(self.tmp, cam_name, DETECTION_SIDECAR_FILENAME)
            with open(sidecar_path, "w", encoding="utf-8") as f:
                json.dump(
                    [{"timestamp_sec": t, "detections": [{"label": "person", "area": 100}]} for t in range(11)],
                    f,
                )

        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        def make_mock():
            m = MagicMock()
            m.fps = 1.0
            m.__len__ = MagicMock(return_value=10)
            m.isOpened.return_value = True
            m.read.side_effect = [(True, frame)] * 10 + [(False, None)]
            m.release = MagicMock()
            return m

        # open_caps: NVDEC tried for both and fails; then CPU for both to get fps/duration. Reopen: cam1 NVDEC, cam2 CPU only.
        mock_video_capture_nv.side_effect = [
            Exception("nv"),
            Exception("nv"),
            make_mock(),
        ]
        mock_video_capture.side_effect = lambda path: make_mock()

        logs = []
        result = extract_target_centric_frames(
            self.tmp,
            max_frames_sec=1.0,
            max_frames_min=5,
            decode_second_camera_cpu_only=True,
            log_callback=logs.append,
        )

        self.assertGreaterEqual(len(result), 1)
        decode_logs = [m for m in logs if "Decoding clips" in m]
        self.assertGreaterEqual(len(decode_logs), 1)
        self.assertTrue(
            any("GPU" in m and "CPU" in m for m in decode_logs),
            f"Expected a mixed GPU/CPU decode message in {decode_logs}",
        )
