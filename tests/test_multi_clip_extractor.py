"""Tests for multi-clip target-centric frame extractor."""

import json
import os
import shutil
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

# Fake nelux so we can patch VideoReader when the real wheel is not installed (e.g. on Windows).
if "nelux" not in sys.modules:
    sys.modules["nelux"] = MagicMock()

from frigate_buffer.models import ExtractedFrame
from frigate_buffer.services.multi_clip_extractor import (
    extract_target_centric_frames,
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
        d = os.path.join(test_dir, "_sidecar_fixture", "missing_" + str(id(self)))
        os.makedirs(d, exist_ok=True)
        try:
            self.assertIsNone(_load_sidecar_for_camera(d))
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_load_sidecar_valid_returns_list(self):
        """Valid detection.json (legacy list) returns (entries, native_w, native_h) with entries list."""
        test_dir = os.path.dirname(os.path.abspath(__file__))
        d = os.path.join(test_dir, "_sidecar_fixture", str(id(self)))
        os.makedirs(d, exist_ok=True)
        try:
            path = os.path.join(d, DETECTION_SIDECAR_FILENAME)
            data = [{"timestamp_sec": 0.0, "detections": [{"label": "person", "area": 100}]}]
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f)
            result = _load_sidecar_for_camera(d)
            self.assertIsNotNone(result)
            entries, nw, nh = result
            self.assertIsInstance(entries, list)
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0]["timestamp_sec"], 0.0)
            self.assertEqual(nw, 0)
            self.assertEqual(nh, 0)
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
        # Use a deterministic workspace path so sandbox/Windows can create subdirs (mkdtemp can block makedirs on Windows).
        self._test_dir = os.path.dirname(os.path.abspath(__file__))
        base = os.path.join(self._test_dir, "_extractor_fixture")
        self.tmp = os.path.join(base, str(id(self)))
        os.makedirs(self.tmp, exist_ok=True)

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

    def _make_nelux_reader_mock(self, fps=1.0, frame_count=10, height=480, width=640):
        """Create a mock NeLux VideoReader: fps, __len__, get_batch(indices) -> BCHW uint8 tensor."""
        try:
            import torch
        except ImportError:
            self.skipTest("torch not available")
        mock_reader = MagicMock()
        mock_reader.fps = fps
        mock_reader.__len__ = MagicMock(return_value=frame_count)

        def get_batch(indices):
            return torch.zeros((len(indices), 3, height, width), dtype=torch.uint8)

        mock_reader.get_batch.side_effect = get_batch
        return mock_reader

    @patch.object(sys.modules["nelux"], "VideoReader")
    def test_extract_with_nelux_mock_returns_tensor_frames(self, mock_video_reader_cls):
        """extract_target_centric_frames uses NeLux get_batch; returns ExtractedFrame with tensor .frame."""
        os.makedirs(os.path.join(self.tmp, "cam1"))
        os.makedirs(os.path.join(self.tmp, "cam2"))
        for c in ("cam1", "cam2"):
            with open(os.path.join(self.tmp, c, "clip.mp4"), "wb"):
                pass
            sidecar_path = os.path.join(self.tmp, c, DETECTION_SIDECAR_FILENAME)
            with open(sidecar_path, "w", encoding="utf-8") as f:
                json.dump(
                    [{"timestamp_sec": t, "detections": [{"label": "person", "area": 100}]} for t in range(11)],
                    f,
                )
        mock_reader = self._make_nelux_reader_mock()
        mock_reader._decoder = MagicMock()
        mock_video_reader_cls.return_value = mock_reader

        result = extract_target_centric_frames(self.tmp, max_frames_sec=1.0, max_frames_min=5)

        self.assertGreaterEqual(len(result), 1)
        self.assertLessEqual(len(result), 5)
        for item in result:
            self.assertIsInstance(item, ExtractedFrame)
            self.assertIsNotNone(item.frame)
            self.assertTrue(
                hasattr(item.frame, "shape") or type(item.frame).__name__ == "Tensor",
                "ExtractedFrame.frame should be tensor (BCHW)",
            )
            self.assertIn(item.camera, ("cam1", "cam2"))
            self.assertIn("person_area", item.metadata)
            self.assertIsInstance(item.metadata["person_area"], int)
            self.assertEqual(item.metadata["person_area"], 100)
        mock_video_reader_cls.assert_called()
        # Phase 5: sequential extractor — one get_batch per sample time that yields a frame
        mock_reader.get_batch.assert_called()
        self.assertEqual(mock_reader.get_batch.call_count, len(result))

    @patch.object(sys.modules["nelux"], "VideoReader")
    def test_extract_returns_empty_when_reader_has_no_decoder(self, mock_video_reader_cls):
        """When VideoReader has no _decoder, open phase fails and extract returns [] (no get_batch)."""
        os.makedirs(os.path.join(self.tmp, "cam1"))
        with open(os.path.join(self.tmp, "cam1", "clip.mp4"), "wb"):
            pass
        sidecar_path = os.path.join(self.tmp, "cam1", DETECTION_SIDECAR_FILENAME)
        with open(sidecar_path, "w", encoding="utf-8") as f:
            json.dump(
                {"native_width": 100, "native_height": 100, "entries": [{"timestamp_sec": 0.0, "detections": [{"label": "person", "area": 100}]}]},
                f,
            )
        reader_no_decoder = MagicMock(spec=["fps", "release"])
        reader_no_decoder.fps = 1.0
        reader_no_decoder.get_batch = MagicMock()
        mock_video_reader_cls.return_value = reader_no_decoder

        result = extract_target_centric_frames(self.tmp, max_frames_sec=1.0, max_frames_min=5)

        self.assertEqual(result, [])
        reader_no_decoder.get_batch.assert_not_called()

    @patch.object(sys.modules["nelux"], "VideoReader")
    def test_extract_drops_camera_when_get_batch_raises(self, mock_video_reader_cls):
        """When one camera's get_batch raises, that camera is dropped and extraction continues with the other(s)."""
        os.makedirs(os.path.join(self.tmp, "cam1"))
        os.makedirs(os.path.join(self.tmp, "cam2"))
        for c in ("cam1", "cam2"):
            with open(os.path.join(self.tmp, c, "clip.mp4"), "wb"):
                pass
            sidecar_path = os.path.join(self.tmp, c, DETECTION_SIDECAR_FILENAME)
            with open(sidecar_path, "w", encoding="utf-8") as f:
                json.dump(
                    [{"timestamp_sec": t, "detections": [{"label": "person", "area": 100}]} for t in range(11)],
                    f,
                )

        def make_reader(path):
            if "cam2" in path:
                m = MagicMock()
                m.fps = 1.0
                m.__len__ = MagicMock(return_value=10)
                m.get_batch.side_effect = RuntimeError("NeLux get_batch failed")
                return m
            return self._make_nelux_reader_mock()

        mock_video_reader_cls.side_effect = make_reader

        result = extract_target_centric_frames(self.tmp, max_frames_sec=1.0, max_frames_min=5)

        self.assertIsInstance(result, list)
        for item in result:
            self.assertIsInstance(item, ExtractedFrame)
            self.assertIsNotNone(item.frame)
            self.assertEqual(item.camera, "cam1", "Failed camera should be dropped; only cam1 should appear")

    @patch("frigate_buffer.services.multi_clip_extractor._get_fps_duration_from_path")
    @patch.object(sys.modules["nelux"], "VideoReader")
    def test_extract_uses_metadata_fallback_when_len_reader_raises(
        self, mock_video_reader_cls, mock_get_fps_duration
    ):
        """When NeLux reader raises on len() (e.g. missing _decoder), frame count comes from _get_fps_duration_from_path and extraction does not crash."""
        os.makedirs(os.path.join(self.tmp, "cam1"))
        with open(os.path.join(self.tmp, "cam1", "clip.mp4"), "wb"):
            pass
        sidecar_path = os.path.join(self.tmp, "cam1", DETECTION_SIDECAR_FILENAME)
        with open(sidecar_path, "w", encoding="utf-8") as f:
            json.dump(
                {"native_width": 100, "native_height": 100, "entries": [{"timestamp_sec": t, "detections": [{"label": "person", "area": 100}]} for t in range(11)]},
                f,
            )
        mock_get_fps_duration.return_value = (1.0, 10.0)
        mock_reader = self._make_nelux_reader_mock(fps=1.0, frame_count=10)
        mock_reader._decoder = MagicMock()
        mock_reader.__len__ = MagicMock(side_effect=AttributeError("'VideoReader' object has no attribute '_decoder'"))
        mock_video_reader_cls.return_value = mock_reader

        result = extract_target_centric_frames(self.tmp, max_frames_sec=1.0, max_frames_min=5)

        self.assertGreaterEqual(len(result), 1)
        for item in result:
            self.assertIsInstance(item, ExtractedFrame)
            self.assertIsNotNone(item.frame)
        mock_get_fps_duration.assert_called()

    @patch.object(sys.modules["nelux"], "VideoReader")
    def test_extract_logs_nelux_nvdec(self, mock_video_reader_cls):
        """With NeLux, log_callback receives Decoding clips ... NeLux NVDEC."""
        os.makedirs(os.path.join(self.tmp, "cam1"))
        with open(os.path.join(self.tmp, "cam1", "clip.mp4"), "wb"):
            pass
        sidecar_path = os.path.join(self.tmp, "cam1", DETECTION_SIDECAR_FILENAME)
        with open(sidecar_path, "w", encoding="utf-8") as f:
            json.dump(
                {"native_width": 100, "native_height": 100, "entries": [{"timestamp_sec": t, "detections": [{"label": "person", "area": 100}]} for t in range(11)]},
                f,
            )
        mock_video_reader_cls.return_value = self._make_nelux_reader_mock()
        logs = []
        result = extract_target_centric_frames(
            self.tmp, max_frames_sec=1.0, max_frames_min=5, log_callback=logs.append
        )
        self.assertGreaterEqual(len(result), 1)
        decode_logs = [m for m in logs if "Decoding clips" in m]
        self.assertGreaterEqual(len(decode_logs), 1)
        self.assertTrue(
            any("NeLux" in m or "NVDEC" in m for m in decode_logs),
            f"Expected NeLux/NVDEC in decode message: {decode_logs}",
        )

    @patch.object(sys.modules["nelux"], "VideoReader")
    def test_primary_camera_accepted(self, mock_video_reader_cls):
        """extract_target_centric_frames accepts primary_camera (EMA pipeline); returns frames without error."""
        os.makedirs(os.path.join(self.tmp, "cam1"))
        os.makedirs(os.path.join(self.tmp, "cam2"))
        for c in ("cam1", "cam2"):
            with open(os.path.join(self.tmp, c, "clip.mp4"), "wb"):
                pass
            sidecar_path = os.path.join(self.tmp, c, DETECTION_SIDECAR_FILENAME)
            with open(sidecar_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"native_width": 100, "native_height": 100, "entries": [{"timestamp_sec": t, "detections": [{"label": "person", "area": 100}]} for t in range(11)]},
                    f,
                )
        mock_video_reader_cls.return_value = self._make_nelux_reader_mock()

        result = extract_target_centric_frames(
            self.tmp, max_frames_sec=1.0, max_frames_min=5, primary_camera="cam1"
        )
        self.assertGreaterEqual(len(result), 1)
        for item in result:
            self.assertIsInstance(item, ExtractedFrame)
            self.assertIn(item.camera, ("cam1", "cam2"))

    @patch.object(sys.modules["nelux"], "VideoReader")
    def test_ema_drop_no_person_excludes_zero_area_frames(self, mock_video_reader_cls):
        """With camera_timeline_final_yolo_drop_no_person=True, frames with person_area=0 are not in the result."""
        os.makedirs(os.path.join(self.tmp, "cam1"))
        os.makedirs(os.path.join(self.tmp, "cam2"))
        for c in ("cam1", "cam2"):
            with open(os.path.join(self.tmp, c, "clip.mp4"), "wb"):
                pass
        times = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        entries = [
            {"timestamp_sec": t, "detections": [{"label": "person", "area": 0 if t < 1.0 else 100}]}
            for t in times
        ]
        for cam_name in ("cam1", "cam2"):
            sidecar_path = os.path.join(self.tmp, cam_name, DETECTION_SIDECAR_FILENAME)
            with open(sidecar_path, "w", encoding="utf-8") as f:
                json.dump({"native_width": 100, "native_height": 100, "entries": entries}, f)
        mock_video_reader_cls.return_value = self._make_nelux_reader_mock()

        result = extract_target_centric_frames(
            self.tmp,
            max_frames_sec=1.0,
            max_frames_min=10,
            camera_timeline_final_yolo_drop_no_person=True,
        )
        for ef in result:
            self.assertIsInstance(ef, ExtractedFrame)
            self.assertGreater(
                ef.metadata.get("person_area", 0),
                0,
                "With camera_timeline_final_yolo_drop_no_person=True, no frame should have person_area=0",
            )
