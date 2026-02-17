"""Tests for multi-clip target-centric frame extractor."""

import json
import unittest
import os
import tempfile
import shutil

from frigate_buffer.services.multi_clip_extractor import (
    extract_target_centric_frames,
    _person_area_from_detections,
    _person_area_at_time,
    _load_sidecar_for_camera,
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
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(_load_sidecar_for_camera(d))

    def test_load_sidecar_valid_returns_list(self):
        """Valid detection.json returns list of entries."""
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, DETECTION_SIDECAR_FILENAME)
            data = [{"timestamp_sec": 0.0, "detections": [{"label": "person", "area": 100}]}]
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f)
            result = _load_sidecar_for_camera(d)
            self.assertIsNotNone(result)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["timestamp_sec"], 0.0)


class TestMultiClipExtractor(unittest.TestCase):
    """Tests for extract_target_centric_frames."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp)

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
