import os
import json
import shutil
import tempfile
import unittest
import sys
from unittest.mock import MagicMock, patch

# Mock external dependencies
for mod in ["cv2", "numpy", "requests", "flask", "paho.mqtt.client", "paho", "schedule", "yaml", "voluptuous"]:
    sys.modules[mod] = MagicMock()

import paho.mqtt.client
paho.mqtt.client.CallbackAPIVersion = MagicMock()
paho.mqtt.client.CallbackAPIVersion.VERSION2 = 2

from frigate_buffer.managers.file import (
    write_ai_frame_analysis_single_cam,
    write_ai_frame_analysis_multi_cam
)

def mock_imwrite(path, img):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"dummy")
    return True

class TestAiFrameAnalysisWriting(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        import cv2
        cv2.imwrite.side_effect = mock_imwrite

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_single_cam_writing(self):
        with patch("frigate_buffer.managers.file._CV2_AVAILABLE", True):
            event_dir = os.path.join(self.test_dir, "event1")
            os.makedirs(event_dir)

            frame = MagicMock()
            frames_with_time = [(frame, 1000.0), (frame, 1001.0)]

            write_ai_frame_analysis_single_cam(
                event_dir,
                frames_with_time,
                camera="front_door",
                write_manifest=True,
                create_zip_flag=True,
                save_frames=True
            )

            frames_dir = os.path.join(event_dir, "ai_frame_analysis", "frames")
            self.assertTrue(os.path.exists(os.path.join(frames_dir, "frame_001.jpg")))
            self.assertTrue(os.path.exists(os.path.join(frames_dir, "frame_002.jpg")))

            manifest_path = os.path.join(event_dir, "ai_frame_analysis", "manifest.json")
            self.assertTrue(os.path.exists(manifest_path))
            with open(manifest_path, "r") as f:
                manifest = json.load(f)

            self.assertEqual(len(manifest), 2)
            self.assertEqual(manifest[0]["filename"], "frame_001.jpg")
            self.assertEqual(manifest[0]["camera"], "front_door")
            self.assertEqual(manifest[0]["timestamp_sec"], 1000.0)

            zip_path = os.path.join(event_dir, "ai_analysis_debug.zip")
            self.assertTrue(os.path.exists(zip_path))

    def test_multi_cam_writing(self):
        with patch("frigate_buffer.managers.file._CV2_AVAILABLE", True):
            event_dir = os.path.join(self.test_dir, "event2")
            os.makedirs(event_dir)

            frame = MagicMock()

            class MockExtractedFrame:
                def __init__(self, frame, timestamp_sec, camera, metadata=None):
                    self.frame = frame
                    self.timestamp_sec = timestamp_sec
                    self.camera = camera
                    self.metadata = metadata or {}

            frames = [
                MockExtractedFrame(frame, 2000.0, "cam1", {"is_full_frame_resize": True}),
                MockExtractedFrame(frame, 2001.0, "cam 2", {}),
            ]

            write_ai_frame_analysis_multi_cam(
                event_dir,
                frames,
                write_manifest=True,
                create_zip_flag=True,
                save_frames=True
            )

            frames_dir = os.path.join(event_dir, "ai_frame_analysis", "frames")
            self.assertTrue(os.path.exists(os.path.join(frames_dir, "frame_001_cam1.jpg")))
            self.assertTrue(os.path.exists(os.path.join(frames_dir, "frame_002_cam_2.jpg")))

            manifest_path = os.path.join(event_dir, "ai_frame_analysis", "manifest.json")
            self.assertTrue(os.path.exists(manifest_path))
            with open(manifest_path, "r") as f:
                manifest = json.load(f)

            self.assertEqual(len(manifest), 2)
            self.assertEqual(manifest[0]["filename"], "frame_001_cam1.jpg")
            self.assertEqual(manifest[0]["camera"], "cam1")
            self.assertEqual(manifest[0]["is_full_frame_resize"], True)
            self.assertEqual(manifest[1]["filename"], "frame_002_cam_2.jpg")
            self.assertEqual(manifest[1]["camera"], "cam 2")
            self.assertNotIn("is_full_frame_resize", manifest[1])
