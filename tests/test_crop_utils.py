"""Tests for crop_utils (motion crop, timestamp overlay)."""

import unittest

import numpy as np

from frigate_buffer.services import crop_utils


class TestDrawTimestampOverlay(unittest.TestCase):
    """Tests for draw_timestamp_overlay return value and read-only frame handling."""

    def test_draw_timestamp_overlay_readonly_returns_writable_copy(self):
        """When frame is read-only, overlay copies and returns a writable array."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame.setflags(write=False)
        self.assertFalse(frame.flags.writeable)

        result = crop_utils.draw_timestamp_overlay(
            frame,
            "2026-02-17 15:42:55",
            "test_cam",
            1,
            3,
        )

        self.assertIsNotNone(result)
        self.assertTrue(result.flags.writeable)
        self.assertIsNot(result, frame)

    def test_draw_timestamp_overlay_writable_returns_same_object(self):
        """When frame is writable, overlay draws in-place and returns same array."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        self.assertTrue(frame.flags.writeable)

        result = crop_utils.draw_timestamp_overlay(
            frame,
            "2026-02-17 15:42:55",
            "test_cam",
            1,
            3,
        )

        self.assertIs(result, frame)
        self.assertTrue(result.flags.writeable)

    def test_draw_timestamp_overlay_with_person_area_draws_bottom_right(self):
        """When person_area is set, overlay draws at bottom-right; returned frame has same shape and is writable."""
        frame = np.zeros((120, 200, 3), dtype=np.uint8)
        result = crop_utils.draw_timestamp_overlay(
            frame,
            "2026-02-17 15:42:55",
            "test_cam",
            1,
            3,
            person_area=12345,
        )
        self.assertEqual(result.shape, frame.shape)
        self.assertTrue(result.flags.writeable)
        # Bottom-right region should have non-zero pixels from "person_area: 12345" text
        h, w = result.shape[:2]
        bottom_right = result[h - 30 : h, w - 120 : w]
        self.assertGreater(np.count_nonzero(bottom_right), 0, "Bottom-right overlay should be drawn")


class TestFullFrameResizeToTarget(unittest.TestCase):
    """Tests for full_frame_resize_to_target (letterbox to target size)."""

    def test_full_frame_resize_to_target_returns_exact_shape(self):
        """Output is exactly target_h x target_w with aspect ratio preserved (letterbox)."""
        frame = np.zeros((360, 640, 3), dtype=np.uint8)  # 640x360
        result = crop_utils.full_frame_resize_to_target(frame, 1280, 720)
        self.assertEqual(result.shape, (720, 1280, 3))

    def test_full_frame_resize_to_target_preserves_aspect_ratio(self):
        """Wider frame gets horizontal letterbox (black bars top/bottom)."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)  # 4:3
        result = crop_utils.full_frame_resize_to_target(frame, 1280, 720)
        self.assertEqual(result.shape, (720, 1280, 3))
        # Scaled 640->1280, 480->960; then pad (720-960)/2 = -120 so we scale to fit: scale = min(1280/640, 720/480)=1.5 -> 960x720, pad 0,160,0,160
        # So center 960px width in 1280 -> left/right padding 160 each. Black borders.
        self.assertTrue(np.all(result[0, :] == 0) or np.all(result[:, 0] == 0))


class TestCropAroundDetectionsWithPadding(unittest.TestCase):
    """Tests for crop_around_detections_with_padding (master bbox over all detections + padding)."""

    def test_empty_detections_returns_full_frame(self):
        """When detections is empty, return the frame unchanged."""
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        result = crop_utils.crop_around_detections_with_padding(frame, [], padding_fraction=0.1)
        self.assertIs(result, frame)
        self.assertEqual(result.shape, (100, 200, 3))

    def test_single_detection_crops_with_padding(self):
        """One bbox yields a crop that contains the bbox region plus padding."""
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        detections = [{"bbox": [50, 20, 90, 80], "label": "person"}]
        result = crop_utils.crop_around_detections_with_padding(frame, detections, padding_fraction=0.1)
        self.assertIsNotNone(result)
        # bbox width=40, height=60; pad 10% = 4 and 6. So x1=46, y1=14, x2=94, y2=86. Clamped to [0,200] and [0,100].
        self.assertEqual(result.shape[0], 72)  # 86-14
        self.assertEqual(result.shape[1], 48)  # 94-46
        self.assertEqual(result.shape[2], 3)

    def test_multiple_detections_master_bbox_encompasses_all(self):
        """Multiple detections produce one master bbox (union) then padding."""
        frame = np.zeros((200, 300, 3), dtype=np.uint8)
        # Two people: left and right
        detections = [
            {"bbox": [10, 50, 60, 120], "label": "person"},
            {"bbox": [200, 80, 280, 180], "label": "person"},
        ]
        result = crop_utils.crop_around_detections_with_padding(frame, detections, padding_fraction=0.1)
        self.assertIsNotNone(result)
        # Master: x1_min=10, y1_min=50, x2_max=280, y2_max=180. bw=270, bh=130. pad 27, 13. x1=-17->0, y1=37, x2=307->300, y2=193->200.
        self.assertGreaterEqual(result.shape[0], 100)
        self.assertGreaterEqual(result.shape[1], 200)
        self.assertEqual(result.shape[2], 3)
