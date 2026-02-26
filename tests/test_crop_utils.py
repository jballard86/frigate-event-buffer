"""Tests for crop_utils (motion crop, timestamp overlay)."""

import unittest

import numpy as np

from frigate_buffer.services import crop_utils

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def _bchw_tensor(h: int, w: int, c: int = 3, dtype: str = "uint8") -> "torch.Tensor":
    """Create a batch-of-one tensor (1, C, H, W) for crop_utils."""
    if not _TORCH_AVAILABLE:
        raise unittest.SkipTest("torch not available")
    d = torch.uint8 if dtype == "uint8" else torch.float32
    return torch.zeros(1, c, h, w, dtype=d)


class TestDrawTimestampOverlay(unittest.TestCase):
    """Tests for draw_timestamp_overlay return value and read-only frame handling."""

    def test_draw_timestamp_overlay_readonly_returns_writable_copy(self):
        """When frame is read-only numpy, overlay copies and returns a writable array."""
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
        """When frame is writable numpy, overlay draws in-place and returns same array."""
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
        h, w = result.shape[:2]
        bottom_right = result[h - 30 : h, w - 120 : w]
        self.assertGreater(
            np.count_nonzero(bottom_right), 0, "Bottom-right overlay should be drawn"
        )

    def test_draw_timestamp_overlay_tensor_returns_numpy_bgr(self):
        """When frame is tensor BCHW RGB, overlay returns numpy HWC BGR."""
        if not _TORCH_AVAILABLE:
            self.skipTest("torch not available")
        frame = _bchw_tensor(100, 100)
        result = crop_utils.draw_timestamp_overlay(
            frame,
            "2026-02-17 15:42:55",
            "test_cam",
            1,
            3,
        )
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (100, 100, 3))


class TestCropAroundCenterToSize(unittest.TestCase):
    """Tests for crop_around_center_to_size (variable crop, fixed output; used by video_compilation)."""

    def test_crop_around_center_to_size_returns_bchw_output_shape(self):
        if not _TORCH_AVAILABLE:
            self.skipTest("torch not available")
        frame = _bchw_tensor(480, 640)
        result = crop_utils.crop_around_center_to_size(
            frame, 320, 240, 320, 240, 1440, 1080
        )
        self.assertEqual(result.shape, (1, 3, 1080, 1440))

    def test_crop_around_center_to_size_rejects_numpy(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        with self.assertRaises(TypeError):
            crop_utils.crop_around_center_to_size(frame, 50, 50, 50, 50, 100, 100)


class TestCenterCrop(unittest.TestCase):
    """Tests for center_crop with tensor BCHW."""

    def test_center_crop_returns_bchw(self):
        if not _TORCH_AVAILABLE:
            self.skipTest("torch not available")
        frame = _bchw_tensor(360, 640)
        result = crop_utils.center_crop(frame, 320, 180)
        self.assertEqual(result.shape, (1, 3, 180, 320))

    def test_center_crop_rejects_numpy(self):
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        with self.assertRaises(TypeError):
            crop_utils.center_crop(frame, 50, 50)


class TestFullFrameResizeToTarget(unittest.TestCase):
    """Tests for full_frame_resize_to_target (letterbox to target size) with tensor BCHW."""

    def test_full_frame_resize_to_target_returns_exact_shape(self):
        """Output is exactly (1, 3, target_h, target_w) with aspect ratio preserved (letterbox)."""
        if not _TORCH_AVAILABLE:
            self.skipTest("torch not available")
        frame = _bchw_tensor(360, 640)  # 640x360
        result = crop_utils.full_frame_resize_to_target(frame, 1280, 720)
        self.assertEqual(result.shape, (1, 3, 720, 1280))

    def test_full_frame_resize_to_target_preserves_aspect_ratio(self):
        """Wider frame gets horizontal letterbox (black bars left/right); BCHW so check padding."""
        if not _TORCH_AVAILABLE:
            self.skipTest("torch not available")
        frame = _bchw_tensor(480, 640)  # 4:3
        result = crop_utils.full_frame_resize_to_target(frame, 1280, 720)
        self.assertEqual(result.shape, (1, 3, 720, 1280))
        # Top row (H=0) or left column (W=0) should have black padding
        top_black = (result[:, :, 0, :] == 0).all().item()
        left_black = (result[:, :, :, 0] == 0).all().item()
        self.assertTrue(top_black or left_black)


class TestCropAroundDetectionsWithPadding(unittest.TestCase):
    """Tests for crop_around_detections_with_padding with tensor BCHW."""

    def test_empty_detections_returns_full_frame(self):
        """When detections is empty, return the frame unchanged."""
        if not _TORCH_AVAILABLE:
            self.skipTest("torch not available")
        frame = _bchw_tensor(100, 200)
        result = crop_utils.crop_around_detections_with_padding(
            frame, [], padding_fraction=0.1
        )
        self.assertIs(result, frame)
        self.assertEqual(result.shape, (1, 3, 100, 200))

    def test_single_detection_crops_with_padding(self):
        """One bbox yields a crop that contains the bbox region plus padding. BCHW out."""
        if not _TORCH_AVAILABLE:
            self.skipTest("torch not available")
        frame = _bchw_tensor(100, 200)
        detections = [{"bbox": [50, 20, 90, 80], "label": "person"}]
        result = crop_utils.crop_around_detections_with_padding(
            frame, detections, padding_fraction=0.1
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.shape[0], 1)
        self.assertEqual(result.shape[1], 3)
        self.assertEqual(result.shape[2], 72)  # 86-14
        self.assertEqual(result.shape[3], 48)  # 94-46

    def test_multiple_detections_master_bbox_encompasses_all(self):
        """Multiple detections produce one master bbox (union) then padding. BCHW out."""
        if not _TORCH_AVAILABLE:
            self.skipTest("torch not available")
        frame = _bchw_tensor(200, 300)
        detections = [
            {"bbox": [10, 50, 60, 120], "label": "person"},
            {"bbox": [200, 80, 280, 180], "label": "person"},
        ]
        result = crop_utils.crop_around_detections_with_padding(
            frame, detections, padding_fraction=0.1
        )
        self.assertIsNotNone(result)
        self.assertGreaterEqual(result.shape[2], 100)
        self.assertGreaterEqual(result.shape[3], 200)
        self.assertEqual(result.shape[1], 3)
