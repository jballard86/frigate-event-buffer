"""Timeline, trim, crop, and segment math for video_compilation."""

import unittest

from frigate_buffer.services.video_compilation import (
    _enforce_slice_boundary_continuity,
    _trim_slices_to_action_window,
    assignments_to_slices,
    calculate_crop_at_time,
    calculate_segment_crop,
    convert_timeline_to_segments,
    smooth_crop_centers_ema,
    smooth_zoom_ema,
)


class TestConvertTimelineToSegments(unittest.TestCase):
    """Tests for convert_timeline_to_segments function."""

    def test_empty_timeline_returns_empty_list(self):
        """Empty timeline should return empty segments list."""
        result = convert_timeline_to_segments([], 60.0)
        assert result == []

    def test_single_camera_continuous_timeline(self):
        """Single camera throughout creates one segment."""
        timeline = [
            (0.0, "doorbell"),
            (1.0, "doorbell"),
            (2.0, "doorbell"),
        ]
        result = convert_timeline_to_segments(timeline, 10.0)
        assert len(result) == 1
        assert result[0]["camera"] == "doorbell"
        assert result[0]["start_sec"] == 0.0
        assert result[0]["end_sec"] == 10.0

    def test_camera_switch_creates_multiple_segments(self):
        """Camera switches create multiple segments."""
        timeline = [
            (0.0, "doorbell"),
            (5.0, "doorbell"),
            (10.0, "carport"),
            (15.0, "carport"),
        ]
        result = convert_timeline_to_segments(timeline, 20.0)
        assert len(result) == 2
        assert result[0]["camera"] == "doorbell"
        assert result[0]["start_sec"] == 0.0
        assert result[0]["end_sec"] == 10.0
        assert result[1]["camera"] == "carport"
        assert result[1]["start_sec"] == 10.0
        assert result[1]["end_sec"] == 20.0

    def test_multiple_switches_create_correct_segments(self):
        """Multiple camera switches create correct segment boundaries."""
        timeline = [
            (0.0, "doorbell"),
            (5.0, "carport"),
            (10.0, "doorbell"),
            (15.0, "carport"),
        ]
        result = convert_timeline_to_segments(timeline, 20.0)
        assert len(result) == 4
        assert result[0]["end_sec"] == 5.0
        assert result[1]["start_sec"] == 5.0
        assert result[1]["end_sec"] == 10.0

    def test_global_end_fallback_when_zero(self):
        """When global_end is 0, fallback to last point + step."""
        timeline = [
            (0.0, "doorbell"),
            (1.0, "doorbell"),
            (2.0, "doorbell"),
        ]
        result = convert_timeline_to_segments(timeline, 0.0)
        assert len(result) == 1
        assert result[0]["end_sec"] == 3.0  # 2.0 + 1.0 step


class TestAssignmentsToSlices(unittest.TestCase):
    """Tests for assignments_to_slices function."""

    def test_empty_assignments_returns_empty_list(self):
        result = assignments_to_slices([], 60.0)
        assert result == []

    def test_one_assignment_creates_one_slice_with_global_end(self):
        assignments = [(0.0, "cam1")]
        result = assignments_to_slices(assignments, 10.0)
        assert len(result) == 1
        assert result[0]["camera"] == "cam1"
        assert result[0]["start_sec"] == 0.0
        assert result[0]["end_sec"] == 10.0

    def test_multiple_assignments_creates_slices_with_correct_boundaries(self):
        assignments = [(0.0, "cam1"), (2.0, "cam1"), (4.0, "cam2")]
        result = assignments_to_slices(assignments, 6.0)
        assert len(result) == 3
        assert result[0]["start_sec"] == 0.0
        assert result[0]["end_sec"] == 2.0
        assert result[1]["start_sec"] == 2.0
        assert result[1]["end_sec"] == 4.0
        assert result[2]["start_sec"] == 4.0
        assert result[2]["end_sec"] == 6.0


class TestTrimSlicesToActionWindow(unittest.TestCase):
    """Tests for _trim_slices_to_action_window (trim to first/last detection ± roll)."""

    def test_trims_to_action_window_and_clamps_overlapping_slices(self):
        """First 2s, last 10s, end=20 → [0,13]; outside dropped, overlapping clamped."""
        sidecars = {
            "cam1": [
                {
                    "timestamp_sec": 2.0,
                    "detections": [{"centerpoint": [100, 100], "area": 1.0}],
                },
                {
                    "timestamp_sec": 10.0,
                    "detections": [{"centerpoint": [200, 200], "area": 1.0}],
                },
            ],
        }
        slices = [
            {"camera": "cam1", "start_sec": 0.0, "end_sec": 5.0},
            {"camera": "cam1", "start_sec": 5.0, "end_sec": 15.0},
            {"camera": "cam1", "start_sec": 15.0, "end_sec": 20.0},
        ]
        result = _trim_slices_to_action_window(slices, sidecars, global_end=20.0)
        # action_start = max(0, 2-3)=0, action_end = min(20, 10+3)=13
        assert len(result) == 2
        assert result[0]["camera"] == "cam1"
        assert result[0]["start_sec"] == 0.0
        assert result[0]["end_sec"] == 5.0
        assert result[1]["start_sec"] == 5.0
        assert result[1]["end_sec"] == 13.0
        # Slice [15, 20] is entirely outside [0, 13] so dropped

    def test_no_detections_keeps_full_range(self):
        """When no sidecar has detections, no trimming (full 0..global_end)."""
        sidecars = {
            "cam1": [
                {"timestamp_sec": 1.0, "detections": []},
                {"timestamp_sec": 5.0, "detections": []},
            ]
        }
        slices = [{"camera": "cam1", "start_sec": 0.0, "end_sec": 10.0}]
        result = _trim_slices_to_action_window(slices, sidecars, global_end=10.0)
        assert len(result) == 1
        assert result[0]["start_sec"] == 0.0
        assert result[0]["end_sec"] == 10.0

    def test_discards_slice_entirely_after_action_end(self):
        """Slice entirely after action_end is discarded."""
        sidecars = {"cam1": [{"timestamp_sec": 1.0, "detections": [{"area": 1.0}]}]}
        slices = [{"camera": "cam1", "start_sec": 20.0, "end_sec": 30.0}]
        result = _trim_slices_to_action_window(slices, sidecars, global_end=30.0)
        assert len(result) == 0

    def test_discards_slice_entirely_before_action_start(self):
        """Slice entirely before action_start discarded when first detection is late."""
        sidecars = {"cam1": [{"timestamp_sec": 10.0, "detections": [{"area": 1.0}]}]}
        slices = [
            {"camera": "cam1", "start_sec": 0.0, "end_sec": 5.0},
            {"camera": "cam1", "start_sec": 7.0, "end_sec": 15.0},
        ]
        result = _trim_slices_to_action_window(slices, sidecars, global_end=20.0)
        # action_start = 10-3 = 7, action_end = 13.
        # First slice [0,5] entirely before 7 → dropped.
        assert len(result) == 1
        assert result[0]["start_sec"] == 7.0
        assert result[0]["end_sec"] == 13.0


class TestCalculateCropAtTime(unittest.TestCase):
    """Tests for calculate_crop_at_time function."""

    def test_no_entries_uses_center_crop(self):
        sidecar = {"entries": []}
        x, y, w, h = calculate_crop_at_time(sidecar, 5.0, 1920, 1080, 1440, 1080)
        assert w == 1440
        assert h == 1080
        assert x == 240
        assert y == 0

    def test_single_entry_detection_uses_weighted_center(self):
        sidecar = {
            "entries": [
                {
                    "timestamp_sec": 5.0,
                    "detections": [{"centerpoint": [1000, 500], "area": 1000.0}],
                }
            ]
        }
        x, y, w, h = calculate_crop_at_time(sidecar, 5.0, 1920, 1080, 1440, 1080)
        expected_x = max(0, min(1920 - 1440, int(1000 - 1440 / 2.0)))
        expected_y = max(0, min(1080 - 1080, int(500 - 1080 / 2.0)))
        assert x == expected_x
        assert y == expected_y

    def test_with_timestamps_sorted_uses_bisect(self):
        sidecar = {
            "entries": [
                {
                    "timestamp_sec": 1.0,
                    "detections": [{"centerpoint": [100, 100], "area": 100.0}],
                },
                {
                    "timestamp_sec": 3.0,
                    "detections": [{"centerpoint": [500, 500], "area": 100.0}],
                },
            ]
        }
        ts_sorted = [1.0, 3.0]
        x, y, w, h = calculate_crop_at_time(
            sidecar, 2.0, 1920, 1080, 1440, 1080, timestamps_sorted=ts_sorted
        )
        assert w == 1440
        assert h == 1080

    def test_hold_last_known_crop_when_nearest_has_no_detections(self):
        """When person leaves at t=4, query at t=6: nearest (t=6) has no detections;
        entry at t=4 has detections within 5s → crop holds at t=4 position."""
        sidecar = {
            "entries": [
                {
                    "timestamp_sec": 4.0,
                    "detections": [{"centerpoint": [800, 400], "area": 500.0}],
                },
                {"timestamp_sec": 5.0, "detections": []},
                {"timestamp_sec": 6.0, "detections": []},
            ]
        }
        x, y, w, h = calculate_crop_at_time(sidecar, 6.0, 1920, 1080, 1440, 1080)
        # Should use t=4 entry: center (800, 400), not frame center (960, 540)
        expected_x = max(0, min(1920 - 1440, int(800 - 1440 / 2.0)))
        expected_y = max(0, min(1080 - 1080, int(400 - 1080 / 2.0)))
        assert x == expected_x
        assert y == expected_y
        assert w == 1440
        assert h == 1080

    def test_hold_crop_fallback_to_center_when_no_detections_within_threshold(self):
        """When no entry with detections exists within 5s, crop remains center."""
        sidecar = {
            "entries": [
                {
                    "timestamp_sec": 0.0,
                    "detections": [{"centerpoint": [200, 200], "area": 100.0}],
                },
                {"timestamp_sec": 10.0, "detections": []},
            ]
        }
        x, y, w, h = calculate_crop_at_time(sidecar, 10.0, 1920, 1080, 1440, 1080)
        # t=0 is > 5s away from t=10, so fallback to center
        assert x == 240
        assert y == 0
        assert w == 1440
        assert h == 1080

    def test_nearest_entry_has_detections_unchanged(self):
        """Nearest entry has detections → behavior unchanged (no regression)."""
        sidecar = {
            "entries": [
                {
                    "timestamp_sec": 4.0,
                    "detections": [{"centerpoint": [800, 400], "area": 500.0}],
                },
                {
                    "timestamp_sec": 6.0,
                    "detections": [{"centerpoint": [1000, 500], "area": 500.0}],
                },
            ]
        }
        x, y, w, h = calculate_crop_at_time(sidecar, 5.5, 1920, 1080, 1440, 1080)
        # Nearest is t=6 (distance 0.5) with detections at (1000, 500)
        expected_x = max(0, min(1920 - 1440, int(1000 - 1440 / 2.0)))
        expected_y = max(0, min(1080 - 1080, int(500 - 1080 / 2.0)))
        assert x == expected_x
        assert y == expected_y

    def test_zoom_uses_content_area_when_tracking_target_frame_percent_set(self):
        """tracking_target_frame_percent>0 + detections → crop from content, target%."""
        sidecar = {
            "entries": [
                {
                    "timestamp_sec": 1.0,
                    "detections": [{"bbox": [100, 100, 200, 200], "area": 10000.0}],
                }
            ]
        }
        x, y, w, h = calculate_crop_at_time(
            sidecar,
            1.0,
            1920,
            1080,
            1440,
            1080,
            tracking_target_frame_percent=40,
        )
        # Content = bbox + 10% padding; crop area = content/0.4; aspect 1440/1080.
        # Crop (w,h) clamped to [0.4*1920,1920] and [0.4*1080,1080].
        assert w > 0
        assert w <= 1920
        assert h > 0
        assert h <= 1080
        assert w % 2 == 0
        assert h % 2 == 0
        assert w >= int(1920 * 0.4)
        assert h >= int(1080 * 0.4)
        assert x >= 0
        assert x + w <= 1920
        assert y >= 0
        assert y + h <= 1080

    def test_no_zoom_when_tracking_target_frame_percent_zero(self):
        """When tracking_target_frame_percent=0, crop is fixed target_w x target_h."""
        sidecar = {
            "entries": [
                {
                    "timestamp_sec": 1.0,
                    "detections": [{"bbox": [100, 100, 200, 200], "area": 10000.0}],
                }
            ]
        }
        x, y, w, h = calculate_crop_at_time(
            sidecar,
            1.0,
            1920,
            1080,
            1440,
            1080,
            tracking_target_frame_percent=0,
        )
        assert w == 1440
        assert h == 1080


class TestSmoothCropCentersEma(unittest.TestCase):
    """Tests for smooth_crop_centers_ema function."""

    def test_alpha_zero_or_one_does_not_change_slices(self):
        slices = [
            {"crop_start": (100, 50, 1440, 1080), "crop_end": (200, 60, 1440, 1080)},
        ]
        orig = [dict(s) for s in slices]
        smooth_crop_centers_ema(slices, 0.0)
        assert slices[0]["crop_start"] == orig[0]["crop_start"]
        smooth_crop_centers_ema(slices, 1.0)
        assert slices[0]["crop_start"] == orig[0]["crop_start"]

    def test_ema_smooths_center_trajectory(self):
        slices = [
            {"crop_start": (0, 0, 100, 100), "crop_end": (10, 10, 100, 100)},
            {"crop_start": (100, 100, 100, 100), "crop_end": (110, 110, 100, 100)},
        ]
        smooth_crop_centers_ema(slices, 0.5)
        assert "crop_start" in slices[0]
        assert "crop_end" in slices[0]
        xs0, ys0, _, _ = slices[0]["crop_start"]
        xs1, ys1, _, _ = slices[1]["crop_start"]
        assert xs1 < 100


class TestSmoothZoomEma(unittest.TestCase):
    """Tests for smooth_zoom_ema function."""

    def test_alpha_zero_or_one_does_not_change_zoom(self):
        """Alpha 0 or 1 leaves zoom dimensions unchanged (or no-op)."""
        slices = [
            {
                "crop_start": (0, 0, 800, 450),
                "crop_end": (0, 0, 800, 450),
                "native_width": 1920,
                "native_height": 1080,
            },
        ]
        orig_w, orig_h = slices[0]["crop_start"][2], slices[0]["crop_start"][3]
        smooth_zoom_ema(slices, 0.0, 1440, 1080)
        assert slices[0]["crop_start"][2] == orig_w
        assert slices[0]["crop_start"][3] == orig_h
        smooth_zoom_ema(slices, 1.0, 1440, 1080)
        assert slices[0]["crop_start"][2] == orig_w
        assert slices[0]["crop_start"][3] == orig_h

    def test_smooth_zoom_ema_updates_crop_size_in_place(self):
        """smooth_zoom_ema updates (w,h) per slice; dimensions even and in bounds."""
        slices = [
            {
                "crop_start": (100, 50, 960, 540),
                "crop_end": (120, 60, 960, 540),
                "native_width": 1920,
                "native_height": 1080,
            },
            {
                "crop_start": (200, 100, 800, 450),
                "crop_end": (220, 110, 800, 450),
                "native_width": 1920,
                "native_height": 1080,
            },
        ]
        smooth_zoom_ema(slices, 0.3, 1440, 1080)
        for sl in slices:
            xs, ys, w, h = sl["crop_start"]
            assert w > 0
            assert w <= 1920
            assert h > 0
            assert h <= 1080
            assert w % 2 == 0
            assert h % 2 == 0
            assert xs >= 0
            assert xs + w <= 1920
            assert ys >= 0
            assert ys + h <= 1080

    def test_smooth_zoom_ema_resets_on_camera_change_no_blend_across_cameras(self):
        """When camera changes, second slice uses its own raw zoom (no blend)."""
        # Slice 0: cam A zoomed in. Slice 1: cam B full frame. After smooth_zoom_ema,
        # slice 1's crop_start (w,h) should match its raw full-frame zoom (target
        # aspect), not blended with slice 0's smaller zoom.
        slices = [
            {
                "camera": "doorbell",
                "crop_start": (400, 200, 800, 450),
                "crop_end": (420, 220, 800, 450),
                "native_width": 1920,
                "native_height": 1080,
            },
            {
                "camera": "carport",
                "crop_start": (0, 0, 1920, 1080),
                "crop_end": (0, 0, 1920, 1080),
                "native_width": 1920,
                "native_height": 1080,
            },
        ]
        smooth_zoom_ema(slices, 0.3, 1440, 1080)
        # Full-frame zoom at aspect 1440/1080: w=1662, h clamped to source 1080.
        _, _, w1, h1 = slices[1]["crop_start"]
        assert w1 == 1662
        assert h1 == 1080


class TestEnforceSliceBoundaryContinuity(unittest.TestCase):
    """Tests for _enforce_slice_boundary_continuity."""

    def test_same_camera_consecutive_slices_crop_start_equals_prev_crop_end(self):
        """Same-camera consecutive: second's crop_start = first's crop_end."""
        crop_end_0 = (100, 50, 960, 540)
        slices = [
            {
                "camera": "doorbell",
                "crop_start": (0, 0, 1000, 600),
                "crop_end": crop_end_0,
            },
            {
                "camera": "doorbell",
                "crop_start": (200, 100, 800, 450),
                "crop_end": (220, 110, 800, 450),
            },
        ]
        _enforce_slice_boundary_continuity(slices)
        assert slices[1]["crop_start"] == slices[0]["crop_end"]
        assert slices[1]["crop_start"] == crop_end_0

    def test_different_camera_slices_unchanged(self):
        """When camera changes, crop_start is not overwritten."""
        slices = [
            {
                "camera": "doorbell",
                "crop_start": (0, 0, 960, 540),
                "crop_end": (10, 10, 960, 540),
            },
            {
                "camera": "carport",
                "crop_start": (100, 100, 800, 450),
                "crop_end": (110, 110, 800, 450),
            },
        ]
        orig_start_1 = slices[1]["crop_start"]
        _enforce_slice_boundary_continuity(slices)
        assert slices[1]["crop_start"] == orig_start_1

    def test_single_slice_no_op(self):
        """Single slice leaves list unchanged."""
        slices = [
            {
                "camera": "doorbell",
                "crop_start": (0, 0, 1920, 1080),
                "crop_end": (0, 0, 1920, 1080),
            }
        ]
        _enforce_slice_boundary_continuity(slices)
        assert slices[0]["crop_start"] == (0, 0, 1920, 1080)


class TestCalculateSegmentCrop(unittest.TestCase):
    """Tests for calculate_segment_crop function."""

    def test_no_detections_uses_center_crop(self):
        """When no detections, crop is centered on source frame."""
        segment = {"start_sec": 0.0, "end_sec": 10.0}
        sidecar = {"entries": []}
        x, y, w, h = calculate_segment_crop(segment, sidecar, 1920, 1080, 1440, 1080)
        assert w == 1440
        assert h == 1080
        # Should be centered
        assert x == 240  # (1920 - 1440) / 2
        assert y == 0  # (1080 - 1080) / 2

    def test_detections_use_weighted_center(self):
        """Detections are used for area-weighted center calculation."""
        segment = {"start_sec": 0.0, "end_sec": 10.0}
        sidecar = {
            "entries": [
                {
                    "timestamp_sec": 5.0,
                    "detections": [{"centerpoint": [1000, 500], "area": 1000.0}],
                }
            ]
        }
        x, y, w, h = calculate_segment_crop(segment, sidecar, 1920, 1080, 1440, 1080)
        assert w == 1440
        assert h == 1080
        # Center should be near the detection centerpoint
        expected_x = int(1000 - 1440 / 2.0)
        expected_y = int(500 - 1080 / 2.0)
        assert x == max(0, min(1920 - 1440, expected_x))
        assert y == max(0, min(1080 - 1080, expected_y))

    def test_bbox_fallback_when_no_centerpoint(self):
        """Uses bbox when centerpoint is not available."""
        segment = {"start_sec": 0.0, "end_sec": 10.0}
        sidecar = {
            "entries": [
                {
                    "timestamp_sec": 5.0,
                    "detections": [{"box": [800, 400, 1200, 600], "area": 40000.0}],
                }
            ]
        }
        x, y, w, h = calculate_segment_crop(segment, sidecar, 1920, 1080, 1440, 1080)
        # Center from bbox: (800 + 1200) / 2 = 1000, (400 + 600) / 2 = 500
        expected_x = int(1000 - 1440 / 2.0)
        expected_y = int(500 - 1080 / 2.0)
        assert x == max(0, min(1920 - 1440, expected_x))
        assert y == max(0, min(1080 - 1080, expected_y))

    def test_crop_clamped_to_source_bounds(self):
        """Crop box is clamped to not exceed source video boundaries."""
        segment = {"start_sec": 0.0, "end_sec": 10.0}
        sidecar = {
            "entries": [
                {
                    "timestamp_sec": 5.0,
                    "detections": [
                        {"centerpoint": [100, 100], "area": 1000.0}  # Near edge
                    ],
                }
            ]
        }
        x, y, w, h = calculate_segment_crop(segment, sidecar, 1920, 1080, 1440, 1080)
        assert x >= 0
        assert x + w <= 1920
        assert y >= 0
        assert y + h <= 1080

    def test_source_smaller_than_target_scales_down(self):
        """When source is smaller than target, target is scaled down."""
        segment = {"start_sec": 0.0, "end_sec": 10.0}
        sidecar = {"entries": []}
        x, y, w, h = calculate_segment_crop(segment, sidecar, 640, 480, 1440, 1080)
        # Target should be scaled to fit within source
        assert w <= 640
        assert h <= 480
