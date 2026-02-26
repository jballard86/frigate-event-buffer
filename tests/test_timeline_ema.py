"""Tests for Phase 1 timeline logic: dense grid, EMA, hysteresis, merge
(including first-segment roll-forward)."""

from frigate_buffer.services.timeline_ema import (
    build_dense_times,
    build_phase1_assignments,
)


class TestBuildDenseTimes:
    """build_dense_times produces a grid with step_sec interval and
    max_frames_min cap."""

    def test_returns_empty_for_invalid_inputs(self):
        assert build_dense_times(0, 60, 2, 10.0) == []
        assert build_dense_times(1, 60, 0, 10.0) == []
        assert build_dense_times(1, 60, 2, 0) == []

    def test_step_is_interval_cap_limits_count(self):
        # step_sec=1, cap=60, global_end=2 => times 0, 1, 2 (interval 1s, at most 60)
        times = build_dense_times(1.0, 60, 2.0, 2.0)
        assert len(times) == 3
        assert times[0] == 0.0
        assert times[1] == 1.0
        assert times[-1] == 2.0

    def test_respects_cap(self):
        # step_sec=0.1, cap=10, global_end=100 => at most 10 times (0, 0.1, ..., 0.9)
        times = build_dense_times(0.1, 10, 3.0, 100.0)
        assert len(times) <= 10
        assert times[0] == 0.0


class TestBuildPhase1Assignments:
    """build_phase1_assignments: EMA, hysteresis, merge short segments."""

    def test_empty_times_returns_empty(self):
        result = build_phase1_assignments(
            [], ["cam1", "cam2"], lambda c, t: 100.0, {}, ema_alpha=0.4
        )
        assert result == []

    def test_single_camera_returns_all_that_camera(self):
        times = [0.0, 0.5, 1.0]
        result = build_phase1_assignments(
            times, ["cam1"], lambda c, t: 50.0, {"cam1": (1920, 1080)}, ema_alpha=0.5
        )
        assert len(result) == 3
        assert all(c == "cam1" for _, c in result)
        assert [t for t, _ in result] == times

    def test_two_cameras_returns_assignments_per_time(self):
        times = [0.0, 0.5, 1.0]

        def area(cam: str, t: float) -> float:
            return 100.0 if cam == "cam1" else 50.0

        result = build_phase1_assignments(
            times,
            ["cam1", "cam2"],
            area,
            {"cam1": (1920, 1080), "cam2": (1920, 1080)},
            ema_alpha=0.5,
            hysteresis_margin=1.2,
        )
        assert len(result) == 3
        assert all((t, c) for t, c in result)
        cameras = [c for _, c in result]
        assert "cam1" in cameras

    def test_first_segment_short_merges_forward(self):
        # First run = 2 frames of cam2, then cam1 for 10. Should become cam1 for
        # first 2 as well (roll forward).
        times = [0.0 + i * 0.5 for i in range(14)]  # 14 times

        def area(cam: str, t: float) -> float:
            if t < 1.0 and cam == "cam2":
                return 600.0
            if t >= 1.0 and cam == "cam1":
                return 600.0
            return 100.0

        result = build_phase1_assignments(
            times,
            ["cam1", "cam2"],
            area,
            {"cam1": (1920, 1080), "cam2": (1920, 1080)},
            ema_alpha=0.6,
            hysteresis_margin=1.1,
            min_segment_frames=5,
        )
        cameras = [c for _, c in result]
        # First segment (cam2) has only 2–3 samples; should be merged into next (cam1)
        assert len(cameras) == 14
        # After merge, first 2 indices should be cam1 (rolled forward from next segment)
        assert cameras[0] == "cam1"
        assert cameras[1] == "cam1"

    def test_short_mid_segment_merges_into_previous(self):
        # cam1, then 2 frames cam2, then cam1. Middle cam2 run should become cam1.
        times = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

        def area(cam: str, t: float) -> float:
            if 1.0 <= t <= 1.5 and cam == "cam2":
                return 600.0
            if cam == "cam1":
                return 500.0
            return 50.0

        result = build_phase1_assignments(
            times,
            ["cam1", "cam2"],
            area,
            {"cam1": (1920, 1080), "cam2": (1920, 1080)},
            ema_alpha=0.5,
            hysteresis_margin=1.2,
            min_segment_frames=3,
        )
        cameras = [c for _, c in result]
        # Mid segment cam2 (2 frames) should merge into previous cam1
        assert cameras[2] == "cam1"
        assert cameras[3] == "cam1"
