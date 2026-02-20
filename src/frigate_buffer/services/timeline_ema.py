"""
Phase 1 timeline logic for multi-cam: dense grid, EMA smoothing, segment assignment with hysteresis, merge short segments.

Used by multi_clip_extractor for the Trust-the-EMA pipeline. No decode or file I/O here;
caller provides area_at_t(camera, t_sec) and native sizes for normalization.
"""

from __future__ import annotations

import logging
from typing import Callable

logger = logging.getLogger("frigate-buffer")


def build_dense_times(
    step_sec: float,
    max_frames_min: int,
    analysis_multiplier: float,
    global_end: float,
) -> list[float]:
    """
    Build a denser sample-time grid for Phase 1 analysis.

    step_sec is the base interval; we use step_sec / analysis_multiplier.
    Cap length at max_frames_min * analysis_multiplier (with a sensible upper bound).
    """
    if step_sec <= 0 or analysis_multiplier <= 0 or global_end <= 0:
        return []
    step_analysis = step_sec / analysis_multiplier
    if step_analysis <= 0:
        step_analysis = step_sec
    cap = min(int(max_frames_min * analysis_multiplier), max(1000, max_frames_min * 3))
    times: list[float] = []
    t = 0.0
    while t <= global_end and len(times) < cap:
        times.append(t)
        t += step_analysis
    return times


def _normalize_area(
    area: float,
    native_size: tuple[int, int],
    normalize: bool,
) -> float:
    """Return area as fraction of frame when normalize and native size valid; else raw area."""
    if not normalize or not native_size or native_size[0] <= 0 or native_size[1] <= 0:
        return area
    frame_px = native_size[0] * native_size[1]
    return area / frame_px if frame_px > 0 else area


def build_phase1_assignments(
    times: list[float],
    cameras: list[str],
    area_at_t: Callable[[str, float], float],
    native_size_per_cam: dict[str, tuple[int, int]],
    *,
    ema_alpha: float = 0.4,
    primary_bias_multiplier: float = 1.2,
    primary_camera: str | None = None,
    hysteresis_margin: float = 1.15,
    min_segment_frames: int = 5,
) -> list[tuple[float, str]]:
    """
    Compute per–sample-time camera assignment from area curves with EMA and hysteresis.

    - For each camera and t, area is taken from area_at_t(cam, t). When cameras have
      different native resolutions, area is normalized by (native_width * native_height)
      so cameras compete on fraction of frame, not raw pixels.
    - EMA: smooth_cam(t) = ema_alpha * area_cam(t) + (1 - ema_alpha) * smooth_cam(t_prev).
    - Best view at t: argmax over cameras of (smoothed area * primary_bias if cam == primary_camera).
    - Hysteresis: switch from A to B only when B's smoothed value >= A's * hysteresis_margin.
    - Short segments (run length < min_segment_frames) are merged: normally into the
      preceding segment; the first segment (index 0) has no predecessor so it is merged
      forward into the next segment.

    Returns list of (t_sec, camera) in order of times; length equals len(times).
    """
    if not times or not cameras:
        return []

    # Resolutions differ => mandatory normalization
    sizes = [native_size_per_cam.get(c) for c in cameras]
    has_valid = any(s and s[0] > 0 and s[1] > 0 for s in sizes)
    unique_res = len({(s or (0, 0)) for s in sizes}) > 1
    normalize = has_valid and unique_res

    # Raw (and possibly normalized) area per (cam, t)
    def area_cam_t(cam: str, t: float) -> float:
        raw = area_at_t(cam, t)
        return _normalize_area(raw, native_size_per_cam.get(cam) or (0, 0), normalize)

    # EMA per camera along time
    smooth: dict[str, float] = {c: 0.0 for c in cameras}
    smoothed_series: dict[str, list[float]] = {c: [] for c in cameras}
    for t in times:
        for cam in cameras:
            a = area_cam_t(cam, t)
            smooth[cam] = ema_alpha * a + (1.0 - ema_alpha) * smooth[cam]
            smoothed_series[cam].append(smooth[cam])

    # Best view at each t with primary bias and hysteresis
    assignments: list[str] = []
    current: str | None = None
    for i, t in enumerate(times):
        scores = {}
        for cam in cameras:
            s = smoothed_series[cam][i]
            if primary_camera and cam == primary_camera:
                s *= primary_bias_multiplier
            scores[cam] = s
        best = max(cameras, key=lambda c: scores[c])
        if current is None:
            current = best
        else:
            # Switch only if new camera is strictly better by hysteresis margin
            if best != current and scores[best] >= scores[current] * hysteresis_margin:
                current = best
        assignments.append(current)

    # Merge short segments: run-length encode, then merge short runs
    # Rule: short run -> merge into preceding; first run (no preceding) -> merge into next
    min_len = max(1, min_segment_frames)
    n = len(assignments)
    if n == 0:
        return list(zip(times, assignments))

    # Run-length: list of (camera, start_index, length)
    runs: list[tuple[str, int, int]] = []
    i = 0
    while i < n:
        cam = assignments[i]
        start = i
        while i < n and assignments[i] == cam:
            i += 1
        runs.append((cam, start, i - start))
    if not runs:
        return list(zip(times, assignments))

    # Merge short runs: first run short -> assign to next run's camera; other short -> assign to previous run's camera
    merged = list(assignments)
    for run_idx, (cam, start, length) in enumerate(runs):
        if length >= min_len:
            continue
        if run_idx == 0:
            # First segment: roll forward into next
            if len(runs) > 1:
                next_cam = runs[1][0]
                for j in range(start, start + length):
                    merged[j] = next_cam
        else:
            # Merge into preceding segment
            prev_cam = runs[run_idx - 1][0]
            for j in range(start, start + length):
                merged[j] = prev_cam

    return list(zip(times, merged))
