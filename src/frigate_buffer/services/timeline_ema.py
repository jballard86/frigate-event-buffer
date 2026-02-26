"""
Phase 1 timeline logic for multi-cam: dense grid, EMA smoothing, segment
assignment with hysteresis, merge short segments.

Also provides timeline/slice building for compilation: convert_timeline_to_segments,
assignments_to_slices, _trim_slices_to_action_window. No decode or file I/O here;
caller provides area_at_t(camera, t_sec) and native sizes for normalization.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from frigate_buffer.constants import ACTION_POSTROLL_SEC, ACTION_PREROLL_SEC

logger = logging.getLogger("frigate-buffer")


def build_dense_times(
    step_sec: float,
    max_frames_min: int,
    analysis_multiplier: float,
    global_end: float,
) -> list[float]:
    """
    Build a sample-time grid for Phase 1 analysis.

    step_sec is the target interval between frames (max_frames_sec from config).
    We generate at most max_frames_min sample times so the frame count respects
    both the interval and the cap. analysis_multiplier is kept for API compatibility;
    if <= 0 we return [] for backward compatibility.
    """
    if step_sec <= 0 or global_end <= 0:
        return []
    if analysis_multiplier <= 0:
        return []
    step_analysis = step_sec
    cap = max(1, max_frames_min)
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
    """Return area as fraction of frame when normalize and native size valid;
    else raw area."""
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
    - EMA: smooth_cam(t) = ema_alpha * area_cam(t) + (1 - ema_alpha) *
      smooth_cam(t_prev).
    - Best view at t: argmax over cameras of (smoothed area * primary_bias if
      cam == primary_camera).
    - Hysteresis: switch from A to B only when B's smoothed value >= A's *
      hysteresis_margin.
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
    smooth: dict[str, float] = dict.fromkeys(cameras, 0.0)
    smoothed_series: dict[str, list[float]] = {c: [] for c in cameras}
    for t in times:
        for cam in cameras:
            a = area_cam_t(cam, t)
            smooth[cam] = ema_alpha * a + (1.0 - ema_alpha) * smooth[cam]
            smoothed_series[cam].append(smooth[cam])

    # Best view at each t with primary bias and hysteresis
    assignments: list[str] = []
    current: str | None = None
    for i, _t in enumerate(times):
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
    # Rule: short run -> merge into preceding; first run (no preceding) ->
    # merge into next
    min_len = max(1, min_segment_frames)
    n = len(assignments)
    if n == 0:
        return list(zip(times, assignments, strict=True))

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
        return list(zip(times, assignments, strict=True))

    # Merge short runs: first run short -> assign to next run's camera; other
    # short -> assign to previous run's camera
    merged = list(assignments)
    for run_idx, (_cam, start, length) in enumerate(runs):
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

    return list(zip(times, merged, strict=True))


def convert_timeline_to_segments(
    timeline_points: list[tuple[float, str]], global_end: float
) -> list[dict]:
    """
    Groups continuous time steps for the same camera into segments.
    global_end is required, but if for some reason it is 0 or None, it falls back
    to the last point + the step interval of the first two points.
    """
    segments = []
    if not timeline_points:
        return segments

    if not global_end:
        step = 1.0
        if len(timeline_points) > 1:
            step = timeline_points[1][0] - timeline_points[0][0]
        global_end = timeline_points[-1][0] + step

    current_cam = timeline_points[0][1]
    seg_start = timeline_points[0][0]

    for i in range(1, len(timeline_points)):
        t, cam = timeline_points[i]
        if cam != current_cam:
            segments.append(
                {"camera": current_cam, "start_sec": seg_start, "end_sec": t}
            )
            current_cam = cam
            seg_start = t

    segments.append(
        {"camera": current_cam, "start_sec": seg_start, "end_sec": global_end}
    )

    return segments


def assignments_to_slices(
    assignments: list[tuple[float, str]], global_end: float
) -> list[dict]:
    """
    Build one slice per assignment. Slice i covers [t_i, t_{i+1}] with camera cam_i.
    Last slice uses global_end as end_sec. Time O(len(assignments)), space O(n).
    """
    if not assignments:
        return []
    slices: list[dict] = []
    for i, (t_sec, camera) in enumerate(assignments):
        end_sec = assignments[i + 1][0] if i + 1 < len(assignments) else global_end
        slices.append(
            {
                "camera": camera,
                "start_sec": t_sec,
                "end_sec": end_sec,
            }
        )
    return slices


def _trim_slices_to_action_window(
    slices: list[dict],
    sidecars: dict[str, list],
    global_end: float,
) -> list[dict]:
    """
    Compute first/last detection time across cameras' sidecar entries (with detections),
    discard slices outside [action_start, action_end], clamp overlapping
    slices to that window. Returns new list; does not mutate input.
    """
    first_detection_sec: float = global_end
    last_detection_sec: float = 0.0
    for entries in sidecars.values():
        for e in entries or []:
            if len(e.get("detections") or []) > 0:
                ts = float(e.get("timestamp_sec") or 0)
                first_detection_sec = min(first_detection_sec, ts)
                last_detection_sec = max(last_detection_sec, ts)
    if first_detection_sec > last_detection_sec:
        action_start = 0.0
        action_end = global_end
    else:
        action_start = max(0.0, first_detection_sec - ACTION_PREROLL_SEC)
        action_end = min(global_end, last_detection_sec + ACTION_POSTROLL_SEC)

    result: list[dict] = []
    for sl in slices:
        if sl["end_sec"] <= action_start or sl["start_sec"] >= action_end:
            continue
        s_start = max(sl["start_sec"], action_start)
        s_end = min(sl["end_sec"], action_end)
        if s_start >= s_end:
            continue
        new_slice = dict(sl)
        new_slice["start_sec"] = s_start
        new_slice["end_sec"] = s_end
        result.append(new_slice)
    return result
