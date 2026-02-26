"""
Crop, zoom, and EMA math for video compilation.

Pure logic only: detection bbox/center, crop size, smooth zoom and crop-center EMA.
No I/O, no video_compilation import. Used by video_compilation service.
"""

from __future__ import annotations

import bisect
import math
from typing import Any

from frigate_buffer.constants import (
    HOLD_CROP_MAX_DISTANCE_SEC,
    ZOOM_CONTENT_PADDING,
    ZOOM_MIN_FRAME_FRACTION,
)


def _weighted_center_from_detections(
    detections: list[dict],
    source_width: int,
    source_height: int,
) -> tuple[float, float]:
    """
    Area-weighted center (cx, cy) from detection list; box/bbox/centerpoint supported.
    If no valid detections, returns frame center (source_width/2, source_height/2).
    """
    if not detections or source_width <= 0 or source_height <= 0:
        return (source_width / 2.0, source_height / 2.0)
    weighted_cx_sum: float = 0.0
    weighted_cy_sum: float = 0.0
    total_area: float = 0.0
    for det in detections:
        area = float(det.get("area", 1.0))
        if area <= 0:
            area = 1.0
        box = det.get("box") or det.get("bbox")
        if isinstance(box, (list, tuple)) and len(box) >= 4:
            x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            weighted_cx_sum += cx * area
            weighted_cy_sum += cy * area
            total_area += area
        else:
            cp = det.get("centerpoint")
            if isinstance(cp, (list, tuple)) and len(cp) >= 2:
                cx, cy = float(cp[0]), float(cp[1])
                weighted_cx_sum += cx * area
                weighted_cy_sum += cy * area
                total_area += area
    if total_area <= 0:
        return (source_width / 2.0, source_height / 2.0)
    return (weighted_cx_sum / total_area, weighted_cy_sum / total_area)


def _nearest_entry_at_t(
    entries: list[dict], t_sec: float, timestamps_sorted: list[float] | None = None
) -> dict | None:
    """
    Return the sidecar entry with timestamp_sec nearest to t_sec.
    If timestamps_sorted is provided (sorted list of timestamp_sec), use binary search.
    Time O(log n) with bisect, O(n) without; space O(1).
    """
    if not entries:
        return None
    if timestamps_sorted is not None and len(timestamps_sorted) == len(entries):
        idx = bisect.bisect_left(timestamps_sorted, t_sec)
        if idx >= len(entries):
            return entries[-1]
        if idx == 0:
            return entries[0]
        if abs(timestamps_sorted[idx] - t_sec) < abs(
            timestamps_sorted[idx - 1] - t_sec
        ):
            return entries[idx]
        return entries[idx - 1]
    return min(entries, key=lambda e: abs((e.get("timestamp_sec") or 0) - t_sec))


def _nearest_entry_with_detections_at_t(
    entries: list[dict],
    t_sec: float,
    max_distance_sec: float,
    timestamps_sorted: list[float] | None = None,  # noqa: ARG001
) -> dict | None:
    """
    Return the sidecar entry with detections whose timestamp_sec is nearest to t_sec
    and within max_distance_sec. Used to hold the last-known crop when a person leaves
    the frame instead of panning to center.
    """
    candidates = [
        e
        for e in entries
        if len(e.get("detections") or []) > 0
        and abs((e.get("timestamp_sec") or 0) - t_sec) <= max_distance_sec
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda e: abs((e.get("timestamp_sec") or 0) - t_sec),
    )


def _content_area_and_center(
    detections: list[dict],
    source_width: int,
    source_height: int,
    padding: float = ZOOM_CONTENT_PADDING,
) -> tuple[float, float, float]:
    """
    Compute union of detection bboxes expanded by padding, and area-weighted center.
    Returns (content_area, center_x, center_y). If no valid detections, returns
    (0, frame_center_x, frame_center_y).
    """
    if not detections or source_width <= 0 or source_height <= 0:
        cx, cy = _weighted_center_from_detections(detections, source_width, source_height)
        return (0.0, cx, cy)
    x1_min = float(source_width)
    y1_min = float(source_height)
    x2_max = 0.0
    y2_max = 0.0
    for det in detections:
        box = det.get("box") or det.get("bbox")
        if isinstance(box, (list, tuple)) and len(box) >= 4:
            x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
            x1_min = min(x1_min, x1)
            y1_min = min(y1_min, y1)
            x2_max = max(x2_max, x2)
            y2_max = max(y2_max, y2)
    if x1_min >= x2_max or y1_min >= y2_max:
        cx, cy = _weighted_center_from_detections(detections, source_width, source_height)
        return (0.0, cx, cy)
    bw = x2_max - x1_min
    bh = y2_max - y1_min
    pad_w = max(bw * padding, 1.0)
    pad_h = max(bh * padding, 1.0)
    content_x1 = max(0.0, x1_min - pad_w)
    content_y1 = max(0.0, y1_min - pad_h)
    content_x2 = min(float(source_width), x2_max + pad_w)
    content_y2 = min(float(source_height), y2_max + pad_h)
    content_area = (content_x2 - content_x1) * (content_y2 - content_y1)
    cx, cy = _weighted_center_from_detections(detections, source_width, source_height)
    return (content_area, cx, cy)


def _zoom_crop_size(
    content_area: float,
    source_width: int,
    source_height: int,
    target_percent: int,
    target_w: int,
    target_h: int,
) -> tuple[int, int]:
    """
    Compute crop (w, h) from content area and target frame percent, with aspect ratio
    target_w/target_h. Clamps to [ZOOM_MIN_FRAME_FRACTION * source, source] and
    returns even dimensions.
    """
    if content_area <= 0 or target_percent <= 0 or target_w <= 0 or target_h <= 0:
        return (target_w, target_h)
    if source_width <= 0 or source_height <= 0:
        return (target_w, target_h)
    desired_crop_area = content_area / (target_percent / 100.0)
    aspect = target_w / target_h
    crop_w_f = math.sqrt(desired_crop_area * aspect)
    crop_h_f = math.sqrt(desired_crop_area / aspect)
    min_w = max(1, int(source_width * ZOOM_MIN_FRAME_FRACTION))
    min_h = max(1, int(source_height * ZOOM_MIN_FRAME_FRACTION))
    crop_w = max(min_w, min(source_width, int(crop_w_f)))
    crop_h = max(min_h, min(source_height, int(crop_h_f)))
    crop_w = max(1, (crop_w & ~1))
    crop_h = max(1, (crop_h & ~1))
    if crop_w > source_width or crop_h > source_height:
        crop_w = min(crop_w, source_width)
        crop_h = min(crop_h, source_height)
        crop_w = max(1, (crop_w & ~1))
        crop_h = max(1, (crop_h & ~1))
    return (crop_w, crop_h)


def calculate_crop_at_time(
    sidecar_data: dict,
    t_sec: float,
    source_width: int,
    source_height: int,
    target_w: int = 1440,
    target_h: int = 1080,
    *,
    timestamps_sorted: list[float] | None = None,
    tracking_target_frame_percent: int = 0,
) -> tuple[int, int, int, int]:
    """
    Compute crop (x, y, w, h) at a single timestamp from the nearest sidecar entry.
    Uses area-weighted center of detections. When tracking_target_frame_percent > 0
    and detections exist, crop size is derived from content (bbox + padding) vs target
    percent for dynamic zoom; otherwise uses fixed target_w/target_h. Returns clamped rect.
    If timestamps_sorted is provided (same order as sidecar entries), lookup is O(log n).
    """
    entries = sidecar_data.get("entries") or []
    content_area: float = 0.0
    if not entries:
        avg_cx = source_width / 2.0
        avg_cy = source_height / 2.0
        use_zoom = False
    else:
        entry = _nearest_entry_at_t(entries, t_sec, timestamps_sorted)
        if entry and len(entry.get("detections") or []) == 0:
            fallback = _nearest_entry_with_detections_at_t(
                entries, t_sec, HOLD_CROP_MAX_DISTANCE_SEC, timestamps_sorted
            )
            if fallback is not None:
                entry = fallback
        if not entry:
            avg_cx = source_width / 2.0
            avg_cy = source_height / 2.0
            use_zoom = False
        else:
            dets = entry.get("detections") or []
            if tracking_target_frame_percent > 0 and dets:
                content_area, avg_cx, avg_cy = _content_area_and_center(
                    dets, source_width, source_height
                )
                use_zoom = content_area > 0
            else:
                avg_cx, avg_cy = _weighted_center_from_detections(
                    dets, source_width, source_height
                )
                use_zoom = False

    if use_zoom:
        crop_w, crop_h = _zoom_crop_size(
            content_area,
            source_width,
            source_height,
            tracking_target_frame_percent,
            target_w,
            target_h,
        )
    else:
        if source_width < target_w or source_height < target_h:
            scale = min(source_width / target_w, source_height / target_h)
            target_w = int(target_w * scale)
            target_h = int(target_h * scale)
        crop_w, crop_h = target_w, target_h

    x = int(avg_cx - crop_w / 2.0)
    y = int(avg_cy - crop_h / 2.0)
    x = max(0, min(source_width - crop_w, x))
    y = max(0, min(source_height - crop_h, y))
    return (x, y, crop_w, crop_h)


def smooth_zoom_ema(
    slices: list[dict],
    alpha: float,
    target_w: int,
    target_h: int,
) -> None:
    """
    Smooth zoom (crop size) in-place with EMA so zoom changes are fluid, not snapping.
    Slices must have crop_start, crop_end as (x, y, w, h) and native_width, native_height.
    Derives a zoom scalar from crop area / frame area, runs EMA, then recomputes (w, h)
    from smoothed zoom with aspect ratio target_w/target_h and clamps. Reclamps (x, y)
    so the crop stays within source bounds after size change.
    """
    if alpha <= 0 or alpha >= 1 or not slices or target_w <= 0 or target_h <= 0:
        return
    aspect = target_w / target_h
    smooth_zoom_s: float = 0.0
    smooth_zoom_e: float = 0.0
    for idx, sl in enumerate(slices):
        xs, ys, ws, hs = sl.get("crop_start", (0, 0, 1440, 1080))
        xe, ye, we, he = sl.get("crop_end", (0, 0, 1440, 1080))
        sw = int(sl.get("native_width") or 0)
        sh = int(sl.get("native_height") or 0)
        if sw <= 0 or sh <= 0:
            continue
        frame_area = sw * sh
        zoom_s = math.sqrt((ws * hs) / frame_area) if frame_area > 0 else 1.0
        zoom_e = math.sqrt((we * he) / frame_area) if frame_area > 0 else 1.0
        zoom_s = max(ZOOM_MIN_FRAME_FRACTION, min(1.0, zoom_s))
        zoom_e = max(ZOOM_MIN_FRAME_FRACTION, min(1.0, zoom_e))
        if idx == 0:
            smooth_zoom_s, smooth_zoom_e = zoom_s, zoom_e
        else:
            smooth_zoom_s = alpha * zoom_s + (1.0 - alpha) * smooth_zoom_s
            smooth_zoom_e = alpha * zoom_e + (1.0 - alpha) * smooth_zoom_e
        crop_area_s = (smooth_zoom_s**2) * frame_area
        crop_area_e = (smooth_zoom_e**2) * frame_area
        nw_s = max(1, min(sw, int(math.sqrt(crop_area_s * aspect))))
        nh_s = max(1, min(sh, int(math.sqrt(crop_area_s / aspect))))
        nw_e = max(1, min(sw, int(math.sqrt(crop_area_e * aspect))))
        nh_e = max(1, min(sh, int(math.sqrt(crop_area_e / aspect))))
        nw_s, nh_s = max(1, (nw_s & ~1)), max(1, (nh_s & ~1))
        nw_e, nh_e = max(1, (nw_e & ~1)), max(1, (nh_e & ~1))
        cx_s = xs + ws / 2.0
        cy_s = ys + hs / 2.0
        cx_e = xe + we / 2.0
        cy_e = ye + he / 2.0
        new_xs = max(0, min(sw - nw_s, int(cx_s - nw_s / 2.0)))
        new_ys = max(0, min(sh - nh_s, int(cy_s - nh_s / 2.0)))
        new_xe = max(0, min(sw - nw_e, int(cx_e - nw_e / 2.0)))
        new_ye = max(0, min(sh - nh_e, int(cy_e - nh_e / 2.0)))
        sl["crop_start"] = (new_xs, new_ys, nw_s, nh_s)
        sl["crop_end"] = (new_xe, new_ye, nw_e, nh_e)


def smooth_crop_centers_ema(slices: list[dict], alpha: float) -> None:
    """
    Smooth crop center trajectory in-place with single-pass EMA to reduce detection jitter.
    Slices must have "crop_start" and "crop_end" as (x, y, w, h). Updates those with
    smoothed (x, y) derived from EMA of center (cx, cy) = (x + w/2, y + h/2).
    Clamps center so the crop stays within native_width/native_height.
    """
    if alpha <= 0 or alpha >= 1 or not slices:
        return
    smooth_cx_s: float = 0.0
    smooth_cy_s: float = 0.0
    smooth_cx_e: float = 0.0
    smooth_cy_e: float = 0.0
    for idx, sl in enumerate(slices):
        xs, ys, ws, hs = sl.get("crop_start", (0, 0, 1440, 1080))
        xe, ye, we, he = sl.get("crop_end", (0, 0, 1440, 1080))
        sw = int(sl.get("native_width") or 0)
        sh = int(sl.get("native_height") or 0)
        cx_s = xs + ws / 2.0
        cy_s = ys + hs / 2.0
        cx_e = xe + we / 2.0
        cy_e = ye + he / 2.0
        if idx == 0:
            smooth_cx_s, smooth_cy_s = cx_s, cy_s
            smooth_cx_e, smooth_cy_e = cx_e, cy_e
        else:
            smooth_cx_s = alpha * cx_s + (1.0 - alpha) * smooth_cx_s
            smooth_cy_s = alpha * cy_s + (1.0 - alpha) * smooth_cy_s
            smooth_cx_e = alpha * cx_e + (1.0 - alpha) * smooth_cx_e
            smooth_cy_e = alpha * cy_e + (1.0 - alpha) * smooth_cy_e
        if sw > 0 and sh > 0:
            smooth_cx_s = max(ws / 2.0, min(sw - ws / 2.0, smooth_cx_s))
            smooth_cy_s = max(hs / 2.0, min(sh - hs / 2.0, smooth_cy_s))
            smooth_cx_e = max(we / 2.0, min(sw - we / 2.0, smooth_cx_e))
            smooth_cy_e = max(he / 2.0, min(sh - he / 2.0, smooth_cy_e))
        sl["crop_start"] = (
            int(smooth_cx_s - ws / 2.0),
            int(smooth_cy_s - hs / 2.0),
            ws,
            hs,
        )
        sl["crop_end"] = (
            int(smooth_cx_e - we / 2.0),
            int(smooth_cy_e - he / 2.0),
            we,
            he,
        )


def calculate_segment_crop(
    segment: dict,
    sidecar_data: dict,
    source_width: int,
    source_height: int,
    target_w: int = 1440,
    target_h: int = 1080,
) -> tuple[int, int, int, int]:
    """
    Finds the center of mass for a segment based on sidecar detections in [start_sec, end_sec)
    and returns clamped (x, y, w, h) crop variables.
    """
    start_sec = segment["start_sec"]
    end_sec = segment["end_sec"]

    if source_width < target_w or source_height < target_h:
        scale = min(source_width / target_w, source_height / target_h)
        target_w = int(target_w * scale)
        target_h = int(target_h * scale)

    collected: list[dict] = []
    for entry in sidecar_data.get("entries", []):
        t = entry.get("timestamp_sec", 0.0)
        if start_sec <= t < end_sec:
            collected.extend(entry.get("detections") or [])

    avg_cx, avg_cy = _weighted_center_from_detections(
        collected, source_width, source_height
    )

    x = int(avg_cx - target_w / 2.0)
    y = int(avg_cy - target_h / 2.0)
    x = max(0, min(source_width - target_w, x))
    y = max(0, min(source_height - target_h, y))

    return (x, y, target_w, target_h)
