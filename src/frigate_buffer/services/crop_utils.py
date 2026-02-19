"""
Shared crop and overlay helpers for AI analyzer and multi-cam frame extractor.

Single source of truth for CV-based motion crop (absdiff, contours, minimum area)
and timestamp overlay (time, camera name, photo sequence) with shadow/outline for contrast.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger("frigate-buffer")

# Default minimum area for motion-based crop (avoid noise pulling crop)
# Fraction of frame area; if largest motion blob is below this, fall back to center crop.
DEFAULT_MOTION_CROP_MIN_AREA_FRACTION = 0.001
# Alternative: minimum area in pixels (e.g. 500). If both set, use the stricter.
DEFAULT_MOTION_CROP_MIN_PX = 500


def center_crop(frame: Any, target_w: int, target_h: int) -> Any:
    """Crop frame to target_w x target_h centered. Resize if crop larger than frame."""
    logger.debug("Cropping frame (center)")
    h, w = frame.shape[:2]
    if target_w <= 0 or target_h <= 0:
        return frame
    x1 = max(0, (w - target_w) // 2)
    y1 = max(0, (h - target_h) // 2)
    x2 = min(w, x1 + target_w)
    y2 = min(h, y1 + target_h)
    crop = frame[y1:y2, x1:x2]
    if crop.shape[1] != target_w or crop.shape[0] != target_h:
        crop = cv2.resize(crop, (target_w, target_h))
    return crop


def crop_around_center(
    frame: Any,
    center_x: float | int,
    center_y: float | int,
    target_w: int,
    target_h: int,
) -> Any:
    """
    Crop frame centered on (center_x, center_y), clamped to frame bounds, then resize to target.

    YOLO/Ultralytics often return float centerpoints; we cast to int before slice bounds
    so OpenCV/NumPy array indexing does not raise TypeError.
    """
    cx = int(center_x)
    cy = int(center_y)
    logger.debug("Cropping frame (centerpoint)")
    return _crop_around_center(frame, cx, cy, target_w, target_h)


def _crop_around_center(
    frame: Any, center_x: int, center_y: int, target_w: int, target_h: int
) -> Any:
    """Crop frame centered on (center_x, center_y), clamped to frame bounds, then resize to target."""
    h, w = frame.shape[:2]
    if target_w <= 0 or target_h <= 0:
        return frame
    x1 = center_x - target_w // 2
    y1 = center_y - target_h // 2
    x2 = x1 + target_w
    y2 = y1 + target_h
    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > w:
        x1 -= x2 - w
        x2 = w
    if y2 > h:
        y1 -= y2 - h
        y2 = h
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return center_crop(frame, target_w, target_h)
    if crop.shape[1] != target_w or crop.shape[0] != target_h:
        crop = cv2.resize(crop, (target_w, target_h))
    return crop


def full_frame_resize_to_target(frame: Any, target_w: int, target_h: int) -> Any:
    """
    Scale the entire frame to fit inside target_w x target_h preserving aspect ratio, then pad with black to exactly target_w x target_h.

    Used when the subject occupies too much of the crop area (e.g. 40% threshold); the AI proxy receives full context with letterboxing instead of a tight crop.
    """
    if target_w <= 0 or target_h <= 0:
        return frame
    h, w = frame.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    scaled = cv2.resize(frame, (new_w, new_h))
    # Pad to exact target size (letterbox): top/bottom or left/right black bars
    top = (target_h - new_h) // 2
    bottom = target_h - new_h - top
    left = (target_w - new_w) // 2
    right = target_w - new_w - left
    return cv2.copyMakeBorder(scaled, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))


def motion_crop(
    frame: Any,
    prev_gray: Any | None,
    target_w: int,
    target_h: int,
    min_area_fraction: float = DEFAULT_MOTION_CROP_MIN_AREA_FRACTION,
    min_area_px: int = DEFAULT_MOTION_CROP_MIN_PX,
    center_override: tuple[int, int] | None = None,
) -> tuple[Any, Any]:
    """
    Crop frame centered on the largest motion region (absdiff + contours).
    If prev_gray is None or motion is below threshold, return center crop.
    Returns (cropped_frame, next_gray) so caller can pass next_gray as prev_gray for next frame.

    Optional: center_override (cx, cy) — when provided (e.g. from an on-device person detector),
    use this center instead of motion. Enables "center on person" when a detector is used;
    add e.g. Ultralytics YOLO or OpenCV DNN in the caller and pass the bbox center here.
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if center_override is not None:
        cx, cy = center_override
        cropped = _crop_around_center(frame, cx, cy, target_w, target_h)
        return cropped, gray

    if prev_gray is None:
        return center_crop(frame, target_w, target_h), gray

    diff = cv2.absdiff(prev_gray, gray)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return center_crop(frame, target_w, target_h), gray

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    frame_area = h * w
    min_from_fraction = frame_area * min_area_fraction
    min_effective = max(min_area_px, min_from_fraction)

    if area < min_effective:
        return center_crop(frame, target_w, target_h), gray

    M = cv2.moments(largest)
    if M["m00"] == 0:
        return center_crop(frame, target_w, target_h), gray
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    cropped = _crop_around_center(frame, cx, cy, target_w, target_h)
    return cropped, gray


def draw_timestamp_overlay(
    frame: Any,
    time_str: str,
    camera_name: str,
    seq_index: int,
    seq_total: int,
    font_scale: float = 0.7,
    thickness_outline: int = 2,
    thickness_text: int = 1,
    position: tuple[int, int] = (10, 30),
) -> Any:
    """
    Draw timestamp overlay on frame (top-left by default).

    OpenCV putText requires a writable buffer; if the frame is read-only (e.g. from
    ffmpegcv or a crop view), a copy is made so drawing succeeds. Callers must use
    the returned frame—the overlay is drawn on that array.

    Uses shadow/outline: black with thicker stroke first, then white with thinner
    stroke, so text is readable on both bright (snow) and dark (night) backgrounds.
    Format: time_str | camera_name | seq_index/seq_total (e.g. "12:34:56 | Doorbell | 3/24").

    Returns:
        The frame with overlay drawn (same object if already writable, else a copy).
    """
    if not getattr(frame, "flags", None) or not frame.flags.writeable:
        frame = np.array(frame, copy=True)
    label = f"{time_str} | {camera_name} | {seq_index}/{seq_total}"
    # Black outline (thicker) first
    cv2.putText(
        frame,
        label,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 0),
        thickness_outline,
        cv2.LINE_AA,
    )
    # White text (thinner) on top
    cv2.putText(
        frame,
        label,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        thickness_text,
        cv2.LINE_AA,
    )
    return frame
