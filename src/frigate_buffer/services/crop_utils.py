"""
Shared crop and overlay helpers for AI analyzer and multi-cam frame extractor.

All crop/resize/letterbox functions accept PyTorch tensors in BCHW format.
Motion crop uses GPU absdiff with int16 cast to avoid uint8 underflow; only the
1-bit mask is transferred to CPU for cv2.findContours. Timestamp overlay accepts
tensor or numpy (Phase 1 compat); converts RGB→BGR only at the OpenCV boundary
and returns numpy HWC BGR.
"""

from __future__ import annotations

import logging
from typing import Any, cast

import cv2
import numpy as np

from frigate_buffer.constants import is_tensor

logger = logging.getLogger("frigate-buffer")

# Default minimum area for motion-based crop (avoid noise pulling crop)
# Fraction of frame area; if largest motion blob is below this,
# fall back to center crop.
DEFAULT_MOTION_CROP_MIN_AREA_FRACTION = 0.001
# Alternative: minimum area in pixels (e.g. 500). If both set, use the stricter.
DEFAULT_MOTION_CROP_MIN_PX = 500


def _get_tensor_hw(tensor: Any) -> tuple[int, int]:
    """Return (height, width) from BCHW tensor. Raises if not 4D."""
    if tensor.dim() != 4:
        raise ValueError(f"Expected BCHW tensor with 4 dims, got dim()={tensor.dim()}")
    return int(tensor.shape[2]), int(tensor.shape[3])


def center_crop(frame: Any, target_w: int, target_h: int) -> Any:
    """
    Crop frame to target_w x target_h centered. Resize if crop larger than frame.

    frame: torch.Tensor BCHW (batch, channels, height, width). Returns BCHW tensor.
    """
    import torch
    import torch.nn.functional as F

    if not is_tensor(frame):
        raise TypeError("center_crop expects torch.Tensor BCHW")
    h, w = _get_tensor_hw(frame)
    if target_w <= 0 or target_h <= 0:
        return frame
    x1 = max(0, (w - target_w) // 2)
    y1 = max(0, (h - target_h) // 2)
    x2 = min(w, x1 + target_w)
    y2 = min(h, y1 + target_h)
    crop = frame[:, :, y1:y2, x1:x2]
    if crop.shape[2] != target_h or crop.shape[3] != target_w:
        crop = F.interpolate(
            crop.float(),
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )
        if frame.dtype == torch.uint8:
            crop = crop.clamp(0, 255).round().to(torch.uint8)
    return crop


def crop_around_center(
    frame: Any,
    center_x: float | int,
    center_y: float | int,
    target_w: int,
    target_h: int,
) -> Any:
    """
    Crop frame centered on (center_x, center_y), clamped to frame bounds,
    then resize to target.

    frame: torch.Tensor BCHW. Returns BCHW tensor.
    """
    cx = int(center_x)
    cy = int(center_y)
    return _crop_around_center(frame, cx, cy, target_w, target_h)


def _crop_around_center(
    frame: Any, center_x: int, center_y: int, target_w: int, target_h: int
) -> Any:
    """Crop frame centered on (center_x, center_y), clamped to frame bounds,
    then resize to target. BCHW in/out."""
    import torch
    import torch.nn.functional as F

    if not is_tensor(frame):
        raise TypeError("_crop_around_center expects torch.Tensor BCHW")
    h, w = _get_tensor_hw(frame)
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
    crop = frame[:, :, y1:y2, x1:x2]
    if crop.numel() == 0:
        return center_crop(frame, target_w, target_h)
    if crop.shape[2] != target_h or crop.shape[3] != target_w:
        crop = F.interpolate(
            crop.float(),
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )
        if frame.dtype == torch.uint8:
            crop = crop.clamp(0, 255).round().to(torch.uint8)
    return crop


def crop_around_center_to_size(
    frame: Any,
    center_x: float | int,
    center_y: float | int,
    crop_w: int,
    crop_h: int,
    output_w: int,
    output_h: int,
) -> Any:
    """
    Crop a region of size (crop_w, crop_h) centered at (center_x, center_y),
    clamped to frame bounds, then resize to (output_w, output_h). Used for
    compilation with variable zoom; output size is fixed for the encoder.

    frame: torch.Tensor BCHW. Returns BCHW tensor of shape (B, C, output_h, output_w).
    Uses bicubic interpolation for the resize step (sharper when downscaling).
    """
    import torch
    import torch.nn.functional as F

    if not is_tensor(frame):
        raise TypeError("crop_around_center_to_size expects torch.Tensor BCHW")
    h, w = _get_tensor_hw(frame)
    if crop_w <= 0 or crop_h <= 0 or output_w <= 0 or output_h <= 0:
        return frame
    cx = int(center_x)
    cy = int(center_y)
    x1 = cx - crop_w // 2
    y1 = cy - crop_h // 2
    x2 = x1 + crop_w
    y2 = y1 + crop_h
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
    crop = frame[:, :, y1:y2, x1:x2]
    if crop.numel() == 0:
        return center_crop(frame, output_w, output_h)
    if crop.shape[2] != output_h or crop.shape[3] != output_w:
        crop = F.interpolate(
            crop.float(),
            size=(output_h, output_w),
            mode="bicubic",
            align_corners=False,
        )
        if frame.dtype == torch.uint8:
            crop = crop.clamp(0, 255).round().to(torch.uint8)
    return crop


def crop_around_detections_with_padding(
    frame: Any,
    detections: list[dict[str, Any]],
    padding_fraction: float = 0.1,
) -> Any:
    """
    Crop the frame to a region that encompasses ALL detections plus padding.

    frame: torch.Tensor BCHW. Returns BCHW (variable size crop; no fixed resize).
    Each detection must have "bbox" as [x1, y1, x2, y2] in pixel coordinates.
    """
    if not is_tensor(frame):
        raise TypeError("crop_around_detections_with_padding expects torch.Tensor BCHW")
    h, w = _get_tensor_hw(frame)
    if not detections:
        return frame
    if w <= 0 or h <= 0:
        return frame
    x1_min = float(w)
    y1_min = float(h)
    x2_max = 0.0
    y2_max = 0.0
    for d in detections:
        bbox = d.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            continue
        x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        x1_min = min(x1_min, x1)
        y1_min = min(y1_min, y1)
        x2_max = max(x2_max, x2)
        y2_max = max(y2_max, y2)
    if x1_min >= x2_max or y1_min >= y2_max:
        return frame
    bw = x2_max - x1_min
    bh = y2_max - y1_min
    pad_w = max(bw * padding_fraction, 1)
    pad_h = max(bh * padding_fraction, 1)
    x1 = int(x1_min - pad_w)
    y1 = int(y1_min - pad_h)
    x2 = int(x2_max + pad_w)
    y2 = int(y2_max + pad_h)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return frame
    crop = frame[:, :, y1:y2, x1:x2]
    return crop if crop.numel() > 0 else frame


def full_frame_resize_to_target(frame: Any, target_w: int, target_h: int) -> Any:
    """
    Scale the entire frame to fit inside target_w x target_h preserving aspect ratio,
    then pad with black to exactly target_w x target_h (letterbox).

    frame: torch.Tensor BCHW. Returns BCHW tensor.
    """
    import torch
    import torch.nn.functional as F

    if not is_tensor(frame):
        raise TypeError("full_frame_resize_to_target expects torch.Tensor BCHW")
    if target_w <= 0 or target_h <= 0:
        return frame
    _, _, h, w = frame.shape
    h, w = int(h), int(w)
    scale = min(target_w / w, target_h / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    scaled = F.interpolate(
        frame.float(),
        size=(new_h, new_w),
        mode="bilinear",
        align_corners=False,
    )
    if frame.dtype == torch.uint8:
        scaled = scaled.clamp(0, 255).round().to(torch.uint8)
    # Letterbox: pad to (target_h, target_w)
    pad_top = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top
    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left
    padded = F.pad(
        scaled,
        (pad_left, pad_right, pad_top, pad_bottom),
        mode="constant",
        value=0,
    )
    return padded


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
    Crop frame centered on the largest motion region (absdiff + contours on GPU;
    findContours on CPU).

    frame, prev_gray: torch.Tensor. frame is BCHW RGB; prev_gray is 1CHW or HW.
    Returns (cropped_frame BCHW, next_gray 1CHW or HW). Cast to int16 before
    subtraction to avoid uint8 underflow.
    """
    import torch

    if not is_tensor(frame):
        raise TypeError("motion_crop expects frame as torch.Tensor BCHW")
    _, c, h, w = frame.shape
    # Grayscale from RGB: 0.299*R + 0.587*G + 0.114*B (on GPU)
    if c == 3:
        gray = (
            0.299 * frame[:, 0:1] + 0.587 * frame[:, 1:2] + 0.114 * frame[:, 2:3]
        ).to(frame.dtype)
    else:
        gray = frame[:, :1]

    if center_override is not None:
        cx, cy = center_override
        cropped = _crop_around_center(frame, cx, cy, target_w, target_h)
        return cropped, gray

    if prev_gray is None:
        return center_crop(frame, target_w, target_h), gray

    # Cast to int16 before subtraction to avoid uint8 underflow
    # (e.g. 50 - 60 = 246 in uint8).
    prev_s = prev_gray.to(torch.int16)
    gray_s = gray.to(torch.int16)
    diff = torch.abs(prev_s - gray_s)
    thresh = (diff > 25).to(torch.uint8) * 255
    # Transfer only the 1-bit mask to CPU for findContours
    mask_cpu = thresh.squeeze().cpu().numpy()
    if mask_cpu.ndim == 3:
        mask_cpu = mask_cpu[0]
    contours, _ = cv2.findContours(mask_cpu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
    frame: Any | np.ndarray,
    time_str: str,
    camera_name: str,
    seq_index: int,
    seq_total: int,
    font_scale: float = 0.7,
    thickness_outline: int = 2,
    thickness_text: int = 1,
    position: tuple[int, int] = (10, 30),
    person_area: int | None = None,
) -> Any:
    """
    Draw timestamp overlay on frame (top-left by default).

    Accepts torch.Tensor BCHW RGB or numpy ndarray HWC BGR (Phase 1 compat).
    For tensor: converts to HWC BGR at OpenCV boundary, draws, returns numpy HWC BGR.
    For numpy: draws in-place (or copy if read-only) and returns the array.
    """
    if is_tensor(frame):
        import torch
        t = cast(torch.Tensor, frame)
        # BCHW RGB → HWC BGR for OpenCV
        if t.dim() == 4:
            frame_np = (
                t[:, [2, 1, 0], :, :].permute(0, 2, 3, 1).squeeze(0).cpu().numpy()
            )
        else:
            # CHW
            frame_np = t[[2, 1, 0], :, :].permute(1, 2, 0).cpu().numpy()
        frame = np.ascontiguousarray(frame_np)
    else:
        if not getattr(frame, "flags", None) or not frame.flags.writeable:
            frame = np.array(frame, copy=True)

    label = f"{time_str} | {camera_name} | {seq_index}/{seq_total}"
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
    if person_area is not None:
        br_label = f"person_area: {person_area}"
        h_frame, w_frame = frame.shape[:2]
        margin = 10
        (tw, _), _ = cv2.getTextSize(
            br_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness_outline
        )
        br_x = w_frame - tw - margin
        br_y = h_frame - margin
        cv2.putText(
            frame,
            br_label,
            (br_x, br_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            thickness_outline,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            br_label,
            (br_x, br_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness_text,
            cv2.LINE_AA,
        )
    return frame
