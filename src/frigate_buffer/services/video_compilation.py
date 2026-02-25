"""
Video compilation service.
Handles segment-level video processing using hardware acceleration.
Generates single, stitched, cropped compilation videos.
"""

from __future__ import annotations

import bisect
import json
import logging
import os
import subprocess
from collections import Counter, defaultdict

from frigate_buffer.constants import NVDEC_INIT_FAILURE_PREFIX, ZOOM_CONTENT_PADDING, ZOOM_MIN_FRAME_FRACTION
from frigate_buffer.services import crop_utils
from frigate_buffer.services import timeline_ema
from frigate_buffer.services.gpu_decoder import create_decoder
from frigate_buffer.services.video import GPU_LOCK, _get_video_metadata

logger = logging.getLogger("frigate-buffer")

# Output frame rate for compilation (smooth panning samples at this rate).
COMPILATION_OUTPUT_FPS = 20

# Chunk size for batched decode to protect VRAM (same pattern as video.py). Decoder returns
# up to this many frames per get_frames call; after each chunk we del batch and empty_cache.
BATCH_SIZE = 4

# When the nearest sidecar entry has no detections (e.g. person left frame), search for the
# nearest entry with detections within this many seconds and hold that crop instead of panning to center.
HOLD_CROP_MAX_DISTANCE_SEC = 5.0

# Dynamic trimming: pre-roll and post-roll around first/last detection across all cameras.
ACTION_PREROLL_SEC = 3.0
ACTION_POSTROLL_SEC = 3.0


def convert_timeline_to_segments(timeline_points: list[tuple[float, str]], global_end: float) -> list[dict]:
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
            segments.append({
                "camera": current_cam,
                "start_sec": seg_start,
                "end_sec": t
            })
            current_cam = cam
            seg_start = t

    segments.append({
        "camera": current_cam,
        "start_sec": seg_start,
        "end_sec": global_end
    })

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
        slices.append({
            "camera": camera,
            "start_sec": t_sec,
            "end_sec": end_sec,
        })
    return slices


def _trim_slices_to_action_window(
    slices: list[dict],
    sidecars: dict[str, list],
    global_end: float,
) -> list[dict]:
    """
    Compute first/last detection time across all cameras' sidecar entries (with detections),
    then discard slices entirely outside [action_start, action_end] and clamp overlapping
    slices to that window. Returns a new list; does not mutate input.
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
        if abs(timestamps_sorted[idx] - t_sec) < abs(timestamps_sorted[idx - 1] - t_sec):
            return entries[idx]
        return entries[idx - 1]
    return min(entries, key=lambda e: abs((e.get("timestamp_sec") or 0) - t_sec))


def _nearest_entry_with_detections_at_t(
    entries: list[dict],
    t_sec: float,
    max_distance_sec: float,
    timestamps_sorted: list[float] | None = None,  # noqa: ARG001 — reserved for future bisect optimization
) -> dict | None:
    """
    Return the sidecar entry with detections whose timestamp_sec is nearest to t_sec
    and within max_distance_sec. Used to hold the last-known crop when a person leaves
    the frame instead of panning to center.
    """
    candidates = [
        e for e in entries
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
    Returns (content_area, center_x, center_y). If no valid detections, returns (0, frame_center_x, frame_center_y).
    """
    if not detections or source_width <= 0 or source_height <= 0:
        return (0.0, source_width / 2.0, source_height / 2.0)
    x1_min = float(source_width)
    y1_min = float(source_height)
    x2_max = 0.0
    y2_max = 0.0
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
            x1_min = min(x1_min, x1)
            y1_min = min(y1_min, y1)
            x2_max = max(x2_max, x2)
            y2_max = max(y2_max, y2)
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
    if x1_min >= x2_max or y1_min >= y2_max or total_area <= 0:
        return (0.0, source_width / 2.0, source_height / 2.0)
    bw = x2_max - x1_min
    bh = y2_max - y1_min
    pad_w = max(bw * padding, 1.0)
    pad_h = max(bh * padding, 1.0)
    content_x1 = max(0.0, x1_min - pad_w)
    content_y1 = max(0.0, y1_min - pad_h)
    content_x2 = min(float(source_width), x2_max + pad_w)
    content_y2 = min(float(source_height), y2_max + pad_h)
    content_area = (content_x2 - content_x1) * (content_y2 - content_y1)
    if total_area > 0:
        cx_avg = weighted_cx_sum / total_area
        cy_avg = weighted_cy_sum / total_area
    else:
        cx_avg = source_width / 2.0
        cy_avg = source_height / 2.0
    return (content_area, cx_avg, cy_avg)


def _zoom_crop_size(
    content_area: float,
    source_width: int,
    source_height: int,
    target_percent: int,
    target_w: int,
    target_h: int,
) -> tuple[int, int]:
    """
    Compute crop (w, h) from content area and target frame percent, with aspect ratio target_w/target_h.
    Clamps to [ZOOM_MIN_FRAME_FRACTION * source, source] and returns even dimensions.
    """
    import math
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
    Uses area-weighted center of detections. When tracking_target_frame_percent > 0 and
    detections exist, crop size is derived from content (bbox + padding) vs target percent
    for dynamic zoom; otherwise uses fixed target_w/target_h. Returns clamped rect.
    If timestamps_sorted is provided (same order as sidecar entries), lookup is O(log n).
    Time O(log(entries)) with bisect, O(entries) without; space O(1).
    """
    entries = sidecar_data.get("entries") or []
    content_area: float = 0.0
    if not entries:
        avg_cx = source_width / 2.0
        avg_cy = source_height / 2.0
        use_zoom = False
    else:
        entry = _nearest_entry_at_t(entries, t_sec, timestamps_sorted)
        # Hold last-known crop: if nearest entry has no detections (e.g. person left frame),
        # use the nearest entry with detections within threshold instead of panning to center.
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
                weighted_cx_sum: float = 0.0
                weighted_cy_sum: float = 0.0
                total_area: float = 0.0
                for det in dets:
                    area = float(det.get("area", 1.0))
                    if area <= 0:
                        area = 1.0
                    cp = det.get("centerpoint")
                    if isinstance(cp, (list, tuple)) and len(cp) >= 2:
                        weighted_cx_sum += float(cp[0]) * area
                        weighted_cy_sum += float(cp[1]) * area
                        total_area += area
                    else:
                        box = det.get("box") or det.get("bbox")
                        if isinstance(box, (list, tuple)) and len(box) >= 4:
                            cx = (float(box[0]) + float(box[2])) / 2.0
                            cy = (float(box[1]) + float(box[3])) / 2.0
                            weighted_cx_sum += cx * area
                            weighted_cy_sum += cy * area
                            total_area += area
                if total_area > 0:
                    avg_cx = weighted_cx_sum / total_area
                    avg_cy = weighted_cy_sum / total_area
                else:
                    avg_cx = source_width / 2.0
                    avg_cy = source_height / 2.0
                use_zoom = False

    if use_zoom:
        crop_w, crop_h = _zoom_crop_size(
            content_area, source_width, source_height,
            tracking_target_frame_percent, target_w, target_h,
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
    import math
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
        crop_area_s = (smooth_zoom_s ** 2) * frame_area
        crop_area_e = (smooth_zoom_e ** 2) * frame_area
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


def smooth_crop_centers_ema(
    slices: list[dict], alpha: float
) -> None:
    """
    Smooth crop center trajectory in-place with single-pass EMA to reduce detection jitter.
    Slices must have "crop_start" and "crop_end" as (x, y, w, h). Updates those with
    smoothed (x, y) derived from EMA of center (cx, cy) = (x + w/2, y + h/2).
    Clamps center so the crop stays within native_width/native_height (avoids out-of-bounds after zoom).
    Time O(n), space O(1) for EMA state.
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
    target_h: int = 1080
) -> tuple[int, int, int, int]:
    """
    Finds the center of mass for a segment based on sidecar detections
    and returns clamped (x, y, w, h) crop variables.
    """
    start_sec = segment["start_sec"]
    end_sec = segment["end_sec"]

    # Prevent target from being larger than source; simulate letterbox or scale down
    if source_width < target_w or source_height < target_h:
        scale = min(source_width / target_w, source_height / target_h)
        target_w = int(target_w * scale)
        target_h = int(target_h * scale)

    weighted_cx_sum: float = 0.0
    weighted_cy_sum: float = 0.0
    total_area: float = 0.0

    for entry in sidecar_data.get("entries", []):
        t = entry.get("timestamp_sec", 0.0)
        # We consider inclusive start, exclusive end for frame gathering typically
        if start_sec <= t < end_sec:
            for det in entry.get("detections", []):
                area = float(det.get("area", 1.0))
                if area <= 0:
                    area = 1.0
                    
                cp = det.get("centerpoint")
                if isinstance(cp, (list, tuple)) and len(cp) >= 2:
                    weighted_cx_sum += float(cp[0]) * area
                    weighted_cy_sum += float(cp[1]) * area
                    total_area += area
                else:
                    box = det.get("box") or det.get("bbox")
                    if isinstance(box, (list, tuple)) and len(box) >= 4:
                        cx = (float(box[0]) + float(box[2])) / 2.0
                        cy = (float(box[1]) + float(box[3])) / 2.0
                        weighted_cx_sum += cx * area
                        weighted_cy_sum += cy * area
                        total_area += area

    if total_area > 0:
        # EMA simulation using area-weighted generic mass tracker
        avg_cx = weighted_cx_sum / total_area
        avg_cy = weighted_cy_sum / total_area
    else:
        # Fallback to center if no detections
        avg_cx = source_width / 2.0
        avg_cy = source_height / 2.0

    # Calculate top-left for the crop box
    x = int(avg_cx - target_w / 2.0)
    y = int(avg_cy - target_h / 2.0)

    # Clamp the coordinates to ensure the crop box doesn't exceed video boundaries
    x = max(0, min(source_width - target_w, x))
    y = max(0, min(source_height - target_h, y))

    return (x, y, target_w, target_h)


def _compilation_ffmpeg_cmd_and_log_path(
    tmp_output_path: str, target_w: int, target_h: int
) -> tuple[list[str], str]:
    """
    Build FFmpeg h264_nvenc command and log path for compilation encode.
    Shared by _encode_frames_via_ffmpeg and _run_pynv_compilation (streaming path).
    """
    log_file_path = os.path.join(os.path.dirname(tmp_output_path), "ffmpeg_compile.log")
    cmd = [
        "ffmpeg",
        "-y",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{target_w}x{target_h}",
        "-r", "20",
        "-thread_queue_size", "512",
        "-i", "pipe:0",
        "-c:v", "h264_nvenc",
        "-preset", "p1",
        "-tune", "hq",
        "-rc", "vbr",
        "-cq", "24",
        "-an",
        "-pix_fmt", "yuv420p",
        tmp_output_path,
    ]
    return cmd, log_file_path


def _encode_frames_via_ffmpeg(
    frames: list,
    target_w: int,
    target_h: int,
    tmp_output_path: str,
) -> None:
    """
    Encode a list of RGB frames to MP4 using FFmpeg with h264_nvenc only (GPU).

    frames: list of numpy arrays (H, W, 3) uint8 RGB with H=target_h, W=target_w.
    FFmpeg stderr is routed to a physical log file in the event folder to avoid
    OS-level pipe deadlock. h264_nvenc cannot ingest rgb24; we request yuv420p output.
    No CPU fallback; on failure logs and raises, advising to check ffmpeg_compile.log.
    """
    if not frames:
        return
    cmd, log_file_path = _compilation_ffmpeg_cmd_and_log_path(
        tmp_output_path, target_w, target_h
    )
    log_file = open(log_file_path, "w", encoding="utf-8")
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=log_file,
        )
    except FileNotFoundError:
        log_file.close()
        logger.error(
            "Compilation encode failed: ffmpeg not found. "
            "Compilation requires GPU encoding (h264_nvenc). No CPU fallback is provided. "
            "Ensure FFmpeg is installed and available on PATH with NVENC support."
        )
        raise RuntimeError(
            "ffmpeg not found; compilation encoding is GPU-only (h264_nvenc), no CPU fallback"
        ) from None
    assert proc.stdin is not None
    try:
        for arr in frames:
            proc.stdin.write(arr.tobytes())
        proc.stdin.close()
        proc.wait()
    except BrokenPipeError:
        logger.error(
            "FFmpeg closed stdin (broken pipe). Command: %s. Check %s for stderr.",
            " ".join(cmd),
            log_file_path,
        )
        proc.wait()
        raise RuntimeError(
            f"FFmpeg broke pipe during encode; check {log_file_path!r} for details"
        ) from None
    except Exception as e:
        logger.error(
            "Compilation encode failed while writing frames: %s. Command: %s. Check %s for stderr.",
            e,
            " ".join(cmd),
            log_file_path,
        )
        proc.wait()
        raise
    finally:
        log_file.close()
    if proc.returncode != 0:
        logger.error(
            "Compilation encode failed: FFmpeg exited with code %s. Command: %s. Check %s for full stderr.",
            proc.returncode,
            " ".join(cmd),
            log_file_path,
        )
        raise RuntimeError(
            f"FFmpeg h264_nvenc encode failed (exit {proc.returncode}); "
            f"check {log_file_path!r} for details"
        )


def _resolve_clip_path(ce_dir: str, camera: str, resolve_clip_in_folder: object) -> str:
    """Resolve clip path for a camera under ce_dir; raise FileNotFoundError if missing."""
    cam_dir = os.path.join(ce_dir, camera)
    clip_name = resolve_clip_in_folder(cam_dir) if callable(resolve_clip_in_folder) else None
    if not clip_name:
        clip_name = f"{camera}.mp4"
    # Coerce to str so join() gets StrPath (resolve_clip_in_folder is typed object for test mocks).
    clip_path = os.path.join(cam_dir, str(clip_name))
    if not os.path.isfile(clip_path):
        raise FileNotFoundError(f"Clip not found: {clip_path}")
    return clip_path


def _run_pynv_compilation(
    slices: list[dict],
    ce_dir: str,
    tmp_output_path: str,
    target_w: int,
    target_h: int,
    resolve_clip_in_folder: object,
    cuda_device_index: int = 0,
) -> None:
    """
    Decode each slice with PyNvVideoCodec (gpu_decoder); crop with smooth panning in tensor space.
    Uses PTS-based frame selection (get_index_from_time_in_seconds) to reduce variable frame rate jitter.
    Frames are streamed directly to FFmpeg stdin (no in-memory frame list) to avoid RAM spikes.
    Encode via FFmpeg h264_nvenc only (GPU); no CPU fallback. Output 20fps, no audio.
    """
    import torch

    if not slices:
        return

    cmd, log_file_path = _compilation_ffmpeg_cmd_and_log_path(
        tmp_output_path, target_w, target_h
    )
    log_file = open(log_file_path, "w", encoding="utf-8")
    proc = None
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=log_file,
        )
    except FileNotFoundError:
        log_file.close()
        logger.error(
            "Compilation encode failed: ffmpeg not found. "
            "Compilation requires GPU encoding (h264_nvenc). No CPU fallback is provided. "
            "Ensure FFmpeg is installed and available on PATH with NVENC support."
        )
        raise RuntimeError(
            "ffmpeg not found; compilation encoding is GPU-only (h264_nvenc), no CPU fallback"
        ) from None
    assert proc is not None and proc.stdin is not None

    logged_cameras: set[str] = set()
    logged_stutter_cameras: set[str] = set()  # One INFO per camera per event for stutter/missing frames.
    slices_per_cam = Counter(sl["camera"] for sl in slices)
    missing_crop_by_cam: dict[str, list[tuple[float, float]]] = defaultdict(list)
    try:
        for slice_idx, sl in enumerate(slices):
            cam = sl["camera"]
            clip_path = _resolve_clip_path(ce_dir, cam, resolve_clip_in_folder)
            t0 = sl["start_sec"]
            t1 = sl["end_sec"]
            duration = t1 - t0
            n_frames = max(1, round(duration * float(COMPILATION_OUTPUT_FPS)))
            if "crop_start" not in sl or "crop_end" not in sl:
                missing_crop_by_cam[cam].append((t0, t1))
            crop_start = sl.get("crop_start")
            crop_end = sl.get("crop_end")
            output_times = [t0 + i / float(COMPILATION_OUTPUT_FPS) for i in range(n_frames)]

            # ffprobe outside GPU_LOCK so the GPU is not held during subprocess I/O (Finding 4.1).
            slice_meta = _get_video_metadata(clip_path)
            fallback_fps = (slice_meta[2] if slice_meta and slice_meta[2] > 0 else 30.0)
            fallback_duration = slice_meta[3] if slice_meta else duration

            batch = None  # So slice finally can always try to del batch after chunk loop or on error.
            try:
                with GPU_LOCK:
                    with create_decoder(clip_path, gpu_id=cuda_device_index) as ctx:
                        frame_count = len(ctx)
                        if frame_count <= 0:
                            fps = fallback_fps
                            frame_count = max(1, int(fallback_duration * fps))
                        # PTS-based frame indices to reduce jitter (decoder time→index mapping).
                        src_indices = [
                            min(max(0, ctx.get_index_from_time_in_seconds(t)), frame_count - 1)
                            for t in output_times
                        ]
                        if not src_indices:
                            continue
                        ih, iw = None, None  # Set from first chunk batch shape; crop params computed once.
                        for chunk_start in range(0, len(src_indices), BATCH_SIZE):
                            chunk_indices = src_indices[chunk_start : chunk_start + BATCH_SIZE]
                            try:
                                batch = ctx.get_frames(chunk_indices)
                            except Exception as e:
                                logger.error(
                                    "%s (decoder get_frames failed). path=%s chunk_indices=%s error=%s",
                                    NVDEC_INIT_FAILURE_PREFIX,
                                    clip_path,
                                    chunk_indices,
                                    e,
                                    exc_info=True,
                                )
                                logger.warning(
                                    "Compilation: decoder get_frames failed for slice %s (%s): %s",
                                    slice_idx,
                                    clip_path,
                                    e,
                                )
                                if torch.cuda.is_available():
                                    try:
                                        logger.debug("VRAM summary: %s", torch.cuda.memory_summary())
                                    except Exception:
                                        pass
                                break
                            if batch.shape[0] == 0:
                                logger.error(
                                    "Compilation: decoder returned 0 frames for chunk (slice %s). Skipping chunk. path=%s",
                                    slice_idx,
                                    clip_path,
                                )
                                del batch
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                continue
                            # First chunk only: set decoder dimensions and compute crop params once per slice.
                            if ih is None:
                                _, _, ih, iw = batch.shape
                                sw = sl.get("native_width") or iw
                                sh = sl.get("native_height") or ih
                                scale_x = (iw / sw) if sw > 0 else 1.0
                                scale_y = (ih / sh) if sh > 0 else 1.0
                                if crop_start and crop_end:
                                    xs, ys, w, h = crop_start
                                    xe, ye, we, he = crop_end
                                    w_d = max(1, min(iw, int(w * scale_x)))
                                    h_d = max(1, min(ih, int(h * scale_y)))
                                    we_d = max(1, min(iw, int(we * scale_x)))
                                    he_d = max(1, min(ih, int(he * scale_y)))
                                    xs_d = xs * scale_x
                                    ys_d = ys * scale_y
                                    xe_d = xe * scale_x
                                    ye_d = ye * scale_y
                                    start_cx = xs_d + w_d / 2.0
                                    start_cy = ys_d + h_d / 2.0
                                    end_cx = xe_d + we_d / 2.0
                                    end_cy = ye_d + he_d / 2.0
                                else:
                                    start_cx = iw / 2.0
                                    start_cy = ih / 2.0
                                    end_cx = iw / 2.0
                                    end_cy = ih / 2.0
                                # Static-frame: decoder returned same index for all frames in slice (first chunk only).
                                if len(src_indices) > 1 and len(set(src_indices)) == 1:
                                    logger.debug(
                                        "Compilation static frame: decoder returned same frame index %s for all %s frames for camera=%s slice [%.2f, %.2f]; check decoder/time mapping.",
                                        src_indices[0], n_frames, cam, t0, t1,
                                    )
                                    if cam not in logged_stutter_cameras:
                                        logger.info(
                                            "Possible stutter or missing frames from %s, check original file for confirmation. path=%s",
                                            cam, clip_path,
                                        )
                                        logged_stutter_cameras.add(cam)
                                if cam not in logged_cameras:
                                    logger.debug(
                                        "Compilation camera=%s: frame %sx%s, crop center (%.0f,%.0f)->(%.0f,%.0f), target %sx%s, n_slices=%s, n_frames=%s",
                                        cam, iw, ih, start_cx, start_cy, end_cx, end_cy, target_w, target_h,
                                        slices_per_cam.get(cam, 0), n_frames,
                                    )
                                    logged_cameras.add(cam)
                            # Every chunk: fewer frames than requested — repeat last frame of chunk to keep sync.
                            if batch.shape[0] < len(chunk_indices):
                                logger.debug(
                                    "Compilation: decoder returned fewer frames than requested for chunk camera=%s slice [%.2f, %.2f] (%s of %s). Repeating last frame of chunk.",
                                    cam, t0, t1, batch.shape[0], len(chunk_indices),
                                )
                                if cam not in logged_stutter_cameras:
                                    logger.info(
                                        "Possible stutter or missing frames from %s, check original file for confirmation. path=%s",
                                        cam, clip_path,
                                    )
                                    logged_stutter_cameras.add(cam)
                            for j in range(len(chunk_indices)):
                                safe_j = min(j, batch.shape[0] - 1)
                                frame = batch[safe_j : safe_j + 1]
                                i = chunk_start + j
                                t = output_times[i]
                                t_progress = (t - t0) / duration if duration > 1e-6 else 0.0
                                current_cx = start_cx + t_progress * (end_cx - start_cx)
                                current_cy = start_cy + t_progress * (end_cy - start_cy)
                                if crop_start and crop_end:
                                    current_w_d = max(2, int(w_d + t_progress * (we_d - w_d)) & ~1)
                                    current_h_d = max(2, int(h_d + t_progress * (he_d - h_d)) & ~1)
                                    current_w_d = min(iw, max(1, current_w_d))
                                    current_h_d = min(ih, max(1, current_h_d))
                                else:
                                    current_w_d = min(iw, target_w)
                                    current_h_d = min(ih, target_h)
                                current_cx_int = int(current_cx)
                                current_cy_int = int(current_cy)
                                cropped = crop_utils.crop_around_center_to_size(
                                    frame, current_cx_int, current_cy_int,
                                    current_w_d, current_h_d,
                                    target_w, target_h,
                                )
                                arr = cropped[0].permute(1, 2, 0).cpu().numpy()
                                proc.stdin.write(arr.tobytes())
                                proc.stdin.flush()
                            del batch
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
            except Exception as e:
                logger.error(
                    "%s (decoder failed for slice %s). path=%s error=%s",
                    NVDEC_INIT_FAILURE_PREFIX,
                    slice_idx,
                    clip_path,
                    e,
                    exc_info=True,
                )
                logger.warning("Skipping slice %s (%s): %s", slice_idx, clip_path, e)
            finally:
                try:
                    del batch
                except NameError:
                    pass
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        for cam, pairs in missing_crop_by_cam.items():
            n = len(pairs)
            first, last = pairs[0], pairs[-1]
            logger.error(
                "Compilation: slice missing crop_start/crop_end for camera=%s in %s slices (e.g. [%.2f, %.2f]–[%.2f, %.2f]); using fallback crop.",
                cam, n, first[0], first[1], last[0], last[1],
            )
    except BrokenPipeError:
        logger.error(
            "FFmpeg closed stdin (broken pipe). Command: %s. Check %s for stderr.",
            " ".join(cmd),
            log_file_path,
        )
        proc.wait()
        raise RuntimeError(
            f"FFmpeg broke pipe during encode; check {log_file_path!r} for details"
        ) from None
    except Exception as e:
        logger.error(
            "Compilation encode failed while writing frames: %s. Command: %s. Check %s for stderr.",
            e,
            " ".join(cmd),
            log_file_path,
        )
        proc.wait()
        raise
    finally:
        if proc.stdin is not None:
            proc.stdin.close()
        proc.wait()
        log_file.close()
    if proc.returncode != 0:
        logger.error(
            "Compilation encode failed: FFmpeg exited with code %s. Command: %s. Check %s for full stderr.",
            proc.returncode,
            " ".join(cmd),
            log_file_path,
        )
        raise RuntimeError(
            f"FFmpeg h264_nvenc encode failed (exit {proc.returncode}); "
            f"check {log_file_path!r} for details"
        )


def generate_compilation_video(
    slices: list[dict],
    ce_dir: str,
    output_path: str,
    target_w: int = 1440,
    target_h: int = 1080,
    crop_smooth_alpha: float = 0.0,
    config: dict | None = None,
    sidecars: dict[str, dict] | None = None,
) -> None:
    """
    Concatenates slices into a final 20fps cropped video. Decode and crop via PyNvVideoCodec (gpu_decoder) and PyTorch;
    encode via FFmpeg h264_nvenc only (GPU; no CPU fallback). Smooth panning uses t/duration interpolation.
    Optional EMA smoothing of crop centers. No audio.
    When sidecars is provided (e.g. from compile_ce_video), detection.json is not read from disk.
    """
    logger.info(f"Starting compilation of {len(slices)} slices to {output_path}")

    from frigate_buffer.services.query import resolve_clip_in_folder

    # Load sidecar once per camera: use provided dict or read from disk
    sidecar_cache: dict[str, dict] = {}
    sidecar_timestamps: dict[str, list[float]] = {}
    if sidecars is not None:
        sidecar_cache = sidecars
        for sl in slices:
            cam = sl["camera"]
            if cam in sidecar_timestamps:
                continue
            entries = sidecar_cache.get(cam, {}).get("entries") or []
            sidecar_timestamps[cam] = sorted(float(e.get("timestamp_sec") or 0) for e in entries)
    else:
        for sl in slices:
            cam = sl["camera"]
            if cam in sidecar_cache:
                continue
            cam_dir = os.path.join(ce_dir, cam)
            sidecar_path = os.path.join(cam_dir, "detection.json")
            if os.path.isfile(sidecar_path):
                try:
                    with open(sidecar_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        sidecar_cache[cam] = data if isinstance(data, dict) else {"entries": data or [], "native_width": 1920, "native_height": 1080}
                except Exception as e:
                    logger.error(
                        "Compilation fallback to center crop: error loading sidecar for camera=%s path=%s error=%s; output will be static.",
                        cam, sidecar_path, e,
                    )
                    sidecar_cache[cam] = {"entries": [], "native_width": 1920, "native_height": 1080}
            else:
                logger.error(
                    "Compilation fallback to center crop: sidecar missing for camera=%s path=%s; output will be static.",
                    cam, sidecar_path,
                )
                sidecar_cache[cam] = {"entries": [], "native_width": 1920, "native_height": 1080}
            entries = sidecar_cache[cam].get("entries") or []
            timestamps = sorted(float(e.get("timestamp_sec") or 0) for e in entries)
            sidecar_timestamps[cam] = timestamps

    no_entries_by_cam: dict[str, list[tuple[float, float]]] = defaultdict(list)
    no_detections_by_cam: dict[str, list[tuple[float, float]]] = defaultdict(list)

    for i, sl in enumerate(slices):
        cam = sl["camera"]
        sidecar_data = sidecar_cache.get(cam) or {}
        sw = int(sidecar_data.get("native_width", 1920) or 1920)
        sh = int(sidecar_data.get("native_height", 1080) or 1080)
        ts_sorted = sidecar_timestamps.get(cam)
        t0 = sl["start_sec"]
        t1 = sl["end_sec"]
        entries = sidecar_data.get("entries") or []
        # crop_start/crop_end are in native (sw, sh) space; stored on slice for decoder-space scaling in _run_pynv_compilation.
        if not entries:
            no_entries_by_cam[cam].append((t0, t1))
        else:
            entry0 = _nearest_entry_at_t(entries, t0, ts_sorted)
            entry1 = _nearest_entry_at_t(entries, t1, ts_sorted)
            dets0 = (entry0 or {}).get("detections") or []
            dets1 = (entry1 or {}).get("detections") or []
            if not dets0 or not dets1:
                no_detections_by_cam[cam].append((t0, t1))
        tracking_target_frame_percent = int(config.get("TRACKING_TARGET_FRAME_PERCENT", 40)) if config else 40
        sl["crop_start"] = calculate_crop_at_time(
            sidecar_data, t0, sw, sh, target_w, target_h, timestamps_sorted=ts_sorted,
            tracking_target_frame_percent=tracking_target_frame_percent,
        )
        # Last slice of a camera run: hold crop (no pan to switch-time position) to avoid panning away from the person at the cut.
        is_last_of_run = (i + 1 < len(slices)) and (slices[i + 1]["camera"] != cam)
        if is_last_of_run:
            sl["crop_end"] = sl["crop_start"]
        else:
            sl["crop_end"] = calculate_crop_at_time(
                sidecar_data, t1, sw, sh, target_w, target_h, timestamps_sorted=ts_sorted,
                tracking_target_frame_percent=tracking_target_frame_percent,
            )
        sl["native_width"] = sw
        sl["native_height"] = sh

    for cam, pairs in no_entries_by_cam.items():
        n = len(pairs)
        first, last = pairs[0], pairs[-1]
        logger.error(
            "Compilation: no sidecar entries for camera=%s in %s slices (e.g. [%.2f, %.2f]–[%.2f, %.2f]); using fallback crop.",
            cam, n, first[0], first[1], last[0], last[1],
        )
    for cam, pairs in no_detections_by_cam.items():
        n = len(pairs)
        logger.error(
            "Compilation: no detections at slice start/end for camera=%s in %s slices; using fallback crop (center or nearby detection within 5s).",
            cam, n,
        )

    zoom_smooth_alpha = float(config.get("COMPILATION_ZOOM_SMOOTH_EMA_ALPHA", 0.25)) if config else 0.25
    if zoom_smooth_alpha > 0:
        smooth_zoom_ema(slices, zoom_smooth_alpha, target_w, target_h)

    if crop_smooth_alpha > 0:
        smooth_crop_centers_ema(slices, crop_smooth_alpha)

    cuda_device_index = int(config.get("CUDA_DEVICE_INDEX", 0)) if config else 0
    temp_path = output_path.replace(".mp4", "_temp.mp4")
    if not temp_path.endswith(".mp4"):
        temp_path = temp_path + ".mp4"
    try:
        _run_pynv_compilation(
            slices=slices,
            ce_dir=ce_dir,
            tmp_output_path=temp_path,
            target_w=target_w,
            target_h=target_h,
            resolve_clip_in_folder=resolve_clip_in_folder,
            cuda_device_index=cuda_device_index,
        )
        if os.path.isfile(temp_path) and os.path.getsize(temp_path) > 0:
            os.rename(temp_path, output_path)
            logger.info(f"Successfully compiled summary video to {output_path}")
        else:
            raise FileNotFoundError(
                f"Compiler finished but tmp result file was empty or missing: {temp_path}"
            )
    except Exception as e:
        log_path = os.path.join(ce_dir, "compilation_error.log")
        try:
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(str(e))
        except OSError:
            pass
        logger.error(f"Compilation failed: {e}")
        raise

def compile_ce_video(ce_dir: str, global_end: float, config: dict, primary_camera: str | None = None) -> str | None:
    """
    High-level orchestrator for video compilation.
    Scans the CE directory, parses detection sidecars, uses timeline_ema to establish segment assignments, 
    and then generates the hardware-accelerated summary video.
    Returns the path to the compiled video if successful, None otherwise.
    """
    logger.info(f"Starting compilation process for {ce_dir}")
    
    # 1. Gather sidecars once (full structure: entries, native_width, native_height) for timeline and compilation
    sidecar_cache: dict[str, dict] = {}
    from frigate_buffer.services.query import resolve_clip_in_folder

    cameras = []
    try:
        with os.scandir(ce_dir) as it:
            for entry in it:
                if entry.is_dir() and not entry.name.startswith("."):
                    cam = entry.name
                    clip = resolve_clip_in_folder(entry.path)
                    if clip:
                        cameras.append(cam)
    except OSError:
        pass

    if not cameras:
        logger.warning(f"No cameras found with clips in {ce_dir}")
        return None

    for cam in cameras:
        path = os.path.join(ce_dir, cam, "detection.json")
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        entries = data.get("entries") or []
                        nw = int(data.get("native_width", 0) or 0) or 1920
                        nh = int(data.get("native_height", 0) or 0) or 1080
                        sidecar_cache[cam] = {"entries": entries, "native_width": nw, "native_height": nh}
                    else:
                        sidecar_cache[cam] = {"entries": data or [], "native_width": 1920, "native_height": 1080}
            except Exception as e:
                logger.error(f"Failed to load sidecar for {cam}: {e}")
                sidecar_cache[cam] = {"entries": [], "native_width": 1920, "native_height": 1080}
        else:
            sidecar_cache[cam] = {"entries": [], "native_width": 1920, "native_height": 1080}

    if not sidecar_cache:
        logger.warning(f"No sidecars available in {ce_dir} for compilation.")
        return None

    # Use actual clip/sidecar duration when longer than requested window, so the summary
    # covers full content on disk (avoids truncation when export window is shorter than clip).
    actual_duration_sec = 0.0
    for cam in sidecar_cache:
        entries = sidecar_cache[cam].get("entries") or []
        if entries:
            last_ts = float(entries[-1].get("timestamp_sec") or 0)
            actual_duration_sec = max(actual_duration_sec, last_ts)
    if actual_duration_sec > global_end:
        logger.info(
            "Compilation: extending timeline from %.1fs to %.1fs (sidecar content longer than requested window).",
            global_end,
            actual_duration_sec,
        )
    global_end = max(global_end, actual_duration_sec)

    native_sizes = {cam: (sidecar_cache[cam]["native_width"], sidecar_cache[cam]["native_height"]) for cam in sidecar_cache}

    # Helper to calculate area of person targets at t_sec
    def _person_area_at_time(cam_name: str, t_sec: float) -> float:
        entries = (sidecar_cache.get(cam_name) or {}).get("entries") or []
        if not entries:
            return 0.0
        # Find nearest
        nearest = min(entries, key=lambda e: abs((e.get("timestamp_sec") or 0) - t_sec))
        area: float = 0.0
        for d in nearest.get("detections") or []:
            if (d.get("label") or "").lower() in ("person", "people", "pedestrian"):
                area += float(d.get("area") or 0)
        return area
        
    # 2. Compute timeline using same config as frame timeline (AI prompt)
    step_sec = float(config.get("MAX_MULTI_CAM_FRAMES_SEC", 2))
    if step_sec <= 0:
        step_sec = 2.0
    max_frames_min = int(config.get("MAX_MULTI_CAM_FRAMES_MIN", 45))
    multiplier = float(config.get("CAMERA_TIMELINE_ANALYSIS_MULTIPLIER", 2.0))

    dense_times = timeline_ema.build_dense_times(step_sec, max_frames_min, multiplier, global_end)

    assignments = timeline_ema.build_phase1_assignments(
        dense_times,
        cameras,
        _person_area_at_time,
        native_sizes,
        ema_alpha=float(config.get("CAMERA_TIMELINE_EMA_ALPHA", 0.4)),
        primary_bias_multiplier=float(config.get("CAMERA_TIMELINE_PRIMARY_BIAS_MULTIPLIER", 1.2)),
        primary_camera=primary_camera,
        hysteresis_margin=float(config.get("CAMERA_SWITCH_HYSTERESIS_MARGIN", 1.15)),
        min_segment_frames=int(config.get("CAMERA_SWITCH_MIN_SEGMENT_FRAMES", 5)),
    )

    if not assignments:
        logger.warning("No camera assignments could be generated for compilation.")
        return None

    logger.debug(f"Generated {len(assignments)} timeline points via EMA.")

    # 3. One slice per assignment; then trim to action window (first/last detection ± pre/post roll)
    slices = assignments_to_slices(assignments, global_end)
    sidecars_entries = {cam: sidecar_cache[cam]["entries"] for cam in sidecar_cache}
    slices = _trim_slices_to_action_window(slices, sidecars_entries, global_end)
    if not slices:
        logger.warning("No slices remain after trimming to action window; skipping compilation.")
        return None

    out_name = os.path.basename(os.path.abspath(ce_dir)) + "_summary.mp4"
    output_path = os.path.join(ce_dir, out_name)
    crop_smooth_alpha = float(config.get("COMPILATION_CROP_SMOOTH_EMA_ALPHA", 0.0))

    try:
        generate_compilation_video(
            slices,
            ce_dir,
            output_path,
            target_w=int(config.get("SUMMARY_TARGET_WIDTH", 1440)),
            target_h=int(config.get("SUMMARY_TARGET_HEIGHT", 1080)),
            crop_smooth_alpha=crop_smooth_alpha,
            config=config,
            sidecars=sidecar_cache,
        )
        return output_path
    except Exception as e:
        logger.error(f"Failed to generate summary video: {e}")
        return None
