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

from frigate_buffer.services import timeline_ema

logger = logging.getLogger("frigate-buffer")


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


def calculate_crop_at_time(
    sidecar_data: dict,
    t_sec: float,
    source_width: int,
    source_height: int,
    target_w: int = 1440,
    target_h: int = 1080,
    *,
    timestamps_sorted: list[float] | None = None,
) -> tuple[int, int, int, int]:
    """
    Compute crop (x, y, w, h) at a single timestamp from the nearest sidecar entry.
    Uses area-weighted center of detections. Returns clamped rect.
    If timestamps_sorted is provided (same order as sidecar entries), lookup is O(log n).
    Time O(log(entries)) with bisect, O(entries) without; space O(1).
    """
    entries = sidecar_data.get("entries") or []
    if not entries:
        avg_cx = source_width / 2.0
        avg_cy = source_height / 2.0
    else:
        entry = _nearest_entry_at_t(entries, t_sec, timestamps_sorted)
        if not entry:
            avg_cx = source_width / 2.0
            avg_cy = source_height / 2.0
        else:
            weighted_cx_sum: float = 0.0
            weighted_cy_sum: float = 0.0
            total_area: float = 0.0
            for det in entry.get("detections") or []:
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

    if source_width < target_w or source_height < target_h:
        scale = min(source_width / target_w, source_height / target_h)
        target_w = int(target_w * scale)
        target_h = int(target_h * scale)

    x = int(avg_cx - target_w / 2.0)
    y = int(avg_cy - target_h / 2.0)
    x = max(0, min(source_width - target_w, x))
    y = max(0, min(source_height - target_h, y))
    return (x, y, target_w, target_h)


def smooth_crop_centers_ema(
    slices: list[dict], alpha: float
) -> None:
    """
    Smooth crop center trajectory in-place with single-pass EMA to reduce detection jitter.
    Slices must have "crop_start" and "crop_end" as (x, y, w, h). Updates those with
    smoothed (x, y) derived from EMA of center (cx, cy) = (x + w/2, y + h/2).
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

def generate_compilation_video(
    slices: list[dict],
    ce_dir: str,
    output_path: str,
    target_w: int = 1440,
    target_h: int = 1080,
    crop_smooth_alpha: float = 0.0,
) -> None:
    """
    Concatenates slices into a final 20fps cropped video. One -i input per slice (no deduplication)
    to avoid FFmpeg "Output pad already connected" crashes. Crop uses 0-based t (after setpts)
    for smooth panning: x/y interpolated as t/duration. Optional EMA smoothing of crop centers.
    """
    logger.info(f"Starting compilation of {len(slices)} slices to {output_path}")

    from frigate_buffer.services.query import resolve_clip_in_folder

    # Load sidecar once per camera (batch I/O)
    sidecar_cache: dict[str, dict] = {}
    sidecar_timestamps: dict[str, list[float]] = {}
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
                logger.error(f"Error loading sidecar for {cam}: {e}")
                sidecar_cache[cam] = {"entries": [], "native_width": 1920, "native_height": 1080}
        else:
            sidecar_cache[cam] = {"entries": [], "native_width": 1920, "native_height": 1080}
        entries = sidecar_cache[cam].get("entries") or []
        timestamps = sorted(float(e.get("timestamp_sec") or 0) for e in entries)
        sidecar_timestamps[cam] = timestamps

    inputs: list[str] = []
    # One -i per slice (Constraint 1: do not deduplicate by camera)
    for i, sl in enumerate(slices):
        cam = sl["camera"]
        cam_dir = os.path.join(ce_dir, cam)
        clip_name = resolve_clip_in_folder(cam_dir)
        if not clip_name:
            clip_name = f"{cam}.mp4"
        clip_path = os.path.join(cam_dir, clip_name)
        inputs.extend(["-i", clip_path])

    for i, sl in enumerate(slices):
        cam = sl["camera"]
        sidecar_data = sidecar_cache.get(cam) or {}
        sw = int(sidecar_data.get("native_width", 1920) or 1920)
        sh = int(sidecar_data.get("native_height", 1080) or 1080)
        ts_sorted = sidecar_timestamps.get(cam)
        t0 = sl["start_sec"]
        t1 = sl["end_sec"]
        sl["crop_start"] = calculate_crop_at_time(
            sidecar_data, t0, sw, sh, target_w, target_h, timestamps_sorted=ts_sorted
        )
        sl["crop_end"] = calculate_crop_at_time(
            sidecar_data, t1, sw, sh, target_w, target_h, timestamps_sorted=ts_sorted
        )

    if crop_smooth_alpha > 0:
        smooth_crop_centers_ema(slices, crop_smooth_alpha)

    tmp_output_path = output_path + ".tmp"
    filter_parts: list[str] = []
    for i, sl in enumerate(slices):
        t0 = sl["start_sec"]
        t1 = sl["end_sec"]
        duration = t1 - t0
        xs, ys, w, h = sl["crop_start"]
        xe, ye, we, he = sl["crop_end"]
        # Constraint 2: after setpts=PTS-STARTPTS, t is 0-based; use t/duration
        if duration <= 1e-6:
            fg = f"[{i}:v]trim=start={t0}:end={t1},setpts=PTS-STARTPTS,fps=20,crop={w}:{h}:{xs}:{ys},format=yuv420p[v{i}]"
        else:
            x_expr = f"min(max(0,{xs}+({xe}-{xs})*(t/{duration})),iw-{w})"
            y_expr = f"min(max(0,{ys}+({ye}-{ys})*(t/{duration})),ih-{h})"
            fg = f"[{i}:v]trim=start={t0}:end={t1},setpts=PTS-STARTPTS,fps=20,crop={w}:{h}:{x_expr}:{y_expr},format=yuv420p[v{i}]"
        filter_parts.append(fg)
    concat_inputs = "".join(f"[v{i}]" for i in range(len(slices)))
    filter_parts.append(concat_inputs + f"concat=n={len(slices)}:v=1:a=0[outv]")
    fg_str = ";".join(filter_parts)
    logger.debug(f"Compilation FFmpeg filter_complex:\n{fg_str}")

    cmd = ["ffmpeg", "-y"]
    cmd.extend(inputs)
    cmd.extend([
        "-filter_complex", fg_str,
        "-map", "[outv]",
        "-an",
        "-c:v", "h264_nvenc",
        "-pix_fmt", "yuv420p",
        "-f", "mp4",
        tmp_output_path,
    ])
    logger.debug(f"Compilation raw FFmpeg command: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if os.path.isfile(tmp_output_path) and os.path.getsize(tmp_output_path) > 0:
            os.rename(tmp_output_path, output_path)
            logger.info(f"Compilation finished successfully. Output saved to: {output_path}")
        else:
            raise FileNotFoundError(
                f"Compiler terminated effectively, but tmp result file was empty or missing: {tmp_output_path}"
            )
    except subprocess.CalledProcessError as e:
        ffmpeg_log_path = os.path.join(ce_dir, "ffmpeg_compilation_error.log")
        with open(ffmpeg_log_path, "w", encoding="utf-8") as f:
            f.write(e.stderr or "")
        logger.error(
            f"Compilation FFmpeg failed! Check {ffmpeg_log_path} for raw output. Process error:\n{e.stderr}"
        )
        raise
    except Exception as general_error:
        logger.error(f"Compilation failed unexpectedly: {general_error}")
        raise

def compile_ce_video(ce_dir: str, global_end: float, config: dict, primary_camera: str | None = None) -> str | None:
    """
    High-level orchestrator for video compilation.
    Scans the CE directory, parses detection sidecars, uses timeline_ema to establish segment assignments, 
    and then generates the hardware-accelerated summary video.
    Returns the path to the compiled video if successful, None otherwise.
    """
    logger.info(f"Starting compilation process for {ce_dir}")
    
    # 1. Gather sidecars and native sizes
    sidecars = {}
    native_sizes = {}
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
                        sidecars[cam] = data.get("entries") or []
                        nw = int(data.get("native_width", 0) or 0)
                        nh = int(data.get("native_height", 0) or 0)
                        native_sizes[cam] = (nw, nh)
                    elif isinstance(data, list):
                        sidecars[cam] = data
                        native_sizes[cam] = (0, 0)
            except Exception as e:
                logger.error(f"Failed to load sidecar for {cam}: {e}")
                
    if not sidecars:
        logger.warning(f"No sidecars available in {ce_dir} for compilation.")
        return None
        
    # Helper to calculate area of person targets at t_sec
    def _person_area_at_time(cam_name: str, t_sec: float) -> float:
        entries = sidecars.get(cam_name) or []
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

    # 3. One slice per assignment; generate with follow-track crop and smooth panning
    slices = assignments_to_slices(assignments, global_end)
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
        )
        return output_path
    except Exception as e:
        logger.error(f"Failed to generate summary video: {e}")
        return None
