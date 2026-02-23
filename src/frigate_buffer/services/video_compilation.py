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

from frigate_buffer.constants import NVDEC_INIT_FAILURE_PREFIX
from frigate_buffer.services import timeline_ema
from frigate_buffer.services.gpu_decoder import create_decoder
from frigate_buffer.services.video import GPU_LOCK, _get_video_metadata

logger = logging.getLogger("frigate-buffer")

# Output frame rate for compilation (smooth panning samples at this rate).
COMPILATION_OUTPUT_FPS = 20


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


def _encode_frames_via_ffmpeg(
    frames: list,
    target_w: int,
    target_h: int,
    tmp_output_path: str,
) -> None:
    """
    Encode a list of RGB frames to MP4 using FFmpeg with h264_nvenc only (GPU).

    frames: list of numpy arrays (H, W, 3) uint8 RGB with H=target_h, W=target_w.
    No CPU fallback; on failure logs full context and raises.
    """
    if not frames:
        return
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
        tmp_output_path,
    ]
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError:
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
    except BrokenPipeError:
        stderr = (
            (proc.stderr.read() or b"").decode("utf-8", errors="replace")
            if proc.stderr else ""
        )
        logger.error(
            "FFmpeg closed stdin (broken pipe). stderr: %s. Command: %s. "
            "Compilation encoding is GPU-only (h264_nvenc). No CPU fallback is provided.",
            stderr,
            " ".join(cmd),
        )
        proc.wait()
        raise RuntimeError(
            f"FFmpeg broke pipe during encode; stderr: {stderr!r}"
        ) from None
    except Exception as e:
        stderr = (
            (proc.stderr.read() or b"").decode("utf-8", errors="replace")
            if proc.stderr else ""
        )
        logger.error(
            "Compilation encode failed while writing frames: %s. "
            "FFmpeg command: %s. stderr: %s. "
            "Compilation encoding is GPU-only (h264_nvenc). No CPU fallback is provided.",
            e,
            " ".join(cmd),
            stderr,
        )
        proc.wait()
        raise
    stdout, stderr = proc.communicate()
    stderr_str = (stderr or b"").decode("utf-8", errors="replace")
    if proc.returncode != 0:
        logger.error(
            "Compilation encode failed: FFmpeg exited with code %s. "
            "Command: %s. stderr: %s. "
            "Compilation encoding is GPU-only (h264_nvenc). No CPU fallback is provided. "
            "Ensure an NVENC-capable GPU and drivers/FFmpeg with h264_nvenc support.",
            proc.returncode,
            " ".join(cmd),
            stderr_str,
        )
        raise RuntimeError(
            f"FFmpeg h264_nvenc encode failed (exit {proc.returncode}); "
            "compilation is GPU-only, no CPU fallback"
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
    Encode via FFmpeg h264_nvenc only (GPU); no CPU fallback. Output 20fps, no audio.
    """
    import torch

    if not slices:
        return

    frames_list: list = []  # list of (target_h, target_w, 3) uint8 RGB numpy arrays

    for slice_idx, sl in enumerate(slices):
        cam = sl["camera"]
        clip_path = _resolve_clip_path(ce_dir, cam, resolve_clip_in_folder)
        t0 = sl["start_sec"]
        t1 = sl["end_sec"]
        duration = t1 - t0
        n_frames = max(1, round(duration * float(COMPILATION_OUTPUT_FPS)))
        xs, ys, w, h = sl["crop_start"]
        xe, ye, we, he = sl["crop_end"]
        output_times = [t0 + i / float(COMPILATION_OUTPUT_FPS) for i in range(n_frames)]

        try:
            with GPU_LOCK:
                with create_decoder(clip_path, gpu_id=cuda_device_index) as ctx:
                    frame_count = len(ctx)
                    if frame_count <= 0:
                        meta = _get_video_metadata(clip_path)
                        fps = (meta[2] if meta and meta[2] > 0 else 30.0)
                        frame_count = max(1, int((meta[3] if meta else duration) * fps))
                    # PTS-based frame indices to reduce jitter (decoder time→index mapping).
                    src_indices = [
                        min(max(0, ctx.get_index_from_time_in_seconds(t)), frame_count - 1)
                        for t in output_times
                    ]
                    if not src_indices:
                        continue
                    batch = ctx.get_frames(src_indices)
            # Decoder context closed; batch is cloned so we can use it.
            _, _, ih, iw = batch.shape
            for i in range(n_frames):
                frame = batch[i : i + 1]
                t = output_times[i]
                t_rel = t - t0
                if duration <= 1e-6:
                    x, y = xs, ys
                else:
                    x = xs + (xe - xs) * (t_rel / duration)
                    y = ys + (ye - ys) * (t_rel / duration)
                x = int(min(max(0, x), iw - w))
                y = int(min(max(0, y), ih - h))
                cropped = frame[:, :, y : y + h, x : x + w]
                if cropped.shape[2] != target_h or cropped.shape[3] != target_w:
                    cropped = torch.nn.functional.interpolate(
                        cropped.float(),
                        size=(target_h, target_w),
                        mode="bilinear",
                        align_corners=False,
                    ).byte()
                arr = cropped[0].permute(1, 2, 0).cpu().numpy()
                frames_list.append(arr)
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

    if frames_list:
        _encode_frames_via_ffmpeg(frames_list, target_w, target_h, tmp_output_path)


def generate_compilation_video(
    slices: list[dict],
    ce_dir: str,
    output_path: str,
    target_w: int = 1440,
    target_h: int = 1080,
    crop_smooth_alpha: float = 0.0,
    config: dict | None = None,
) -> None:
    """
    Concatenates slices into a final 20fps cropped video. Decode and crop via PyNvVideoCodec (gpu_decoder) and PyTorch;
    encode via FFmpeg h264_nvenc only (GPU; no CPU fallback). Smooth panning uses t/duration interpolation.
    Optional EMA smoothing of crop centers. No audio.
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
        # Last slice of a camera run: hold crop (no pan to switch-time position) to avoid panning away from the person at the cut.
        is_last_of_run = (i + 1 < len(slices)) and (slices[i + 1]["camera"] != cam)
        if is_last_of_run:
            sl["crop_end"] = sl["crop_start"]
        else:
            sl["crop_end"] = calculate_crop_at_time(
                sidecar_data, t1, sw, sh, target_w, target_h, timestamps_sorted=ts_sorted
            )

    if crop_smooth_alpha > 0:
        smooth_crop_centers_ema(slices, crop_smooth_alpha)

    cuda_device_index = int(config.get("CUDA_DEVICE_INDEX", 0)) if config else 0
    temp_path = output_path.replace(".mp4", ".tmp.mp4")
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
            logger.info(f"Compilation finished successfully. Output saved to: {output_path}")
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
            config=config,
        )
        return output_path
    except Exception as e:
        logger.error(f"Failed to generate summary video: {e}")
        return None
