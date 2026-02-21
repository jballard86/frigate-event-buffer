"""
Target-centric multi-clip frame extraction for consolidated events.

Requires detection sidecars (detection.json per camera) from generate_detection_sidecar.
Reads sidecars and picks the camera with largest person area per time step.
If any camera lacks a sidecar, returns [] (no on-frame detector fallback). No Frigate metadata.

Uses ffmpegcv for decode and a sequential-read, time-sampled strategy (no seek),
since ffmpegcv readers do not support frame-index seek.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

logger = logging.getLogger("frigate-buffer")

try:
    import cv2
    import ffmpegcv
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

try:
    from frigate_buffer.services import crop_utils as _crop_utils
    _CROP_AVAILABLE = True
except ImportError:
    _CROP_AVAILABLE = False

try:
    from frigate_buffer.models import ExtractedFrame
except ImportError:
    ExtractedFrame = None  # type: ignore[misc, assignment]

try:
    from frigate_buffer.services import timeline_ema as _timeline_ema
    _TIMELINE_EMA_AVAILABLE = True
except ImportError:
    _TIMELINE_EMA_AVAILABLE = False


# Preferred label for target-centric selection (person has highest priority)
PREFERRED_LABELS = ("person", "people", "pedestrian")

# Sidecar filename written by generate_detection_sidecar (video.py)
DETECTION_SIDECAR_FILENAME = "detection.json"


def _is_gpu_not_configured(exc: BaseException) -> bool:
    """
    True if the exception indicates GPU was never configured (no GPU, ffmpeg without NVENC),
    vs GPU decode was attempted but failed at runtime.
    """
    msg = str(exc).lower()
    return (
        "no nvidia gpu" in msg
        or "no gpu found" in msg
        or "not compiled with nvenc" in msg
        or "not compiled with nvdec" in msg
    )


def _person_area_from_detections(detections: list[dict[str, Any]]) -> float:
    """Sum area of detections whose label is in PREFERRED_LABELS. Used for sidecar entries."""
    total = 0.0
    for d in detections or []:
        label = (d.get("label") or "").lower()
        if label in PREFERRED_LABELS:
            total += float(d.get("area") or 0)
    return total


def _load_sidecar_for_camera(camera_folder: str) -> tuple[list[dict[str, Any]], int, int] | None:
    """
    Load detection.json from camera folder.
    Returns (entries, native_width, native_height) or None. Supports both legacy list format
    and new dict format with "entries" and "native_width"/"native_height".
    """
    path = os.path.join(camera_folder, DETECTION_SIDECAR_FILENAME)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return (data, 0, 0)
        if isinstance(data, dict):
            entries = data.get("entries")
            if entries is None:
                return None
            nw = int(data.get("native_width", 0) or 0)
            nh = int(data.get("native_height", 0) or 0)
            return (entries, nw, nh)
        return None
    except (OSError, json.JSONDecodeError) as e:
        logger.debug("Could not load sidecar %s: %s", path, e)
        return None


def _nearest_sidecar_entry(
    sidecar_entries: list[dict[str, Any]], t_sec: float
) -> dict[str, Any] | None:
    """Return the sidecar entry with timestamp_sec closest to t_sec, or None if empty."""
    if not sidecar_entries:
        return None
    return min(sidecar_entries, key=lambda e: abs((e.get("timestamp_sec") or 0) - t_sec))


def _person_area_at_time(sidecar_entries: list[dict[str, Any]], t_sec: float) -> float:
    """Return person area at timestamp t_sec using nearest sidecar entry (by timestamp_sec).
    Empty sidecar or missing detections defaults to 0.0."""
    entry = _nearest_sidecar_entry(sidecar_entries, t_sec)
    if entry is None:
        return 0.0
    detections = entry.get("detections")
    if not detections:
        return 0.0
    return _person_area_from_detections(detections)


def _detection_timestamps_with_person(
    sidecars: dict[str, list[dict[str, Any]]],
    global_end: float,
) -> list[float]:
    """Build sorted list of timestamps (from all cameras) where person area > 0.
    Handles None or empty sidecar per camera safely."""
    timestamps: set[float] = set()
    for cam, entries in sidecars.items():
        for entry in entries or []:
            area = _person_area_from_detections(entry.get("detections") or [])
            if area > 0:
                t = float(entry.get("timestamp_sec") or 0)
                if 0 <= t <= global_end:
                    timestamps.add(t)
    return sorted(timestamps)


def _subsample_with_min_gap(timestamps: list[float], step_sec: float, max_count: int) -> list[float]:
    """Select timestamps with at least step_sec between them, up to max_count.
    Iterates in ascending order; selects when t >= last_selected + step_sec."""
    if not timestamps or step_sec <= 0 or max_count <= 0:
        return []
    selected: list[float] = []
    last = -step_sec if step_sec > 0 else -float("inf")
    for t in timestamps:
        if t >= last + step_sec and len(selected) < max_count:
            selected.append(t)
            last = t
    return selected


def _get_fps_and_duration(cap: Any, path: str) -> tuple[float, float, bool]:
    """
    Get (fps, duration_sec, used_read_to_eof) from a video capture.
    Prefer _get_fps_duration_from_path(path) when caps are not yet opened to avoid double-open.
    """
    fps = getattr(cap, "fps", None) or 1.0
    count: int = 0
    if hasattr(cap, "__len__"):
        try:
            count = int(len(cap))
        except (TypeError, ValueError):
            pass
    if count <= 0:
        count = int(getattr(cap, "count", 0) or 0)
    if count > 0 and fps > 0:
        return (fps, count / fps, False)
    # Try ffprobe before consuming stream
    path_meta = _get_fps_duration_from_path(path)
    if path_meta is not None:
        return (path_meta[0], path_meta[1], False)
    import subprocess
    try:
        out = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_entries",
                "format=duration", "-of",
                "default=noprint_wrappers=1:nokey=1", path
            ],
            capture_output=True,
            timeout=5,
            check=False,
        )
        if out.returncode == 0 and out.stdout:
            duration_str = out.stdout.decode("utf-8", errors="replace").strip()
            if duration_str and duration_str != "N/A":
                duration = float(duration_str)
                if duration > 0:
                    logger.debug("ffprobe successfully returned duration %.2fs for %s", duration, path)
                    return (fps, duration, False)
    except Exception as e:
        logger.debug("ffprobe duration check failed for %s: %s", path, e)

    # Unknown count: read until EOF to get frame count (consumes stream; caller must reopen).
    logger.info("Falling back to full sequential decode to calculate duration for %s", path)
    frame_idx = 0
    while True:
        ret, _ = cap.read()
        if not ret:
            break
        frame_idx += 1
    duration = (frame_idx / fps) if fps > 0 and frame_idx > 0 else 0.0
    return (fps, duration, True)


def _get_fps_duration_from_path(path: str) -> tuple[float, float] | None:
    """
    Get (fps, duration_sec) from clip path via a single ffprobe call.
    Avoids opening the video capture for metadata so caps can be opened once and not reopened.
    """
    try:
        from frigate_buffer.services.video import _get_video_metadata
        meta = _get_video_metadata(path)
        if meta is not None:
            _, _, fps, duration = meta
            if fps > 0 and duration >= 0:
                return (fps, duration)
    except Exception as e:
        logger.debug("_get_fps_duration_from_path failed for %s: %s", path, e)
    return None


def _advance_one_camera_to_T(
    cam: str,
    cap: Any,
    state_tuple: tuple[Any, float, Any, float],
    T: float,
    fps: float,
    frame_idx: int,
    duration_cam: float,
) -> tuple[tuple[Any, float, Any, float] | None, int | None]:
    """
    Advance one camera's reader until curr_t >= T.
    Returns (new_state, new_frame_idx). On read error returns (None, None) to signal drop.
    Used by parallel decode: one call per camera per sample time.
    """
    _errs = (ValueError, OSError, BrokenPipeError, RuntimeError)
    prev_f, prev_t, curr_f, curr_t = state_tuple
    if T >= duration_cam or duration_cam <= 0:
        return (state_tuple, frame_idx)
    try:
        while curr_t < T:
            ret, frame = cap.read()
            if not ret or frame is None:
                if curr_f is not None:
                    return ((curr_f, curr_t, curr_f, curr_t), frame_idx)
                return (state_tuple, frame_idx)
            frame_idx += 1
            new_t = frame_idx / fps if fps > 0 else curr_t
            state_tuple = (curr_f, curr_t, frame, new_t)
            prev_f, prev_t, curr_f, curr_t = state_tuple
        return (state_tuple, frame_idx)
    except _errs:
        return (None, None)


def extract_target_centric_frames(
    ce_folder_path: str,
    max_frames_sec: float,
    max_frames_min: int,
    *,
    crop_width: int = 0,
    crop_height: int = 0,
    tracking_target_frame_percent: int = 40,
    first_camera_bias: str | None = None,
    first_camera_bias_decay_seconds: float = 1.0,
    first_camera_bias_initial: float = 1.5,
    first_camera_bias_cap_seconds: float = 0.0,
    person_area_switch_threshold: int = 0,
    camera_switch_ratio: float = 1.2,
    camera_switch_bias: float = 1.2,
    camera_switch_min_hold_frames: int = 5,
    decode_second_camera_cpu_only: bool = False,
    log_callback: Callable[[str], None] | None = None,
    config: dict[str, Any] | None = None,
    # Trust-the-EMA pipeline (Phase 1 dense grid + EMA + hysteresis + merge)
    use_ema_pipeline: bool = False,
    camera_timeline_analysis_multiplier: float = 2.0,
    camera_timeline_ema_alpha: float = 0.4,
    camera_timeline_primary_bias_multiplier: float = 1.2,
    camera_switch_min_segment_frames: int = 5,
    camera_switch_hysteresis_margin: float = 1.15,
    camera_timeline_final_yolo_drop_no_person: bool = False,
) -> list[Any]:
    """
    Extract time-ordered, target-centric frames from all clips under a CE folder.

    Uses full clip per camera with a sequential-read, time-sampled strategy (no seek).
    Steps at max_frames_sec intervals; at each sample time picks the camera with
    largest person area from sidecars. Returns list of ExtractedFrame (frame, timestamp_sec, camera, metadata).
    When person area >= tracking_target_frame_percent of reference_area (min of target crop area and frame area),
    uses full-frame resize with letterbox instead of crop and sets metadata is_full_frame_resize=True.
    Caps at max_frames_min. Returns [] if any camera lacks detection.json.

    Optional config may provide LOG_EXTRACTION_PHASE_TIMING (bool) for phase elapsed-time logs,
    and MERGE_FRAME_TIMEOUT_SEC (int) for timeout when waiting for a camera frame in the merge step.

    camera_switch_bias applies only to non-initial (non-primary) cameras: when we have switched away
    from the first camera, the currently selected camera's area is multiplied by a stickiness factor
    (initial camera_switch_bias, clamped so 0 becomes 0.1) that decays using first_camera_bias_decay_seconds
    and first_camera_bias_cap_seconds, so non-initial cameras are slightly stickier and reduce flip-flop.

    camera_switch_min_hold_frames: after any switch, do not allow another switch for this many timeline
    samples unless the current camera has no person (area 0). Reduces rapid flip-flop; 0 disables.
    """
    if not _CV2_AVAILABLE or ExtractedFrame is None:
        logger.warning("cv2/ffmpegcv or ExtractedFrame not available, skipping multi-clip extraction")
        return []

    # Single scan: discover camera clips ce_folder/CameraName/*.mp4 (dynamic clip names)
    from frigate_buffer.services.query import resolve_clip_in_folder
    clip_paths: list[tuple[str, str]] = []  # (camera_name, path)
    try:
        with os.scandir(ce_folder_path) as it:
            for entry in it:
                if entry.is_dir() and not entry.name.startswith("."):
                    sub = entry.path
                    name = entry.name
                    clip_basename = resolve_clip_in_folder(sub)
                    if clip_basename:
                        clip_path = os.path.join(sub, clip_basename)
                        if os.path.isfile(clip_path):
                            clip_paths.append((name, clip_path))
    except OSError as e:
        logger.warning("Could not scan CE folder %s: %s", ce_folder_path, e)
        return []

    if len(clip_paths) < 1:
        logger.debug("No clips found in CE folder %s", ce_folder_path)
        return []

    if log_callback:
        cams_str = ", ".join(sorted(cam for cam, _ in clip_paths))
        log_callback(f"Found {len(clip_paths)} clip(s): {cams_str}.")

    step_sec = float(max_frames_sec) if max_frames_sec > 0 else 1.0
    collected: list[Any] = []
    path_per_cam = {cam: path for cam, path in clip_paths}
    target_crop_area = (crop_width * crop_height) if (crop_width > 0 and crop_height > 0) else 0

    # Load sidecars in parallel (one JSON read per camera).
    camera_folders = {cam: os.path.dirname(path) for cam, path in clip_paths}
    sidecars: dict[str, list[dict[str, Any]]] = {}
    native_size_per_cam: dict[str, tuple[int, int]] = {}
    all_have_sidecar = True
    max_workers = min(len(camera_folders), 8)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_cam = {
            executor.submit(_load_sidecar_for_camera, folder): cam
            for cam, folder in camera_folders.items()
        }
        for future in as_completed(future_to_cam):
            cam = future_to_cam[future]
            try:
                loaded = future.result()
                if loaded is not None:
                    entries, nw, nh = loaded
                    sidecars[cam] = entries
                    native_size_per_cam[cam] = (nw, nh)
                else:
                    all_have_sidecar = False
            except Exception as e:
                logger.debug("Sidecar load failed for %s: %s", cam, e)
                all_have_sidecar = False

    if log_callback:
        log_callback(f"Loaded sidecars for {len(sidecars)} camera(s).")

    _log_phase_timing = bool(config.get("LOG_EXTRACTION_PHASE_TIMING", False)) if config else False
    if not _log_phase_timing and logger.isEnabledFor(logging.DEBUG):
        _log_phase_timing = True

    _t0_open = time.monotonic() if _log_phase_timing else None
    # Pre-probe fps/duration from path (single ffprobe per clip) to avoid opening caps for metadata and double-open.
    durations: dict[str, float] = {}
    fps_per_cam: dict[str, float] = {}
    for cam, path in clip_paths:
        path_meta = _get_fps_duration_from_path(path)
        if path_meta is not None:
            fps_per_cam[cam] = path_meta[0]
            durations[cam] = path_meta[1]
        else:
            fps_per_cam[cam] = 1.0
            durations[cam] = 1.0  # Fallback so we do not early-return when ffprobe fails (e.g. placeholder clip)

    if log_callback:
        log_callback("Opening clips for fps/duration.")
    def open_caps() -> tuple[dict[str, Any], dict[str, str]]:
        caps_out: dict[str, Any] = {}
        backends_out: dict[str, str] = {}
        for cam, path in clip_paths:
            try:
                cap = ffmpegcv.VideoCaptureNV(path)
                caps_out[cam] = cap
                backends_out[cam] = "GPU"
            except Exception as e:
                if _is_gpu_not_configured(e):
                    logger.debug(
                        "VideoCaptureNV skipped for %s (GPU not configured/available), using CPU decode.",
                        path,
                    )
                else:
                    logger.warning(
                        "VideoCaptureNV failed for %s (GPU decode attempted, error: %s), falling back to CPU decode.",
                        path,
                        e,
                    )
                try:
                    cap = ffmpegcv.VideoCapture(path)
                    caps_out[cam] = cap
                    backends_out[cam] = "CPU"
                except Exception as e2:
                    logger.warning("Could not open clip %s: %s", path, e2)
        return (caps_out, backends_out)

    caps, decode_backends = open_caps()
    if not caps:
        return []

    if _log_phase_timing and _t0_open is not None and log_callback:
        log_callback(f"Opening clips for fps/duration: {time.monotonic() - _t0_open:.1f}s")

    # Single open: caps already opened; use pre-probed durations/fps (no release/reopen).

    for cam in list(caps.keys()):
        if caps.get(cam) is not None and cam in durations:
            logger.info(
                "Multi-clip open: camera=%s backend=%s path=%s duration_sec=%.2f fps=%.2f",
                cam,
                decode_backends.get(cam, "?"),
                path_per_cam.get(cam, ""),
                durations[cam],
                fps_per_cam.get(cam, 0),
            )

    if _log_phase_timing and _t0_open is not None and log_callback:
        log_callback(f"Clips opened once (no reopen): {time.monotonic() - _t0_open:.1f}s")

    if log_callback and decode_backends:
        cams = ", ".join(sorted(decode_backends.keys()))
        backends_set = set(decode_backends.values())
        if len(backends_set) == 1 and "GPU" in backends_set:
            log_callback(f"Decoding clips ({cams}): GPU (NVDEC).")
        elif len(backends_set) == 1 and "CPU" in backends_set:
            log_callback(f"Decoding clips ({cams}): CPU (GPU not configured or fallback).")
        else:
            parts = [f"{cam}: {decode_backends[cam]}" for cam in sorted(decode_backends.keys())]
            log_callback(f"Decoding clips: {', '.join(parts)}.")

    global_end = max(durations.values()) if durations else 0.0
    if global_end <= 0 or not caps:
        for cap in caps.values():
            if cap is not None:
                cap.release()
        return []

    use_sidecar = all_have_sidecar and len(sidecars) == len(clip_paths)
    if not use_sidecar:
        missing_cameras = [cam for cam, _ in clip_paths if cam not in sidecars]
        logger.warning(
            "Skipping multi-clip extraction: not all cameras have detection sidecars. CE folder=%s cameras_missing_sidecar=%s",
            ce_folder_path,
            missing_cameras,
        )
        for cap in caps.values():
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
        return []
    if log_callback:
        log_callback("Creating extraction metadata (sidecar).")
    logger.debug("Using detection sidecars for multi-clip selection")

    # Phase 1 (EMA pipeline): dense grid, EMA, hysteresis, merge short segments (including first-segment roll-forward).
    assignment_list: list[str] | None = None
    if use_ema_pipeline and _TIMELINE_EMA_AVAILABLE:
        dense_times = _timeline_ema.build_dense_times(
            step_sec, max_frames_min, camera_timeline_analysis_multiplier, global_end
        )
        cameras_list = list(sidecars.keys())
        assignments = _timeline_ema.build_phase1_assignments(
            dense_times,
            cameras_list,
            lambda cam, t: _person_area_at_time(sidecars.get(cam) or [], t),
            native_size_per_cam,
            ema_alpha=camera_timeline_ema_alpha,
            primary_bias_multiplier=camera_timeline_primary_bias_multiplier,
            primary_camera=first_camera_bias,
            hysteresis_margin=camera_switch_hysteresis_margin,
            min_segment_frames=camera_switch_min_segment_frames,
        )
        sample_times_list = [t for t, _ in assignments]
        assignment_list = [c for _, c in assignments]
        if log_callback:
            log_callback(f"Phase 1 EMA: {len(sample_times_list)} sample times, segment merge (min {camera_switch_min_segment_frames} frames).")
    else:
        # Legacy: first-camera bias and hysteresis/min_hold for in-loop selection.
        def _bias_multiplier(cam: str, t_sec: float) -> float:
            if first_camera_bias is None or cam != first_camera_bias:
                return 1.0
            if first_camera_bias_cap_seconds > 0 and t_sec >= first_camera_bias_cap_seconds:
                return 0.0
            if first_camera_bias_decay_seconds <= 0:
                return first_camera_bias_initial
            return first_camera_bias_initial * math.exp(-t_sec / first_camera_bias_decay_seconds)

        merged_t = _detection_timestamps_with_person(sidecars, global_end)
        sample_times_list = _subsample_with_min_gap(merged_t, step_sec, max_frames_min)
        if not sample_times_list:
            sample_times_list = []
            t = 0.0
            while t <= global_end and len(sample_times_list) < max_frames_min:
                sample_times_list.append(t)
                t += step_sec
            logger.debug("Detection-aligned sample list empty; using fixed-step grid (%d times)", len(sample_times_list))
        else:
            logger.debug("Using detection-aligned sample times (%d)", len(sample_times_list))
    if log_callback and assignment_list is None:
        log_callback(f"Built {len(sample_times_list)} sample times for frame selection.")

    # For legacy path: hysteresis and min_hold state.
    current_camera: str | None = None
    frames_on_current = 0
    T_switch: float | None = None

    # Sequential-read, time-sampled: per-camera state (prev_frame, prev_t, curr_frame, curr_t).
    # We advance all readers in lockstep; at each sample time T we use the frame at T for each camera.
    # Reader failures (e.g. ValueError: read of closed file when FFmpeg process dies) drop that camera only.
    _READ_ERRORS = (ValueError, OSError, BrokenPipeError, RuntimeError)
    State = tuple[Any, float, Any, float]  # prev_frame, prev_t, curr_frame, curr_t
    state: dict[str, State] = {}
    for cam in caps:
        if caps[cam] is None:
            continue
        try:
            ret, frame = caps[cam].read()
        except _READ_ERRORS as e:
            remaining = [c for c in caps if caps.get(c) is not None and c != cam]
            logger.warning(
                "Reader process died for camera %s (%s); dropping camera for rest of extraction: %s (sample_time=initial_read remaining_cameras=%s). Check GPU memory and driver if multiple NVDEC readers fail.",
                cam,
                path_per_cam.get(cam, ""),
                e,
                remaining,
            )
            try:
                caps[cam].release()
            except Exception:
                pass
            caps[cam] = None
            state[cam] = (None, -1.0, None, -1.0)
            continue
        if not ret or frame is None:
            state[cam] = (None, -1.0, None, -1.0)
            continue
        fps = fps_per_cam.get(cam, 1.0)
        state[cam] = (None, -1.0, frame, 0.0)
    # frame_index per camera: next frame to read will be at index 1 (first frame was index 0, t=0).
    frame_index: dict[str, int] = {cam: 1 for cam in caps if caps[cam] is not None}

    _t0_read = time.monotonic() if _log_phase_timing else None
    _num_sample_times = len(sample_times_list)
    _progress_interval = max(1, _num_sample_times // 10)
    _num_cams = len([c for c in caps if caps.get(c) is not None])
    _max_workers = min(_num_cams, 8) if _num_cams else 1
    with ThreadPoolExecutor(max_workers=_max_workers) as exec_advance:
        for _step_idx, T in enumerate(sample_times_list, start=1):
            if len(collected) >= max_frames_min:
                break
            if log_callback and _step_idx % _progress_interval == 1:
                log_callback(f"Reading frames ({_step_idx}/{_num_sample_times}).")
            best_camera: str | None = None
            best_frame: Any = None
            best_area = 0.0

            # Advance each camera to T in parallel (each camera has its own cap).
            cameras_to_advance = [c for c in state if caps.get(c) is not None]
            if cameras_to_advance:
                future_to_cam = {
                    exec_advance.submit(
                        _advance_one_camera_to_T,
                        cam,
                        caps[cam],
                        state[cam],
                        T,
                        fps_per_cam.get(cam, 1.0),
                        frame_index.get(cam, 0),
                        durations.get(cam, 0.0),
                    ): cam
                    for cam in cameras_to_advance
                }
                for future in as_completed(future_to_cam):
                    cam = future_to_cam[future]
                    try:
                        new_state, new_idx = future.result()
                        if new_state is None and new_idx is None:
                            try:
                                caps[cam].release()
                            except Exception:
                                pass
                            caps[cam] = None
                            state[cam] = (None, -1.0, None, -1.0)
                            remaining = [c for c in caps if caps.get(c) is not None and c != cam]
                            logger.warning(
                                "Reader process died for camera %s (%s); dropping camera (sample_time_sec=%.2f remaining_cameras=%s).",
                                cam,
                                path_per_cam.get(cam, ""),
                                T,
                                remaining,
                            )
                        else:
                            state[cam] = new_state
                            frame_index[cam] = new_idx
                    except Exception as e:
                        logger.warning("Advance camera %s failed: %s", cam, e)
                        try:
                            caps[cam].release()
                        except Exception:
                            pass
                        caps[cam] = None
                        state[cam] = (None, -1.0, None, -1.0)

            # If no cameras left, return what we have so far.
            if not any(c is not None for c in caps.values()):
                break

            if assignment_list is not None:
                # EMA pipeline: camera comes from Phase 1 assignment; get frame for that camera only.
                idx = _step_idx - 1
                if idx >= len(assignment_list):
                    break
                best_camera = assignment_list[idx]
                prev_f, prev_t, curr_f, curr_t = state.get(best_camera, (None, -1.0, None, -1.0))
                if T >= durations.get(best_camera, 0):
                    continue
                if prev_t <= T < curr_t and prev_f is not None:
                    candidate = prev_f
                elif curr_t <= T and curr_f is not None:
                    candidate = curr_f
                elif prev_f is not None:
                    candidate = prev_f
                else:
                    candidate = curr_f
                best_frame = candidate.copy() if (candidate is not None and hasattr(candidate, "copy")) else candidate
                best_area = _person_area_at_time(sidecars.get(best_camera) or [], T) if best_frame is not None else 0.0
            else:
                # Legacy: for each camera get frame at T and person area (with first-camera bias).
                cam_candidates: dict[str, tuple[Any, float]] = {}  # cam -> (frame, area_with_bias)
                for cam in state:
                    prev_f, prev_t, curr_f, curr_t = state[cam]
                    if T >= durations.get(cam, 0):
                        continue
                    if prev_t <= T < curr_t and prev_f is not None:
                        candidate = prev_f
                    elif curr_t <= T and curr_f is not None:
                        candidate = curr_f
                    elif prev_f is not None:
                        candidate = prev_f
                    else:
                        candidate = curr_f
                    if candidate is None:
                        continue
                    area = _person_area_at_time(sidecars.get(cam) or [], T)
                    area *= _bias_multiplier(cam, T)
                    cam_candidates[cam] = (candidate.copy() if hasattr(candidate, "copy") else candidate, area)

                # Hysteresis: stay on current camera for min 3 frames unless area is 0 or below threshold (escape).
                if current_camera is not None and frames_on_current < 3:
                    curr_area = cam_candidates.get(current_camera, (None, 0.0))[1]
                    below_threshold = person_area_switch_threshold > 0 and curr_area < person_area_switch_threshold
                    if curr_area > 0 and not below_threshold:
                        best_camera = current_camera
                        best_frame, best_area = cam_candidates.get(current_camera, (None, 0.0))
                    else:
                        # Escape: person left or area below threshold; pick camera with largest area.
                        best_camera = max(cam_candidates, key=lambda c: cam_candidates[c][1]) if cam_candidates else None
                        best_frame, best_area = (cam_candidates.get(best_camera, (None, 0.0))) if best_camera else (None, 0.0)
                else:
                    best_camera = max(cam_candidates, key=lambda c: cam_candidates[c][1]) if cam_candidates else None
                    best_frame, best_area = (cam_candidates.get(best_camera, (None, 0.0))) if best_camera else (None, 0.0)
                    if best_camera and current_camera and best_camera != current_camera:
                        current_area = cam_candidates.get(current_camera, (None, 0.0))[1]
                        # Stickiness for non-initial cameras: use same decay/cap as first camera; 0 clamped to 0.1.
                        if first_camera_bias is not None and current_camera == first_camera_bias:
                            effective_current = current_area
                        else:
                            bias_val = max(0.1, camera_switch_bias)
                            t_since_switch = (T - T_switch) if T_switch is not None else 0.0
                            if first_camera_bias_cap_seconds > 0 and t_since_switch >= first_camera_bias_cap_seconds:
                                stickiness = 1.0
                            elif first_camera_bias_decay_seconds <= 0:
                                stickiness = bias_val
                            else:
                                stickiness = bias_val * math.exp(-t_since_switch / first_camera_bias_decay_seconds)
                            effective_current = current_area * stickiness
                        allow_switch = (
                            (person_area_switch_threshold > 0 and current_area < person_area_switch_threshold and best_area > current_area)
                            or (best_area >= camera_switch_ratio * effective_current)
                        )
                        if not allow_switch:
                            best_camera = current_camera
                            best_frame, best_area = cam_candidates.get(current_camera, (None, 0.0))

                # Minimum hold: after a switch, stay on current camera for at least this many frames unless person left (area 0).
                if (
                    best_camera is not None
                    and current_camera is not None
                    and best_camera != current_camera
                    and camera_switch_min_hold_frames > 0
                    and frames_on_current < camera_switch_min_hold_frames
                ):
                    current_raw = _person_area_at_time(sidecars.get(current_camera) or [], T)
                    if current_raw > 0:
                        best_camera = current_camera
                        best_frame, best_area = cam_candidates.get(current_camera, (None, 0.0))

            if best_camera is not None and best_frame is not None:
                meta: dict[str, Any] = {}
                if crop_width > 0 and crop_height > 0 and _CROP_AVAILABLE:
                    entry = _nearest_sidecar_entry(sidecars.get(best_camera) or [], T)
                    detections = (entry.get("detections") or []) if entry else []
                    frame_h, frame_w = best_frame.shape[:2]
                    frame_area = frame_w * frame_h
                    native_w, native_h = native_size_per_cam.get(best_camera, (0, 0))
                    if native_w > 0 and native_h > 0:
                        frame_area = native_w * native_h
                    reference_area = min(target_crop_area, frame_area) if target_crop_area > 0 else frame_area
                    if not detections:
                        best_frame = _crop_utils.center_crop(best_frame, crop_width, crop_height)
                    else:
                        person_dets = [d for d in detections if (d.get("label") or "").lower() in PREFERRED_LABELS]
                        if not person_dets:
                            best_frame = _crop_utils.center_crop(best_frame, crop_width, crop_height)
                        else:
                            largest = max(person_dets, key=lambda d: float(d.get("area") or 0))
                            person_area_val = float(largest.get("area") or 0)
                            if (
                                reference_area > 0
                                and tracking_target_frame_percent > 0
                                and (person_area_val / reference_area) >= (tracking_target_frame_percent / 100.0)
                            ):
                                best_frame = _crop_utils.full_frame_resize_to_target(
                                    best_frame, crop_width, crop_height
                                )
                                meta["is_full_frame_resize"] = True
                            else:
                                cp = largest.get("centerpoint")
                                if cp and len(cp) >= 2:
                                    best_frame = _crop_utils.crop_around_center(
                                        best_frame, cp[0], cp[1], crop_width, crop_height
                                    )
                                else:
                                    best_frame = _crop_utils.center_crop(best_frame, crop_width, crop_height)
                                meta["is_full_frame_resize"] = False
                raw_person_area = _person_area_at_time(sidecars.get(best_camera) or [], T)
                meta["person_area"] = int(raw_person_area)
                # Legacy: skip frames with no person when sidecar in use. EMA: drop when config says so.
                if not use_ema_pipeline:
                    skip_append = bool(use_sidecar and sidecars and raw_person_area <= 0)
                else:
                    skip_append = (
                        (raw_person_area <= 0) if camera_timeline_final_yolo_drop_no_person else False
                    )
                if not skip_append:
                    collected.append(ExtractedFrame(frame=best_frame, timestamp_sec=T, camera=best_camera, metadata=meta))
                if best_camera == current_camera:
                    frames_on_current += 1
                else:
                    current_camera = best_camera
                    frames_on_current = 1
                    T_switch = T

    if _log_phase_timing and _t0_read is not None and log_callback:
        log_callback(f"Reading frames: {time.monotonic() - _t0_read:.1f}s")
    for cap in caps.values():
        if cap is not None:
            cap.release()
    return collected
