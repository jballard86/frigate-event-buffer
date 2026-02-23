"""
Target-centric multi-clip frame extraction for consolidated events.

Requires detection sidecars (detection.json per camera) from generate_detection_sidecar.
Reads sidecars and picks the camera with largest person area per time step.
If any camera lacks a sidecar, returns [] (no on-frame detector fallback). No Frigate metadata.

Uses gpu_decoder (PyNvVideoCodec) for GPU decode: one decoder per camera, get_frames([frame_idx])
per sample time. ExtractedFrame.frame is torch.Tensor BCHW RGB. GPU_LOCK serializes decoder
access. No ffmpegcv fallback.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

from frigate_buffer.constants import NVDEC_INIT_FAILURE_PREFIX
from frigate_buffer.services.gpu_decoder import create_decoder
from frigate_buffer.services.video import GPU_LOCK

logger = logging.getLogger("frigate-buffer")

try:
    from frigate_buffer.services import crop_utils as _crop_utils
    _CROP_AVAILABLE = True
except ImportError:
    _CROP_AVAILABLE = False

try:
    from frigate_buffer.models import ExtractedFrame
except ImportError:
    ExtractedFrame = None  # type: ignore[misc, assignment]

from frigate_buffer.services import timeline_ema as _timeline_ema

# Preferred label for target-centric selection (person has highest priority)
PREFERRED_LABELS = ("person", "people", "pedestrian")

# Sidecar filename written by generate_detection_sidecar (video.py)
DETECTION_SIDECAR_FILENAME = "detection.json"


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


def extract_target_centric_frames(
    ce_folder_path: str,
    max_frames_sec: float,
    max_frames_min: int,
    *,
    crop_width: int = 0,
    crop_height: int = 0,
    tracking_target_frame_percent: int = 40,
    primary_camera: str | None = None,
    decode_second_camera_cpu_only: bool = False,  # Ignored: GPU decode only (PyNvVideoCodec), no CPU path
    log_callback: Callable[[str], None] | None = None,
    config: dict[str, Any] | None = None,
    camera_timeline_analysis_multiplier: float = 2.0,
    camera_timeline_ema_alpha: float = 0.4,
    camera_timeline_primary_bias_multiplier: float = 1.2,
    camera_switch_min_segment_frames: int = 5,
    camera_switch_hysteresis_margin: float = 1.15,
    camera_timeline_final_yolo_drop_no_person: bool = False,
) -> list[Any]:
    """
    Extract time-ordered, target-centric frames from all clips under a CE folder.

    Uses the EMA pipeline (dense grid + EMA smoothing + hysteresis + segment merge)
    to assign one camera per sample time from detection sidecars. Decode via gpu_decoder
    (PyNvVideoCodec): one decoder per camera, get_frames([frame_idx]) per sample time.
    ExtractedFrame.frame is torch.Tensor BCHW RGB. When person area >= tracking_target_frame_percent
    of reference_area, uses full-frame resize with letterbox and sets metadata is_full_frame_resize=True.
    Caps at max_frames_min. Returns [] if any camera lacks detection.json or if torch unavailable.

    Optional config: LOG_EXTRACTION_PHASE_TIMING (bool), CUDA_DEVICE_INDEX (int, default 0).
    """
    if ExtractedFrame is None:
        logger.warning("ExtractedFrame not available, skipping multi-clip extraction")
        return []
    try:
        import torch
    except ImportError:
        logger.warning("torch not available, skipping multi-clip extraction")
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

    cuda_device_index = int(config.get("CUDA_DEVICE_INDEX", 0)) if config else 0
    _t0_open = time.monotonic() if _log_phase_timing else None
    decoders: dict[str, Any] = {}  # camera -> DecoderContext
    durations: dict[str, float] = {}
    fps_per_cam: dict[str, float] = {}
    frame_count_per_cam: dict[str, int] = {}
    if log_callback:
        log_callback("Opening clips (PyNvVideoCodec NVDEC).")
    # ffprobe outside GPU_LOCK so the GPU is not held during subprocess I/O (Finding 4.2).
    clip_metadata: dict[str, tuple[float, float] | None] = {}
    for (_cam, path) in clip_paths:
        clip_metadata[path] = _get_fps_duration_from_path(path)
    with contextlib.ExitStack() as stack:
        with GPU_LOCK:
            for (cam, path) in clip_paths:
                try:
                    ctx = stack.enter_context(create_decoder(path, gpu_id=cuda_device_index))
                except Exception as e:
                    logger.error(
                        "%s (decoder open failed). cam=%s path=%s error=%s Check GPU, drivers; container may restart.",
                        NVDEC_INIT_FAILURE_PREFIX,
                        cam,
                        path,
                        e,
                    )
                    logger.warning(
                        "Decoder failed for %s (%s): %s",
                        cam,
                        path,
                        e,
                    )
                    if torch.cuda.is_available():
                        try:
                            logger.debug("CUDA memory: %s", torch.cuda.memory_summary(abbreviated=True))
                        except Exception:
                            pass
                    return []
                count = len(ctx)
                path_meta = clip_metadata.get(path)
                if path_meta is not None:
                    fps = path_meta[0]
                    duration_sec = path_meta[1]
                    if count <= 0:
                        count = max(1, int(duration_sec * fps))
                else:
                    fps = 30.0
                    duration_sec = count / fps if count > 0 and fps > 0 else 0.0
                if fps <= 0:
                    fps = 1.0
                decoders[cam] = ctx
                fps_per_cam[cam] = fps
                frame_count_per_cam[cam] = count
                durations[cam] = count / fps if fps > 0 else 0.0

        if _log_phase_timing and _t0_open is not None and log_callback:
            log_callback(f"Opening clips: {time.monotonic() - _t0_open:.1f}s")
        for cam in decoders:
            logger.info(
                "Multi-clip open: camera=%s path=%s duration_sec=%.2f fps=%.2f",
                cam,
                path_per_cam.get(cam, ""),
                durations[cam],
                fps_per_cam.get(cam, 0),
            )
        if log_callback:
            cams = ", ".join(sorted(decoders.keys()))
            log_callback(f"Decoding clips ({cams}): PyNvVideoCodec NVDEC.")
        global_end = max(durations.values()) if durations else 0.0
        if global_end <= 0 or not decoders:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return []
        use_sidecar = all_have_sidecar and len(sidecars) == len(clip_paths)
        if not use_sidecar:
            missing_cameras = [cam for cam, _ in clip_paths if cam not in sidecars]
            logger.warning(
                "Skipping multi-clip extraction: not all cameras have detection sidecars. CE folder=%s cameras_missing_sidecar=%s",
                ce_folder_path,
                missing_cameras,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return []
        if log_callback:
            log_callback("Creating extraction metadata (sidecar).")
        logger.debug("Using detection sidecars for multi-clip selection")

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
            primary_camera=primary_camera,
            hysteresis_margin=camera_switch_hysteresis_margin,
            min_segment_frames=camera_switch_min_segment_frames,
        )
        sample_times_list = [t for t, _ in assignments]
        assignment_list = [c for _, c in assignments]
        if log_callback:
            log_callback(f"Phase 1 EMA: {len(sample_times_list)} sample times, segment merge (min {camera_switch_min_segment_frames} frames).")

        current_camera: str | None = None
        frames_on_current = 0

        _t0_read = time.monotonic() if _log_phase_timing else None
        _num_sample_times = len(sample_times_list)
        _progress_interval = max(1, _num_sample_times // 10)
        dropped_cameras: set[str] = set()
        for _step_idx, T in enumerate(sample_times_list, start=1):
            if len(collected) >= max_frames_min:
                break
            if log_callback and _step_idx % _progress_interval == 1:
                log_callback(f"Reading frames ({_step_idx}/{_num_sample_times}).")
            idx = _step_idx - 1
            if idx >= len(assignment_list):
                break
            best_camera = assignment_list[idx]
            if best_camera in dropped_cameras:
                continue
            if T >= durations.get(best_camera, 0):
                continue
            ctx = decoders.get(best_camera)
            if ctx is None:
                continue
            fcount = frame_count_per_cam.get(best_camera, 0)
            frame_idx = 0  # set before try so except can log it
            try:
                with GPU_LOCK:
                    # PTS-based index to reduce variable frame rate jitter (decoder time→index mapping).
                    frame_idx = (
                        min(max(0, ctx.get_index_from_time_in_seconds(T)), fcount - 1)
                        if fcount > 0
                        else 0
                    )
                    batch = ctx.get_frames([frame_idx])
            except Exception as e:
                logger.error(
                    "%s (decoder get_frames failed). cam=%s T=%.2f frame_idx=%s error=%s",
                    NVDEC_INIT_FAILURE_PREFIX,
                    best_camera,
                    T,
                    frame_idx,
                    e,
                    exc_info=True,
                )
                logger.warning(
                    "Decoder get_frames failed for camera %s at T=%.2f: %s",
                    best_camera,
                    T,
                    e,
                )
                dropped_cameras.add(best_camera)
                continue
            if batch is None or batch.shape[0] < 1:
                continue
            frame_t = batch[0:1].clone()
            del batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            best_tensor = frame_t
            meta: dict[str, Any] = {}
            if crop_width > 0 and crop_height > 0 and _CROP_AVAILABLE:
                entry = _nearest_sidecar_entry(sidecars.get(best_camera) or [], T)
                detections = (entry.get("detections") or []) if entry else []
                frame_h, frame_w = int(best_tensor.shape[2]), int(best_tensor.shape[3])
                frame_area = frame_w * frame_h
                native_w, native_h = native_size_per_cam.get(best_camera, (0, 0))
                if native_w > 0 and native_h > 0:
                    frame_area = native_w * native_h
                reference_area = min(target_crop_area, frame_area) if target_crop_area > 0 else frame_area
                if not detections:
                    best_tensor = _crop_utils.center_crop(best_tensor, crop_width, crop_height)
                else:
                    person_dets = [d for d in detections if (d.get("label") or "").lower() in PREFERRED_LABELS]
                    if not person_dets:
                        best_tensor = _crop_utils.center_crop(best_tensor, crop_width, crop_height)
                    else:
                        largest = max(person_dets, key=lambda d: float(d.get("area") or 0))
                        person_area_val = float(largest.get("area") or 0)
                        if (
                            reference_area > 0
                            and tracking_target_frame_percent > 0
                            and (person_area_val / reference_area) >= (tracking_target_frame_percent / 100.0)
                        ):
                            best_tensor = _crop_utils.full_frame_resize_to_target(
                                best_tensor, crop_width, crop_height
                            )
                            meta["is_full_frame_resize"] = True
                        else:
                            cp = largest.get("centerpoint")
                            if cp and len(cp) >= 2:
                                best_tensor = _crop_utils.crop_around_center(
                                    best_tensor, cp[0], cp[1], crop_width, crop_height
                                )
                            else:
                                best_tensor = _crop_utils.center_crop(best_tensor, crop_width, crop_height)
                            meta["is_full_frame_resize"] = False
            raw_person_area = _person_area_at_time(sidecars.get(best_camera) or [], T)
            meta["person_area"] = int(raw_person_area)
            skip_append = (
                (raw_person_area <= 0) if camera_timeline_final_yolo_drop_no_person else False
            )
            if not skip_append:
                collected.append(ExtractedFrame(frame=best_tensor, timestamp_sec=T, camera=best_camera, metadata=meta))
            if best_camera == current_camera:
                frames_on_current += 1
            else:
                current_camera = best_camera
                frames_on_current = 1

        if _log_phase_timing and _t0_read is not None and log_callback:
            log_callback(f"Reading frames: {time.monotonic() - _t0_read:.1f}s")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return collected
