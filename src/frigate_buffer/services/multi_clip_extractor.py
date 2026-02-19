"""
Target-centric multi-clip frame extraction for consolidated events.

When detection sidecars (detection.json per camera) exist from transcode, reads them
and picks the camera with largest person area per time step without running a detector.
Otherwise falls back to OpenCV HOG person detection on each frame. No Frigate metadata.

Uses ffmpegcv for decode and a sequential-read, time-sampled strategy (no seek),
since ffmpegcv readers do not support frame-index seek.
"""

from __future__ import annotations

import json
import os
import logging
from typing import Any, Callable

logger = logging.getLogger("frigate-buffer")

try:
    import cv2
    import ffmpegcv
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False


# Preferred label for target-centric selection (person has highest priority)
PREFERRED_LABELS = ("person", "people", "pedestrian")

# Sidecar filename written by transcode (video.py)
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


def _load_sidecar_for_camera(camera_folder: str) -> list[dict[str, Any]] | None:
    """Load detection.json from camera folder. Returns list of {timestamp_sec, detections} or None."""
    path = os.path.join(camera_folder, DETECTION_SIDECAR_FILENAME)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            return None
        return data
    except (OSError, json.JSONDecodeError) as e:
        logger.debug("Could not load sidecar %s: %s", path, e)
        return None


def _person_area_at_time(sidecar_entries: list[dict[str, Any]], t_sec: float) -> float:
    """Return person area at timestamp t_sec using nearest sidecar entry (by timestamp_sec)."""
    if not sidecar_entries:
        return 0.0
    best = min(sidecar_entries, key=lambda e: abs((e.get("timestamp_sec") or 0) - t_sec))
    return _person_area_from_detections(best.get("detections") or [])


def _detect_person_area(frame: Any) -> float:
    """Run HOG person detector on frame. Returns total pixel area of detections (0 if none)."""
    if not _CV2_AVAILABLE:
        return 0.0
    try:
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        # DetectMultiScale returns (rects, weights). Rects are (x, y, w, h)
        rects, _ = hog.detectMultiScale(
            frame,
            winStride=(8, 8),
            padding=(8, 8),
            scale=1.05,
        )
        if rects is None or len(rects) == 0:
            return 0.0
        return float(sum(w * h for (_, _, w, h) in rects))
    except Exception as e:
        logger.debug("HOG detection failed: %s", e)
        return 0.0


def _get_fps_and_duration(cap: Any, path: str) -> tuple[float, float, bool]:
    """
    Get (fps, duration_sec, used_read_to_eof) from a video capture in an ffmpegcv-safe way.

    Uses only ffmpegcv-compatible APIs: getattr(cap, "fps"), getattr(cap, "count"),
    and len(cap). Does NOT use cap.get()—ffmpegcv readers (FFmpegReaderNV, etc.) have
    no OpenCV-style .get(CAP_PROP_*). If frame count is unknown, reads until EOF
    to count frames and returns duration = count/fps with used_read_to_eof=True;
    caller must reopen the clip after this.
    """
    del path  # unused; kept for API clarity
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
    # Unknown count: read until EOF to get frame count (consumes stream; caller must reopen).
    frame_idx = 0
    while True:
        ret, _ = cap.read()
        if not ret:
            break
        frame_idx += 1
    duration = (frame_idx / fps) if fps > 0 and frame_idx > 0 else 0.0
    return (fps, duration, True)


def extract_target_centric_frames(
    ce_folder_path: str,
    max_frames_sec: float,
    max_frames_min: int,
    *,
    log_callback: Callable[[str], None] | None = None,
) -> list[tuple[Any, float, str]]:
    """
    Extract time-ordered, target-centric frames from all clips under a CE folder.

    Uses full clip per camera with a sequential-read, time-sampled strategy (no seek).
    Steps at max_frames_sec intervals; at each sample time picks the camera with
    largest person area (from sidecar or HOG). Returns list of (frame_bgr, timestamp_sec, camera_name).
    Caps at max_frames_min.
    """
    if not _CV2_AVAILABLE:
        logger.warning("cv2/ffmpegcv not available, skipping multi-clip extraction")
        return []

    # Discover camera clips: ce_folder/CameraName/clip.mp4
    clip_paths: list[tuple[str, str]] = []  # (camera_name, path)
    try:
        for name in os.listdir(ce_folder_path):
            sub = os.path.join(ce_folder_path, name)
            if os.path.isdir(sub) and not name.startswith("."):
                clip_path = os.path.join(sub, "clip.mp4")
                if os.path.isfile(clip_path):
                    clip_paths.append((name, clip_path))
    except OSError as e:
        logger.warning("Could not scan CE folder %s: %s", ce_folder_path, e)
        return []

    if len(clip_paths) < 1:
        logger.debug("No clips found in CE folder %s", ce_folder_path)
        return []

    step_sec = float(max_frames_sec) if max_frames_sec > 0 else 1.0
    collected: list[tuple[Any, float, str]] = []
    path_per_cam = {cam: path for cam, path in clip_paths}

    # Prefer sidecar-based selection when every camera has detection.json (from transcode+ultralytics).
    camera_folders = {cam: os.path.dirname(path) for cam, path in clip_paths}
    sidecars: dict[str, list[dict[str, Any]]] = {}
    all_have_sidecar = True
    for cam, folder in camera_folders.items():
        entries = _load_sidecar_for_camera(folder)
        if entries is not None:
            sidecars[cam] = entries
        else:
            all_have_sidecar = False

    def open_caps() -> dict[str, Any]:
        caps_out: dict[str, Any] = {}
        for cam, path in clip_paths:
            try:
                cap = ffmpegcv.VideoCaptureNV(path)
                caps_out[cam] = cap
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
                except Exception as e2:
                    logger.warning("Could not open clip %s: %s", path, e2)
        return caps_out

    caps = open_caps()
    if not caps:
        return []

    # Get fps and duration per camera (ffmpegcv-safe; may consume stream if count unknown).
    durations: dict[str, float] = {}
    fps_per_cam: dict[str, float] = {}
    caps_to_reopen: list[str] = []
    for cam, path in clip_paths:
        cap = caps[cam]
        fps, duration, used_read_to_eof = _get_fps_and_duration(cap, path)
        durations[cam] = duration
        fps_per_cam[cam] = fps
        if used_read_to_eof:
            caps_to_reopen.append(cam)
        cap.release()
    caps.clear()

    # Reopen all caps (we released them above). For caps we consumed, we must reopen to read again.
    decode_backends: dict[str, str] = {}
    for cam, path in clip_paths:
        cap = None
        try:
            cap = ffmpegcv.VideoCaptureNV(path)
            decode_backends[cam] = "GPU"
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
                decode_backends[cam] = "CPU"
            except Exception:
                cap = None
        if cap is not None and getattr(cap, "isOpened", lambda: True)():
            caps[cam] = cap
        else:
            if cap is not None:
                cap.release()
            durations[cam] = 0.0

    if log_callback and decode_backends:
        if all(b == "GPU" for b in decode_backends.values()):
            log_callback("Decoding clips: GPU (NVDEC).")
        else:
            log_callback("Decoding clips: CPU (GPU not configured or fallback).")

    global_end = max(durations.values()) if durations else 0.0
    if global_end <= 0 or not caps:
        for cap in caps.values():
            if cap is not None:
                cap.release()
        return []

    use_sidecar = all_have_sidecar and len(sidecars) == len(clip_paths)
    if use_sidecar:
        logger.debug("Using detection sidecars for multi-clip selection (no on-frame detector)")
    else:
        missing_cameras = [cam for cam, _ in clip_paths if cam not in sidecars]
        logger.warning(
            "HOG fallback in use for multi-clip selection: not all cameras have detection sidecars (expected when transcode used libx264). CE folder=%s cameras_missing_sidecar=%s. Fix by ensuring NVENC transcode runs so detection.json is written per camera.",
            ce_folder_path,
            missing_cameras,
        )

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
            logger.warning(
                "Reader process died for camera %s (%s); dropping camera for rest of extraction: %s",
                cam,
                path_per_cam.get(cam, ""),
                e,
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

    next_sample_time = 0.0
    while next_sample_time <= global_end and len(collected) < max_frames_min:
        T = next_sample_time
        best_camera: str | None = None
        best_frame: Any = None
        best_area = 0.0

        # Advance any camera whose curr_t < T until we have curr_t >= T (or EOF).
        for cam in list(state.keys()):
            prev_f, prev_t, curr_f, curr_t = state[cam]
            while curr_t < T:
                cap = caps.get(cam)
                if cap is None:
                    break
                try:
                    ret, frame = cap.read()
                except _READ_ERRORS as e:
                    logger.warning(
                        "Reader process died for camera %s (%s); dropping camera for rest of extraction: %s",
                        cam,
                        path_per_cam.get(cam, ""),
                        e,
                    )
                    try:
                        cap.release()
                    except Exception:
                        pass
                    caps[cam] = None
                    if curr_f is not None:
                        state[cam] = (curr_f, curr_t, curr_f, curr_t)
                    else:
                        state[cam] = (None, -1.0, None, -1.0)
                    break
                if not ret or frame is None:
                    # EOF: use last frame as final state
                    if curr_f is not None:
                        state[cam] = (curr_f, curr_t, curr_f, curr_t)
                    break
                fps = fps_per_cam.get(cam, 1.0)
                idx = frame_index.get(cam, 0)
                new_t = idx / fps if fps > 0 else curr_t
                frame_index[cam] = idx + 1
                state[cam] = (curr_f, curr_t, frame, new_t)
                prev_f, prev_t, curr_f, curr_t = state[cam]

        # If no cameras left, return what we have so far.
        if not any(c is not None for c in caps.values()):
            break

        # For each camera, the frame at T is the one with timestamp <= T closest to T.
        for cam in state:
            prev_f, prev_t, curr_f, curr_t = state[cam]
            if T >= durations.get(cam, 0):
                continue
            # Frame to use at T: prev_f if prev_t <= T < curr_t, else curr_f if curr_t <= T.
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
            if use_sidecar and sidecars:
                area = _person_area_at_time(sidecars[cam], T)
            else:
                area = _detect_person_area(candidate)
            if area > best_area:
                best_area = area
                best_camera = cam
                best_frame = candidate.copy() if hasattr(candidate, "copy") else candidate

        if best_camera is not None and best_frame is not None:
            collected.append((best_frame, T, best_camera))
        next_sample_time += step_sec

    for cap in caps.values():
        if cap is not None:
            cap.release()
    return collected
