"""
Target-centric multi-clip frame extraction for consolidated events.

When detection sidecars (detection.json per camera) exist from transcode, reads them
and picks the camera with largest person area per time step without running a detector.
Otherwise falls back to OpenCV HOG person detection on each frame. No Frigate metadata.
"""

from __future__ import annotations

import json
import os
import logging
from typing import Any

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


def extract_target_centric_frames(
    ce_folder_path: str,
    max_frames_sec: float,
    max_frames_min: int,
) -> list[tuple[Any, float, str]]:
    """
    Extract time-ordered, target-centric frames from all clips under a CE folder.

    Uses full clip per camera, steps at max_frames_sec intervals, runs object detection
    on each extracted frame, picks the camera with largest person area per time step.
    Returns list of (frame_bgr, timestamp_sec, camera_name). Caps at max_frames_min.
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

    try:
        caps = {cam: ffmpegcv.VideoCaptureNV(path) for cam, path in clip_paths}
    except Exception:
        try:
            caps = {cam: ffmpegcv.VideoCapture(path) for cam, path in clip_paths}
        except Exception as e:
            logger.warning("Could not open clips: %s", e)
            return []

    durations: dict[str, float] = {}
    for cam, path in clip_paths:
        cap = caps[cam]
        fps = cap.get(cv2.CAP_PROP_FPS) or 1
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = count / fps if fps > 0 and count > 0 else 0
        durations[cam] = duration

    global_start = 0.0
    global_end = max(durations.values()) if durations else 0.0

    if global_end <= 0:
        for cap in caps.values():
            cap.release()
        return []

    use_sidecar = all_have_sidecar and len(sidecars) == len(clip_paths)
    if use_sidecar:
        logger.debug("Using detection sidecars for multi-clip selection (no on-frame detector)")

    current_time = global_start
    while current_time <= global_end and len(collected) < max_frames_min:
        best_camera: str | None = None
        best_frame: Any = None
        best_area = 0.0

        if use_sidecar and sidecars:
            for camera in caps:
                if current_time >= durations.get(camera, 0):
                    continue
                area = _person_area_at_time(sidecars[camera], current_time)
                if area > best_area:
                    best_area = area
                    best_camera = camera
            if best_camera is not None:
                cap = caps[best_camera]
                cap.set(cv2.CAP_PROP_POS_MSEC, current_time * 1000)
                ret, frame = cap.read()
                if ret and frame is not None:
                    best_frame = frame.copy()
        else:
            for camera, cap in caps.items():
                if current_time >= durations.get(camera, 0):
                    continue
                cap.set(cv2.CAP_PROP_POS_MSEC, current_time * 1000)
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue
                area = _detect_person_area(frame)
                if area > best_area:
                    best_area = area
                    best_camera = camera
                    best_frame = frame.copy()

        if best_camera is not None and best_frame is not None:
            collected.append((best_frame, current_time, best_camera))

        current_time += step_sec

    for cap in caps.values():
        cap.release()

    return collected
