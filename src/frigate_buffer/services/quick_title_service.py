"""
Quick-title pipeline: fetch latest.jpg, run YOLO, crop around detections, get AI title, update state and notify.

Used by the orchestrator as the on_quick_title_trigger callback. Keeps numpy/tensor
bridge and image fetch logic out of the orchestrator.
"""

import logging
import os
from typing import Any

import cv2
import numpy as np
import requests

from frigate_buffer.models import EventPhase
from frigate_buffer.services import crop_utils

logger = logging.getLogger("frigate-buffer")

# Timeout for fetching latest.jpg from Frigate (seconds).
LATEST_JPG_TIMEOUT = 10


def _numpy_bgr_to_tensor_bchw_rgb(arr: np.ndarray) -> Any:
    """Convert numpy HWC BGR to torch.Tensor BCHW RGB for crop_utils (GPU pipeline)."""
    import torch
    rgb = arr[:, :, [2, 1, 0]].copy()
    t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return t.to(device=device, dtype=torch.uint8)


def _tensor_bchw_rgb_to_numpy_bgr(t: Any) -> np.ndarray:
    """Convert torch.Tensor BCHW RGB to numpy HWC BGR (e.g. for encoding at boundary)."""
    if t.dim() == 4:
        t = t.squeeze(0)
    return t.permute(1, 2, 0).cpu().numpy()[:, :, [2, 1, 0]]


class QuickTitleService:
    """Runs the quick-title pipeline: fetch latest frame, YOLO, crop, AI title, state update, notify."""

    def __init__(
        self,
        config: dict[str, Any],
        state_manager: Any,
        file_manager: Any,
        consolidated_manager: Any,
        video_service: Any,
        ai_analyzer: Any,
        notifier: Any,
    ) -> None:
        self._config = config
        self._state_manager = state_manager
        self._file_manager = file_manager
        self._consolidated_manager = consolidated_manager
        self._video_service = video_service
        self._ai_analyzer = ai_analyzer
        self._notifier = notifier

    def run_quick_title(
        self,
        event_id: str,
        camera: str,
        label: str,
        ce_id: str,
        camera_folder_path: str,
        tag_override: str | None = None,
    ) -> None:
        """Fetch live frame, run YOLO, crop around detections with 10% padding, get AI title, update state and notify."""
        if not self._ai_analyzer:
            return
        frigate_url = (self._config.get("FRIGATE_URL") or "").rstrip("/")
        if not frigate_url:
            logger.debug("Quick title skipped: no Frigate URL")
            return
        url = f"{frigate_url}/api/{camera}/latest.jpg"
        try:
            resp = requests.get(url, timeout=LATEST_JPG_TIMEOUT)
            resp.raise_for_status()
            arr = np.frombuffer(resp.content, dtype=np.uint8)
            image_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if image_bgr is None:
                logger.warning("Quick title: failed to decode latest.jpg for %s", camera)
                return
        except requests.RequestException as e:
            logger.warning("Quick title: failed to fetch latest.jpg for %s: %s", camera, e)
            return
        detections = self._video_service.run_detection_on_image(image_bgr, self._config)
        if detections:
            image_tensor = _numpy_bgr_to_tensor_bchw_rgb(image_bgr)
            cropped_tensor = crop_utils.crop_around_detections_with_padding(
                image_tensor, detections, padding_fraction=0.1
            )
            image_to_send = cropped_tensor
        else:
            image_to_send = image_bgr
        title = self._ai_analyzer.generate_quick_title(image_to_send, camera, label)
        if not title or not isinstance(title, dict) or not (title.get("title") or "").strip():
            logger.debug("Quick title: no title returned for %s", event_id)
            return
        title_str = (title.get("title") or "").strip()
        description_str = (title.get("description") or "").strip()
        event = self._state_manager.get_event(event_id)
        if not event:
            logger.debug("Quick title: event %s no longer in state", event_id)
            return
        self._state_manager.set_genai_metadata(
            event_id,
            title_str,
            description_str,
            event.severity or "detection",
            event.threat_level,
            scene=event.genai_scene,
        )
        event = self._state_manager.get_event(event_id)
        if event and event.folder_path:
            self._file_manager.write_summary(event.folder_path, event)
            self._file_manager.write_metadata_json(event.folder_path, event)
        ce = self._consolidated_manager.get_by_frigate_event(event_id)
        primary = (
            self._state_manager.get_event(ce.primary_event_id)
            if ce and ce.primary_event_id
            else event
        )
        media_folder = (
            os.path.join(
                ce.folder_path,
                self._file_manager.sanitize_camera_name(ce.primary_camera or ""),
            )
            if ce and ce.primary_camera
            else (camera_folder_path if ce else camera_folder_path)
        )
        if not ce:
            media_folder = camera_folder_path
            logger.warning("Quick title: event %s not in a CE (invariant: all events are CEs)", event_id)
        notify_target = type(
            "NotifyTarget",
            (),
            {
                "event_id": ce_id if ce else event_id,
                "camera": ce.camera if ce else camera,
                "label": ce.label if ce else label,
                "folder_path": media_folder,
                "created_at": ce.start_time if ce else event.created_at,
                "end_time": ce.end_time if ce else event.end_time,
                "phase": EventPhase.FINALIZED,
                "genai_title": title_str,
                "genai_description": description_str,
                "ai_description": None,
                "review_summary": None,
                "threat_level": ce.final_threat_level if ce else event.threat_level,
                "severity": ce.severity if ce else (event.severity or "detection"),
                "snapshot_downloaded": getattr(primary, "snapshot_downloaded", False) if primary else False,
                "clip_downloaded": getattr(primary, "clip_downloaded", False) if primary else False,
                "image_url_override": getattr(primary, "image_url_override", None) if primary else None,
            },
        )()
        self._notifier.publish_notification(
            notify_target,
            "snapshot_ready",
            tag_override=tag_override or f"frigate_{ce_id if ce else event_id}",
        )
        logger.info("Quick title applied for %s: %s", event_id, title_str[:50])
