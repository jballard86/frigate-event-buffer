"""
AI Analyzer Service - Gemini proxy integration for clip analysis.

Extracts frames from a video clip, builds a system prompt, sends to an
OpenAI-compatible Gemini proxy, and returns the analysis metadata to the caller.
Does NOT publish to MQTT; the orchestrator persists results and notifies HA.
"""

import os
import json
import base64
import logging
import threading
import time
from datetime import datetime
from typing import Any, Sequence

# Relative path under CE folder for detection sidecar (must match multi_clip_extractor)
_DETECTION_SIDECAR_FILENAME = "detection.json"

from frigate_buffer.managers.file import write_ai_frame_analysis_multi_cam

from frigate_buffer.services import crop_utils
from frigate_buffer.services.multi_clip_extractor import extract_target_centric_frames

import cv2
import requests
import urllib3.exceptions

logger = logging.getLogger('frigate-buffer')


def _log_proxy_failure(proxy_url: str, attempt: int, exc: Exception) -> None:
    """Log proxy request failure with URL, optional response body, and a hint for connection errors."""
    err_name = type(exc).__name__
    err_msg = str(exc)
    hint = ""
    if "refused" in err_msg.lower() or "Connection refused" in err_msg:
        hint = f" Ensure the AI proxy is running at {proxy_url}."
    # When raise_for_status() raised (requests.exceptions.HTTPError), include response body for debugging
    response_body = ""
    resp = getattr(exc, "response", None)
    if resp is not None:
        try:
            data = resp.json()
            # Prefer structured error message (OpenAI/Gemini style)
            err_obj = data.get("error") if isinstance(data, dict) else None
            if isinstance(err_obj, dict) and err_obj.get("message"):
                response_body = json.dumps({"error": {"message": err_obj.get("message"), "type": err_obj.get("type", "unknown")}})
            else:
                response_body = json.dumps(data) if data is not None else ""
        except (ValueError, TypeError, AttributeError):
            try:
                text = getattr(resp, "text", None) or (resp.content.decode("utf-8", errors="replace") if getattr(resp, "content", None) else "")
                response_body = (text or "")[:LOG_MAX_RESPONSE_BODY]
            except Exception:
                response_body = repr(getattr(resp, "content", None))[:LOG_MAX_RESPONSE_BODY]
        if len(response_body) > LOG_MAX_RESPONSE_BODY:
            response_body = response_body[:LOG_MAX_RESPONSE_BODY] + "..."
    if response_body:
        logger.error(
            "Proxy request failed [%s]. URL: %s. Body: %s. attempt=%s/2.%s",
            getattr(resp, "status_code", "?"), proxy_url, response_body, attempt, hint,
        )
    else:
        logger.error(
            "Proxy request failed on attempt %s/2: %s: %s. url=%s.%s",
            attempt, err_name, err_msg, proxy_url, hint,
        )


# Max chars to log for proxy response body
LOG_MAX_RESPONSE_BODY = 2000

# Resize frame width for API (preserve aspect or fixed width)
FRAME_MAX_WIDTH = 1280


class GeminiAnalysisService:
    """Service that analyzes video clips via Gemini proxy and returns metadata to the caller."""

    def __init__(self, config: dict):
        self.config = config
        gemini = config.get('GEMINI') or {}
        # Prefer flat keys; fallback to nested for backward compat. API key: config already applies env override in load_config.
        self._proxy_url = (config.get('GEMINI_PROXY_URL') or gemini.get('proxy_url') or '').strip().rstrip('/')
        self._api_key = gemini.get('api_key') or ''
        self._model = config.get('GEMINI_PROXY_MODEL') or gemini.get('model') or 'gemini-2.5-flash-lite'
        self._enabled = bool(gemini.get('enabled', False))
        self._prompt_template: str | None = None
        # Proxy tuning (flat keys)
        self._temperature = float(config.get('GEMINI_PROXY_TEMPERATURE', 0.3))
        self._top_p = float(config.get('GEMINI_PROXY_TOP_P', 1))
        self._frequency_penalty = float(config.get('GEMINI_PROXY_FREQUENCY_PENALTY', 0))
        self._presence_penalty = float(config.get('GEMINI_PROXY_PRESENCE_PENALTY', 0))
        # Smart seeking and multi-cam
        self.buffer_offset = config.get('EXPORT_BUFFER_BEFORE', 5)
        self.max_frames_sec = config.get('MAX_MULTI_CAM_FRAMES_SEC', 2)
        self.max_frames_min = config.get('MAX_MULTI_CAM_FRAMES_MIN', 45)
        self.crop_width = int(config.get('CROP_WIDTH', 0) or 0)
        self.crop_height = int(config.get('CROP_HEIGHT', 0) or 0)
        self.smart_crop_padding = float(config.get('SMART_CROP_PADDING', 0.15))
        # Rolling hourly frame cap for API cost control; 0 = disabled.
        self._frame_cap_per_hour = int(config.get('GEMINI_FRAMES_PER_HOUR_CAP', 200) or 0)
        self._frame_cap_records: list[tuple[float, int]] = []
        self._frame_cap_lock = threading.Lock()

    def _load_system_prompt_template(self) -> str:
        if self._prompt_template is not None:
            return self._prompt_template
        prompt_file = (self.config.get('MULTI_CAM_SYSTEM_PROMPT_FILE') or '').strip()
        if prompt_file and os.path.isfile(prompt_file):
            try:
                with open(prompt_file, 'r', encoding='utf-8') as f:
                    self._prompt_template = f.read()
                return self._prompt_template
            except Exception as e:
                logger.warning("Could not read prompt file %s: %s", prompt_file, e)
        default_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'ai_analyzer_system_prompt.txt'
        )
        if os.path.isfile(default_path):
            with open(default_path, 'r', encoding='utf-8') as f:
                self._prompt_template = f.read()
        else:
            self._prompt_template = (
                "You are a security recap assistant. Describe what happens in the provided "
                "camera frames. Output a JSON object with: title, scene, shortSummary, confidence, potential_threat_level."
            )
        return self._prompt_template

    def _build_system_prompt(
        self,
        image_count: int,
        camera_list: str,
        first_image_number: int,
        last_image_number: int,
        activity_start_str: str,
        duration_str: str,
        zones_str: str,
        labels_str: str,
    ) -> str:
        template = self._load_system_prompt_template()
        return (
            template.replace('{image_count}', str(image_count))
            .replace('{global_event_camera_list}', camera_list)
            .replace('{first_image_number}', str(first_image_number))
            .replace('{last_image_number}', str(last_image_number))
            .replace('{current day and time}', activity_start_str)
            .replace('{duration of the event}', duration_str)
            .replace('{list of zones in global event, dont repeat zones}', zones_str)
            .replace('{list of labels and sub_labels tracked in scene}', labels_str)
        )

    def _center_crop(self, frame: Any, target_w: int, target_h: int) -> Any:
        """Crop frame to target_w x target_h centered. Resize if crop larger than frame."""
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

    def _smart_crop(
        self,
        frame: Any,
        box: Sequence[float],
        target_w: int,
        target_h: int,
        padding: float | None = None,
    ) -> Any:
        """
        Crop frame centered on the bounding box with padding for visual context.
        box is [ymin, xmin, ymax, xmax] normalized 0-1. Expands box by padding (e.g. 0.15)
        so the model sees subject plus immediate environment.
        """
        h, w = frame.shape[:2]
        if target_w <= 0 or target_h <= 0:
            return frame
        if not box or len(box) != 4:
            return self._center_crop(frame, target_w, target_h)
        pad = padding if padding is not None else self.smart_crop_padding
        ymin, xmin, ymax, xmax = float(box[0]), float(box[1]), float(box[2]), float(box[3])
        # Expand by padding (fraction of box size or frame)
        bw, bh = xmax - xmin, ymax - ymin
        dx = max(bw * pad, 0.05)
        dy = max(bh * pad, 0.05)
        ymin = max(0.0, ymin - dy)
        xmin = max(0.0, xmin - dx)
        ymax = min(1.0, ymax + dy)
        xmax = min(1.0, xmax + dx)
        # To pixel coords
        py1, px1 = int(ymin * h), int(xmin * w)
        py2, px2 = int(ymax * h), int(xmax * w)
        cy = (py1 + py2) // 2
        cx = (px1 + px2) // 2
        x1 = cx - target_w // 2
        y1 = cy - target_h // 2
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
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return self._center_crop(frame, target_w, target_h)
        if crop.shape[1] != target_w or crop.shape[0] != target_h:
            crop = cv2.resize(crop, (target_w, target_h))
        return crop

    def _frame_to_base64_url(self, frame: Any) -> str:
        """Encode a BGR frame as JPEG base64 data URL."""
        h, w = frame.shape[:2]
        if w > FRAME_MAX_WIDTH:
            scale = FRAME_MAX_WIDTH / w
            frame = cv2.resize(frame, (FRAME_MAX_WIDTH, int(h * scale)))
        _, buf = cv2.imencode('.jpg', frame)
        b64 = base64.b64encode(buf.tobytes()).decode('ascii')
        return f"data:image/jpeg;base64,{b64}"

    def _frame_cap_stats_and_check(self, frame_count: int) -> tuple[bool, int, int]:
        """
        Compute rolling-window stats, log them, and check if this request would exceed the cap.
        Returns (allowed, current_frames_in_window, requests_in_window).
        Caller must hold _frame_cap_lock when mutating _frame_cap_records; this method acquires it.
        """
        if self._frame_cap_per_hour <= 0:
            logger.info(
                "Gemini API rate: cap=disabled frames_this_request=%d",
                frame_count,
            )
            return True, 0, 0
        now = time.time()
        window_sec = 3600
        with self._frame_cap_lock:
            self._frame_cap_records = [(ts, n) for ts, n in self._frame_cap_records if now - ts <= window_sec]
            current_frames = sum(n for _, n in self._frame_cap_records)
            requests_in_window = len(self._frame_cap_records)
            would_exceed = (current_frames + frame_count) > self._frame_cap_per_hour
            status = "blocked" if would_exceed else "allowed"
            logger.info(
                "Gemini API rate: current_frames=%d cap=%d requests_in_window=%d "
                "frames_this_request=%d status=%s blocked=%s",
                current_frames,
                self._frame_cap_per_hour,
                requests_in_window,
                frame_count,
                status,
                would_exceed,
            )
            return not would_exceed, current_frames, requests_in_window

    def _record_frames_sent(self, frame_count: int) -> None:
        """Record a successful send for the rolling cap. Call with _frame_cap_lock held only from send_to_proxy."""
        if self._frame_cap_per_hour <= 0 or frame_count <= 0:
            return
        with self._frame_cap_lock:
            self._frame_cap_records.append((time.time(), frame_count))

    def send_to_proxy(
        self,
        system_prompt: str,
        image_buffers: list[Any],
    ) -> dict[str, Any] | None:
        """
        POST system prompt and images to Gemini proxy (OpenAI-compatible).
        image_buffers: list of numpy arrays (BGR, from cv2).
        Returns parsed JSON with title, shortSummary, scene, confidence, potential_threat_level, or None on failure.
        """
        if not self._proxy_url or not self._api_key:
            logger.warning("Gemini proxy_url or api_key not configured")
            return None
        # Rolling cap check and stats log (current rate, status, blocked)
        allowed, _, _ = self._frame_cap_stats_and_check(len(image_buffers))
        if not allowed:
            logger.warning(
                "Gemini API call skipped: hourly frame cap reached (cap=%d). Not sending request.",
                self._frame_cap_per_hour,
            )
            return None
        url = f"{self._proxy_url}/v1/chat/completions"
        user_content = [{"type": "text", "text": "Analyze these security camera frames and respond with the requested JSON."}]
        for frame in image_buffers:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": self._frame_to_base64_url(frame)}
            })
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": self._temperature,
            "top_p": self._top_p,
            "frequency_penalty": self._frequency_penalty,
            "presence_penalty": self._presence_penalty,
        }
        base_headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        for attempt in range(2):
            headers = dict(base_headers)
            if attempt == 1:
                headers["Accept-Encoding"] = "identity"
                headers["Connection"] = "close"
            try:
                resp = requests.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=60,
                )
                resp.raise_for_status()
                data = resp.json()
                content = (data.get("choices") or [{}])[0].get("message", {}).get("content")
                if not content or not content.strip():
                    logger.warning("Empty response content from proxy")
                    return None
                # Strip markdown code blocks if present
                raw = content.strip()
                if raw.startswith("```"):
                    lines = raw.split("\n")
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines and lines[-1].strip() == "```":
                        lines = lines[:-1]
                    raw = "\n".join(lines)
                result = json.loads(raw)
                self._record_frames_sent(len(image_buffers))
                return result
            except (requests.exceptions.ChunkedEncodingError, urllib3.exceptions.ProtocolError) as e:
                logger.warning(
                    "Proxy attempt %s/2 failed with ChunkedEncodingError: %s. Retrying with 'Connection: close'...",
                    attempt + 1, e,
                )
                continue
            except json.JSONDecodeError as e:
                logger.exception("Failed to parse proxy JSON: %s", e)
                if attempt == 1:
                    return None
                continue
            except Exception as e:
                _log_proxy_failure(url, attempt + 1, e)
                if attempt == 1:
                    return None
                continue
        return None

    def send_text_prompt(self, system_prompt: str, user_prompt: str) -> str | None:
        """
        POST text-only messages to Gemini proxy (OpenAI-compatible). No images.
        Uses same GEMINI_PROXY_URL and GEMINI_API_KEY as send_to_proxy.
        Returns raw response content (e.g. Markdown) or None on failure.
        """
        if not self._proxy_url or not self._api_key:
            logger.warning("Gemini proxy_url or api_key not configured")
            return None
        url = f"{self._proxy_url}/v1/chat/completions"
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self._temperature,
            "top_p": self._top_p,
            "frequency_penalty": self._frequency_penalty,
            "presence_penalty": self._presence_penalty,
        }
        base_headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        for attempt in range(2):
            headers = dict(base_headers)
            if attempt == 1:
                headers["Accept-Encoding"] = "identity"
                headers["Connection"] = "close"
            try:
                resp = requests.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=60,
                )
                resp.raise_for_status()
                data = resp.json()
                content = (data.get("choices") or [{}])[0].get("message", {}).get("content")
                if not content or not str(content).strip():
                    logger.warning("Empty response content from proxy (text prompt)")
                    return None
                return str(content).strip()
            except (requests.exceptions.ChunkedEncodingError, urllib3.exceptions.ProtocolError) as e:
                logger.warning(
                    "Proxy attempt %s/2 failed with ChunkedEncodingError: %s. Retrying with 'Connection: close'...",
                    attempt + 1, e,
                )
                continue
            except Exception as e:
                _log_proxy_failure(url, attempt + 1, e)
                if attempt == 1:
                    return None
                continue
        return None

    def analyze_multi_clip_ce(
        self,
        ce_id: str,
        ce_folder_path: str,
        ce_start_time: float = 0,
        primary_camera: str | None = None,
    ) -> dict[str, Any] | None:
        """
        Target-centric multi-clip analysis for consolidated events.
        Extracts frames from full clips via object detection, sends to Gemini, saves to CE root.
        Handles both single-camera and multi-camera CEs (len(cameras) == 1 or more); same pipeline.
        primary_camera: optional camera that initiated the event (first-camera bias in selection).
        """
        if not self._enabled:
            logger.debug("Gemini analysis disabled, skipping CE %s", ce_id)
            return None
        if not self._proxy_url or not self._api_key:
            logger.debug("Gemini proxy not configured, skipping CE %s", ce_id)
            return None
        try:
            if not os.path.isdir(ce_folder_path):
                logger.warning("CE folder not found: %s", ce_folder_path)
                return None

            max_frames_sec = float(self.config.get("MAX_MULTI_CAM_FRAMES_SEC", 1))
            max_frames_min = int(self.config.get("MAX_MULTI_CAM_FRAMES_MIN", 60))
            logger.info("Creating frame timeline")
            frames_raw = extract_target_centric_frames(
                ce_folder_path,
                max_frames_sec=max_frames_sec,
                max_frames_min=max_frames_min,
                crop_width=self.crop_width,
                crop_height=self.crop_height,
                tracking_target_frame_percent=int(self.config.get("TRACKING_TARGET_FRAME_PERCENT", 40)),
                primary_camera=primary_camera,
                decode_second_camera_cpu_only=bool(self.config.get("DECODE_SECOND_CAMERA_CPU_ONLY", False)),
                config=self.config,
                camera_timeline_analysis_multiplier=float(self.config.get("CAMERA_TIMELINE_ANALYSIS_MULTIPLIER", 2)),
                camera_timeline_ema_alpha=float(self.config.get("CAMERA_TIMELINE_EMA_ALPHA", 0.4)),
                camera_timeline_primary_bias_multiplier=float(self.config.get("CAMERA_TIMELINE_PRIMARY_BIAS_MULTIPLIER", 1.2)),
                camera_switch_min_segment_frames=int(self.config.get("CAMERA_SWITCH_MIN_SEGMENT_FRAMES", 5)),
                camera_switch_hysteresis_margin=float(self.config.get("CAMERA_SWITCH_HYSTERESIS_MARGIN", 1.15)),
                camera_timeline_final_yolo_drop_no_person=bool(self.config.get("CAMERA_TIMELINE_FINAL_YOLO_DROP_NO_PERSON", False)),
            )
            if not frames_raw:
                logger.warning("No frames extracted from CE %s", ce_id)
                return None

            # Add timestamp overlay; keep ExtractedFrame with overlay drawn on .frame
            image_count = len(frames_raw)
            cameras_str = ", ".join(sorted({ef.camera for ef in frames_raw}))
            for i, ef in enumerate(frames_raw):
                ts = ce_start_time + ef.timestamp_sec if ce_start_time > 0 else ef.timestamp_sec
                time_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
                person_area = ef.metadata.get("person_area") if self.config.get("PERSON_AREA_DEBUG") else None
                ef.frame = crop_utils.draw_timestamp_overlay(
                    ef.frame, time_str, ef.camera, i + 1, image_count, person_area=person_area
                )

            # Build payload and send to proxy first (disk write deferred so API call starts earlier).
            frames = [ef.frame for ef in frames_raw]
            first_ts = frames_raw[0].timestamp_sec if frames_raw else 0
            activity_start_str = (
                datetime.fromtimestamp(ce_start_time + first_ts).strftime("%Y-%m-%d %H:%M:%S")
                if ce_start_time > 0 else datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
            logger.info("Preparing system prompt")
            system_prompt = self._build_system_prompt(
                image_count=image_count,
                camera_list=cameras_str,
                first_image_number=1,
                last_image_number=image_count,
                activity_start_str=activity_start_str,
                duration_str="multi-camera event",
                zones_str="none recorded",
                labels_str="(none recorded)",
            )
            logger.info("Sending to proxy")
            result = self.send_to_proxy(system_prompt, frames)
            if result and isinstance(result, dict):
                out_path = os.path.join(ce_folder_path, "analysis_result.json")
                try:
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(result, f, indent=2, ensure_ascii=False)
                    logger.debug("Saved analysis_result.json for CE %s to %s", ce_id, out_path)
                except OSError as e:
                    logger.warning("Failed to write analysis_result.json for CE %s: %s", ce_id, e)
                # Write ai_frame_analysis after API return (moves disk I/O off critical path).
                logger.info("Writing frames to disk")
                save_frames = bool(self.config.get("SAVE_AI_FRAMES", True))
                create_zip = bool(self.config.get("CREATE_AI_ANALYSIS_ZIP", True))
                write_ai_frame_analysis_multi_cam(
                    ce_folder_path,
                    frames_raw,
                    write_manifest=True,
                    create_zip_flag=create_zip,
                    save_frames=save_frames,
                )
            return result
        except Exception as e:
            logger.exception(
                "CE analysis failed for %s (%s): %s",
                ce_id,
                ce_folder_path,
                e,
            )
            return None

    def build_multi_cam_payload_for_preview(
        self,
        ce_folder_path: str,
        ce_start_time: float = 0,
        log_messages: list[str] | None = None,
    ) -> tuple[tuple[str, list[Any], list[str]] | None, str | None]:
        """
        Build the same payload as analyze_multi_clip_ce (system prompt + user content with frames)
        and write frame files to ce_folder_path, but do not call send_to_proxy.

        Used by the event_test orchestrator to produce system_prompt.txt and ai_request.html
        with download links to the written frame files.

        Returns (result, error_message). result is (system_prompt, user_content, frame_relative_paths)
        or None; error_message is a short string when something failed (no frames, exception).
        When log_messages is provided, appends human-readable progress strings for SSE.
        """
        if not os.path.isdir(ce_folder_path):
            logger.warning("CE folder not found: %s", ce_folder_path)
            return (None, "CE folder not found")
        try:
            def _log(msg: str) -> None:
                if log_messages is not None:
                    log_messages.append(msg)

            # Frame selection: require detection sidecars for every camera
            camera_subdirs: list[str] = []
            try:
                with os.scandir(ce_folder_path) as it:
                    camera_subdirs = [e.name for e in it if e.is_dir() and not e.name.startswith(".")]
            except OSError:
                pass
            all_have_sidecar = True
            missing: list[str] = []
            for cam in camera_subdirs:
                sidecar_path = os.path.join(ce_folder_path, cam, _DETECTION_SIDECAR_FILENAME)
                if not os.path.isfile(sidecar_path):
                    all_have_sidecar = False
                    missing.append(cam)
            if all_have_sidecar and camera_subdirs:
                _log("Frame selection: using detection sidecars (picks best camera per moment from pre-computed object detections).")
            else:
                part = f" (missing for: {', '.join(missing)})" if missing else ""
                _log(f"Frame selection: detection sidecars missing; skipping multi-clip extraction.{part}")

            max_frames_sec = float(self.config.get("MAX_MULTI_CAM_FRAMES_SEC", 1))
            max_frames_min = int(self.config.get("MAX_MULTI_CAM_FRAMES_MIN", 60))
            _log("Creating frame timeline")
            frames_raw = extract_target_centric_frames(
                ce_folder_path,
                max_frames_sec=max_frames_sec,
                max_frames_min=max_frames_min,
                crop_width=self.crop_width,
                crop_height=self.crop_height,
                tracking_target_frame_percent=int(self.config.get("TRACKING_TARGET_FRAME_PERCENT", 40)),
                primary_camera=None,
                decode_second_camera_cpu_only=bool(self.config.get("DECODE_SECOND_CAMERA_CPU_ONLY", False)),
                log_callback=_log if log_messages is not None else None,
                config=self.config,
                camera_timeline_analysis_multiplier=float(self.config.get("CAMERA_TIMELINE_ANALYSIS_MULTIPLIER", 2)),
                camera_timeline_ema_alpha=float(self.config.get("CAMERA_TIMELINE_EMA_ALPHA", 0.4)),
                camera_timeline_primary_bias_multiplier=float(self.config.get("CAMERA_TIMELINE_PRIMARY_BIAS_MULTIPLIER", 1.2)),
                camera_switch_min_segment_frames=int(self.config.get("CAMERA_SWITCH_MIN_SEGMENT_FRAMES", 5)),
                camera_switch_hysteresis_margin=float(self.config.get("CAMERA_SWITCH_HYSTERESIS_MARGIN", 1.15)),
                camera_timeline_final_yolo_drop_no_person=bool(self.config.get("CAMERA_TIMELINE_FINAL_YOLO_DROP_NO_PERSON", False)),
            )
            if not frames_raw:
                logger.warning("No frames extracted from CE folder %s", ce_folder_path)
                return (None, "No frames extracted (check clips and sidecars).")

            cameras_from_frames = ", ".join(sorted({ef.camera for ef in frames_raw}))
            _log(f"Extracted {len(frames_raw)} frames from clips ({cameras_from_frames}).")
            _log(f"Adding timestamps to frames ({cameras_from_frames}).")

            image_count = len(frames_raw)
            cameras_str = ", ".join(sorted({ef.camera for ef in frames_raw}))
            for i, ef in enumerate(frames_raw):
                ts = ce_start_time + ef.timestamp_sec if ce_start_time > 0 else ef.timestamp_sec
                time_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
                person_area = ef.metadata.get("person_area") if self.config.get("PERSON_AREA_DEBUG") else None
                ef.frame = crop_utils.draw_timestamp_overlay(
                    ef.frame, time_str, ef.camera, i + 1, image_count, person_area=person_area
                )

            _log(f"Organizing timeline (saving {len(frames_raw)} frames).")
            save_frames = bool(self.config.get("SAVE_AI_FRAMES", True))
            write_ai_frame_analysis_multi_cam(
                ce_folder_path,
                frames_raw,
                write_manifest=True,
                create_zip_flag=False,
                save_frames=save_frames,
            )

            frames = [ef.frame for ef in frames_raw]
            first_ts = frames_raw[0].timestamp_sec if frames_raw else 0
            activity_start_str = (
                datetime.fromtimestamp(ce_start_time + first_ts).strftime("%Y-%m-%d %H:%M:%S")
                if ce_start_time > 0 else datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
            _log("Preparing system prompt")
            system_prompt = self._build_system_prompt(
                image_count=image_count,
                camera_list=cameras_str,
                first_image_number=1,
                last_image_number=image_count,
                activity_start_str=activity_start_str,
                duration_str="multi-camera event",
                zones_str="none recorded",
                labels_str="(none recorded)",
            )
            _log("Building payload for preview")
            user_content: list[Any] = [
                {"type": "text", "text": "Analyze these security camera frames and respond with the requested JSON."}
            ]
            for frame in frames:
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": self._frame_to_base64_url(frame)},
                })

            frame_relative_paths: list[str] = []
            frames_dir = os.path.join(ce_folder_path, "ai_frame_analysis", "frames")
            if os.path.isdir(frames_dir):
                for name in sorted(os.listdir(frames_dir)):
                    if name.startswith("frame_") and name.endswith(".jpg"):
                        frame_relative_paths.append(os.path.join("ai_frame_analysis", "frames", name))

            return ((system_prompt, user_content, frame_relative_paths), None)
        except Exception as e:
            logger.exception("build_multi_cam_payload_for_preview failed for %s: %s", ce_folder_path, e)
            return (None, str(e))
