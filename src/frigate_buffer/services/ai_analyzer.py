"""
AI Analyzer Service - Gemini proxy integration for clip analysis.

Extracts frames from a video clip, builds a system prompt, sends to an
OpenAI-compatible Gemini proxy, and returns the analysis metadata to the caller.
Does NOT publish to MQTT; the orchestrator persists results and notifies HA.
"""

import bisect
import os
import json
import base64
import logging
from datetime import datetime
from typing import Any, Sequence

from frigate_buffer.models import FrameMetadata
from frigate_buffer.managers.file import write_ai_frame_analysis_single_cam

from frigate_buffer.services import crop_utils

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

# Default max frames to send to proxy (cap token/image cost)
DEFAULT_MAX_FRAMES = 20
# Frame sampling interval in seconds when extracting from video
FRAME_SAMPLE_INTERVAL_SEC = 0.5
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
        self._max_frames = int(config.get('FINAL_REVIEW_IMAGE_COUNT', DEFAULT_MAX_FRAMES) or DEFAULT_MAX_FRAMES)
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
        self.motion_threshold_px = int(config.get('MOTION_THRESHOLD_PX', 0))
        self.crop_width = int(config.get('CROP_WIDTH', 0) or 0)
        self.crop_height = int(config.get('CROP_HEIGHT', 0) or 0)
        self.smart_crop_padding = float(config.get('SMART_CROP_PADDING', 0.15))
        self.motion_crop_min_area_fraction = float(config.get('MOTION_CROP_MIN_AREA_FRACTION', 0.001))
        self.motion_crop_min_px = int(config.get('MOTION_CROP_MIN_PX', 500))

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

    def _camera_from_clip_path(self, clip_path: str) -> str:
        """Derive camera name from clip path (e.g. .../Doorbell/clip.mp4 -> Doorbell)."""
        try:
            parent = os.path.basename(os.path.dirname(os.path.abspath(clip_path)))
            if parent and parent != 'clip.mp4':
                return parent
        except Exception:
            pass
        return "unknown"

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

    def _extract_frames(
        self,
        clip_path: str,
        event_start_ts: float = 0,
        event_end_ts: float = 0,
        frame_metadata: Sequence[FrameMetadata] | None = None,
    ) -> list[tuple]:
        """Extract frames from video. With event_start_ts > 0, seeks past pre-capture buffer and limits to event segment.
        Uses CV-based motion crop (crop_utils.motion_crop) when CROP_WIDTH/CROP_HEIGHT set; otherwise center crop.
        Uses grayscale frame differencing for motion-aware selection when MOTION_THRESHOLD_PX > 0; first and last frame
        are always kept. When frame_metadata is provided, prioritizes by score*area.
        Returns list of (frame, frame_time_sec) for overlay in analyze_clip."""
        result: list[tuple] = []
        cap = cv2.VideoCapture(clip_path)
        if not cap.isOpened():
            logger.warning("Could not open video: %s", clip_path)
            return result
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 1.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            use_meta = bool(frame_metadata)
            use_cv_crop = self.crop_width > 0 and self.crop_height > 0

            if event_start_ts > 0 and total_frames > 0:
                start_frame = int(fps * self.buffer_offset)
                event_duration_sec = (event_end_ts - event_start_ts) if event_end_ts > event_start_ts else 0
                end_frame = (
                    start_frame + int(fps * event_duration_sec)
                    if event_duration_sec > 0
                    else total_frames
                )
                end_frame = min(end_frame, total_frames)
                start_frame = min(start_frame, end_frame)
                step = max(1, int(fps / self.max_frames_sec)) if self.max_frames_sec > 0 else 1

                use_motion = self.motion_threshold_px > 0
                candidates: list[tuple] = []
                prev_gray = None
                sorted_meta: list[Any] = []
                times: list[float] = []
                if use_meta and frame_metadata:
                    sorted_meta = sorted(frame_metadata, key=lambda m: m.frame_time)
                    times = [m.frame_time for m in sorted_meta]

                frame_idx = start_frame
                while frame_idx <= end_frame:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_time_approx = event_start_ts + (frame_idx - start_frame) / fps if fps else event_start_ts
                    meta = None
                    if sorted_meta and times:
                        idx = bisect.bisect_left(times, frame_time_approx)
                        closest = sorted_meta[min(idx, len(sorted_meta) - 1)]
                        if idx > 0 and abs(sorted_meta[idx - 1].frame_time - frame_time_approx) < abs(closest.frame_time - frame_time_approx):
                            closest = sorted_meta[idx - 1]
                        if abs(closest.frame_time - frame_time_approx) < 2.0:
                            meta = closest

                    motion_score = 0.0
                    if use_cv_crop:
                        frame, next_gray = crop_utils.motion_crop(
                            frame,
                            prev_gray,
                            self.crop_width,
                            self.crop_height,
                            min_area_fraction=self.motion_crop_min_area_fraction,
                            min_area_px=self.motion_crop_min_px,
                        )
                        if use_motion and prev_gray is not None:
                            motion_score = float(cv2.sumElems(cv2.absdiff(prev_gray, next_gray))[0])
                        prev_gray = next_gray
                    else:
                        if self.crop_width > 0 and self.crop_height > 0:
                            frame = self._center_crop(frame, self.crop_width, self.crop_height)
                        if use_motion:
                            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            if prev_gray is not None:
                                motion_score = float(cv2.sumElems(cv2.absdiff(prev_gray, gray))[0])
                            prev_gray = gray

                    priority = (meta.score * meta.area) if meta else motion_score
                    candidates.append((frame_idx, frame, frame_time_approx, motion_score, meta, priority))
                    frame_idx += step

                if not candidates:
                    return result
                if len(candidates) <= self._max_frames:
                    result = [(c[1], c[2]) for c in candidates]
                else:
                    middle = candidates[1:-1]
                    n_keep = self._max_frames - 2
                    middle_sorted = sorted(middle, key=lambda c: c[5], reverse=True)[:n_keep]
                    middle_sorted.sort(key=lambda c: c[0])
                    result = [(candidates[0][1], candidates[0][2])] + [(c[1], c[2]) for c in middle_sorted] + [(candidates[-1][1], candidates[-1][2])]
                return result

            if total_frames <= 0:
                idx = 0
                while len(result) < self._max_frames:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if use_cv_crop:
                        frame, _ = crop_utils.motion_crop(
                            frame, None, self.crop_width, self.crop_height,
                            min_area_fraction=self.motion_crop_min_area_fraction,
                            min_area_px=self.motion_crop_min_px,
                        )
                    elif self.crop_width > 0 and self.crop_height > 0:
                        frame = self._center_crop(frame, self.crop_width, self.crop_height)
                    if len(result) % max(1, int(fps * FRAME_SAMPLE_INTERVAL_SEC)) == 0:
                        result.append((frame, event_start_ts + idx / fps if fps else event_start_ts))
                    if len(result) >= self._max_frames:
                        break
                    idx += 1
                return result
            step = max(1, int(fps * FRAME_SAMPLE_INTERVAL_SEC))
            frame_idx = 0
            while frame_idx < total_frames and len(result) < self._max_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    break
                if use_cv_crop:
                    frame, _ = crop_utils.motion_crop(
                        frame, None, self.crop_width, self.crop_height,
                        min_area_fraction=self.motion_crop_min_area_fraction,
                        min_area_px=self.motion_crop_min_px,
                    )
                elif self.crop_width > 0 and self.crop_height > 0:
                    frame = self._center_crop(frame, self.crop_width, self.crop_height)
                frame_time = (event_start_ts + frame_idx / fps) if (event_start_ts > 0 and fps) else (frame_idx / fps if fps else 0)
                result.append((frame, frame_time))
                frame_idx += step
        finally:
            cap.release()
        return result

    def _frame_to_base64_url(self, frame: Any) -> str:
        """Encode a BGR frame as JPEG base64 data URL."""
        h, w = frame.shape[:2]
        if w > FRAME_MAX_WIDTH:
            scale = FRAME_MAX_WIDTH / w
            frame = cv2.resize(frame, (FRAME_MAX_WIDTH, int(h * scale)))
        _, buf = cv2.imencode('.jpg', frame)
        b64 = base64.b64encode(buf.tobytes()).decode('ascii')
        return f"data:image/jpeg;base64,{b64}"

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
                return json.loads(raw)
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

    def _save_analysis_result(self, event_id: str, clip_path: str, result: dict[str, Any]) -> None:
        """
        Save the analysis result as analysis_result.json in the same directory as the clip.
        Required for Step 7 (Daily Reporter). Logs a warning if shortSummary, title, or
        potential_threat_level are missing but still saves the full dict.
        """
        required = ("shortSummary", "title", "potential_threat_level")
        missing = [k for k in required if k not in result]
        if missing:
            logger.warning("Analysis result for %s missing fields %s; saving anyway", event_id, missing)
        event_dir = os.path.dirname(os.path.abspath(clip_path))
        out_path = os.path.join(event_dir, "analysis_result.json")
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            logger.debug("Saved analysis_result.json for %s to %s", event_id, out_path)
        except OSError as e:
            logger.warning("Failed to write analysis_result.json for %s: %s", event_id, e)

    def analyze_clip(
        self,
        event_id: str,
        clip_path: str,
        event_start_ts: float = 0,
        event_end_ts: float = 0,
        frame_metadata: Sequence[FrameMetadata] | None = None,
    ) -> dict[str, Any] | None:
        """
        Extract frames from clip_path, call proxy, return analysis metadata.
        When event_start_ts > 0, seeks past pre-capture buffer and limits to event segment.
        When frame_metadata is provided, uses smart crop and score*area prioritization.
        Called from orchestrator (e.g. in a background thread). Returns None if disabled, path missing, or proxy failure.
        Does NOT publish to MQTT; caller is responsible for persisting and notifying.
        """
        if not self._enabled:
            logger.debug("Gemini analysis disabled, skipping %s", event_id)
            return None
        if not self._proxy_url or not self._api_key:
            logger.debug("Gemini proxy not configured, skipping %s", event_id)
            return None
        if not os.path.isfile(clip_path):
            logger.warning("Clip not found: %s", clip_path)
            return None
        try:
            frames_with_time = self._extract_frames(
                clip_path, event_start_ts, event_end_ts,
                frame_metadata=frame_metadata,
            )
            if not frames_with_time:
                logger.warning("No frames extracted from %s", clip_path)
                return None
            camera = self._camera_from_clip_path(clip_path)
            image_count = len(frames_with_time)
            for i, (frame, frame_time_sec) in enumerate(frames_with_time):
                time_str = datetime.fromtimestamp(frame_time_sec).strftime("%Y-%m-%d %H:%M:%S")
                crop_utils.draw_timestamp_overlay(
                    frame, time_str, camera, i + 1, image_count
                )
            frames = [f for f, _ in frames_with_time]
            activity_start_str = (
                datetime.fromtimestamp(frames_with_time[0][1]).strftime("%Y-%m-%d %H:%M:%S")
                if frames_with_time else datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
            duration_str = "unknown"
            system_prompt = self._build_system_prompt(
                image_count=image_count,
                camera_list=camera,
                first_image_number=1,
                last_image_number=image_count,
                activity_start_str=activity_start_str,
                duration_str=duration_str,
                zones_str="none recorded",
                labels_str="(none recorded)",
            )
            result = self.send_to_proxy(system_prompt, frames)
            if result and isinstance(result, dict):
                self._save_analysis_result(event_id, clip_path, result)
                event_dir = os.path.dirname(os.path.abspath(clip_path))
                save_frames = bool(self.config.get("SAVE_AI_FRAMES", True))
                create_zip = bool(self.config.get("CREATE_AI_ANALYSIS_ZIP", True))
                write_ai_frame_analysis_single_cam(
                    event_dir,
                    frames_with_time,
                    camera,
                    write_manifest=True,
                    create_zip_flag=create_zip,
                    save_frames=save_frames,
                )
            return result
        except Exception as e:
            logger.exception("Analyze clip failed for %s: %s", event_id, e)
            return None
