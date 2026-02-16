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
from datetime import datetime
from typing import Optional, List, Dict, Any

import cv2
import requests

logger = logging.getLogger('frigate-buffer')

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
        self._prompt_template: Optional[str] = None
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

    def _extract_frames(
        self,
        clip_path: str,
        event_start_ts: float = 0,
        event_end_ts: float = 0,
    ) -> List[Any]:
        """Extract frames from video. With event_start_ts > 0, seeks past pre-capture buffer and limits to event segment.
        Uses grayscale frame differencing for motion-aware selection when MOTION_THRESHOLD_PX > 0; first and last frame
        are always kept. Optional center crop when CROP_WIDTH/CROP_HEIGHT set. Returns list of numpy arrays (BGR)."""
        frames = []
        cap = cv2.VideoCapture(clip_path)
        if not cap.isOpened():
            logger.warning(f"Could not open video: {clip_path}")
            return frames
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 1.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

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

                # Collect candidates (frame_idx, frame, motion_score) with grayscale differencing
                use_motion = self.motion_threshold_px > 0
                candidates = []
                prev_gray = None

                frame_idx = start_frame
                while frame_idx <= end_frame:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if self.crop_width > 0 and self.crop_height > 0:
                        frame = self._center_crop(frame, self.crop_width, self.crop_height)
                    motion_score = 0.0
                    if use_motion:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        if prev_gray is not None:
                            diff = cv2.absdiff(prev_gray, gray)
                            motion_score = float(cv2.sumElems(diff)[0])
                        prev_gray = gray
                    candidates.append((frame_idx, frame, motion_score))
                    frame_idx += step

                if not candidates:
                    return frames
                # Always keep first and last for context (before/after for the model)
                first_idx, first_frame, _ = candidates[0]
                last_idx, last_frame, _ = candidates[-1]
                if len(candidates) <= self._max_frames:
                    frames = [c[1] for c in candidates]
                else:
                    middle = candidates[1:-1]
                    n_keep = self._max_frames - 2
                    # Keep middle frames with highest motion, then sort by frame index
                    middle_sorted = sorted(middle, key=lambda c: c[2], reverse=True)[:n_keep]
                    middle_sorted.sort(key=lambda c: c[0])
                    frames = [first_frame] + [c[1] for c in middle_sorted] + [last_frame]
                return frames

            if total_frames <= 0:
                while len(frames) < self._max_frames:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if self.crop_width > 0 and self.crop_height > 0:
                        frame = self._center_crop(frame, self.crop_width, self.crop_height)
                    if len(frames) % max(1, int(fps * FRAME_SAMPLE_INTERVAL_SEC)) == 0:
                        frames.append(frame)
                    if len(frames) >= self._max_frames:
                        break
                return frames
            step = max(1, int(fps * FRAME_SAMPLE_INTERVAL_SEC))
            frame_idx = 0
            while frame_idx < total_frames and len(frames) < self._max_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    break
                if self.crop_width > 0 and self.crop_height > 0:
                    frame = self._center_crop(frame, self.crop_width, self.crop_height)
                frames.append(frame)
                frame_idx += step
        finally:
            cap.release()
        return frames

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
        image_buffers: List[Any],
    ) -> Optional[Dict[str, Any]]:
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
        try:
            resp = requests.post(
                url,
                json=payload,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
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
        except requests.exceptions.RequestException as e:
            logger.exception("Proxy request failed: %s", e)
            return None
        except json.JSONDecodeError as e:
            logger.exception("Failed to parse proxy JSON: %s", e)
            return None

    def analyze_clip(
        self,
        event_id: str,
        clip_path: str,
        event_start_ts: float = 0,
        event_end_ts: float = 0,
    ) -> Optional[Dict[str, Any]]:
        """
        Extract frames from clip_path, call proxy, return analysis metadata.
        When event_start_ts > 0, seeks past pre-capture buffer and limits to event segment.
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
            frames = self._extract_frames(clip_path, event_start_ts, event_end_ts)
            if not frames:
                logger.warning("No frames extracted from %s", clip_path)
                return None
            camera = self._camera_from_clip_path(clip_path)
            image_count = len(frames)
            activity_start_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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
            return self.send_to_proxy(system_prompt, frames)
        except Exception as e:
            logger.exception("Analyze clip failed for %s: %s", event_id, e)
            return None
