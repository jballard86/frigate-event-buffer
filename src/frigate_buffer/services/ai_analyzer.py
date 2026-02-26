"""
AI Analyzer Service - Gemini proxy integration for clip analysis.

Extracts frames from a video clip, builds a system prompt, sends to an
OpenAI-compatible Gemini proxy, and returns the analysis metadata to the caller.
Does NOT publish to MQTT; the orchestrator persists results and notifies HA.
"""

import base64
import json
import logging
import os
import threading
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any

# Relative path under CE folder for detection sidecar (must match multi_clip_extractor)
_DETECTION_SIDECAR_FILENAME = "detection.json"

import cv2
import requests

from frigate_buffer.constants import (
    FRAME_MAX_WIDTH,
    GEMINI_PROXY_ANALYSIS_TIMEOUT,
    GEMINI_PROXY_QUICK_TITLE_TIMEOUT,
    is_tensor,
)
from frigate_buffer.managers.file import write_ai_frame_analysis_multi_cam
from frigate_buffer.services import crop_utils
from frigate_buffer.services.gemini_proxy_client import GeminiProxyClient
from frigate_buffer.services.multi_clip_extractor import extract_target_centric_frames

logger = logging.getLogger("frigate-buffer")


class GeminiAnalysisService:
    """Analyzes video clips via Gemini proxy; returns metadata to caller."""

    def __init__(self, config: dict):
        self.config = config
        gemini = config.get("GEMINI") or {}
        # Prefer flat keys; fallback to nested. load_config applies env for API key.
        self._proxy_url = (
            (config.get("GEMINI_PROXY_URL") or gemini.get("proxy_url") or "")
            .strip()
            .rstrip("/")
        )
        self._api_key = gemini.get("api_key") or ""
        self._model = (
            config.get("GEMINI_PROXY_MODEL")
            or gemini.get("model")
            or "gemini-2.5-flash-lite"
        )
        self._enabled = bool(gemini.get("enabled", False))
        self._prompt_template: str | None = None
        # Proxy tuning (flat keys)
        self._temperature = float(config.get("GEMINI_PROXY_TEMPERATURE", 0.3))
        self._top_p = float(config.get("GEMINI_PROXY_TOP_P", 1))
        self._frequency_penalty = float(config.get("GEMINI_PROXY_FREQUENCY_PENALTY", 0))
        self._presence_penalty = float(config.get("GEMINI_PROXY_PRESENCE_PENALTY", 0))
        # Smart seeking and multi-cam
        self.buffer_offset = config.get("EXPORT_BUFFER_BEFORE", 5)
        self.max_frames_sec = config.get("MAX_MULTI_CAM_FRAMES_SEC", 2)
        self.max_frames_min = config.get("MAX_MULTI_CAM_FRAMES_MIN", 45)
        self.crop_width = int(config.get("CROP_WIDTH", 0) or 0)
        self.crop_height = int(config.get("CROP_HEIGHT", 0) or 0)
        self.motion_threshold_px = int(config.get("MOTION_THRESHOLD_PX", 0) or 0)
        self.smart_crop_padding = float(config.get("SMART_CROP_PADDING", 0.15))
        # Rolling hourly frame cap for API cost control; 0 = disabled.
        self._frame_cap_per_hour = int(
            config.get("GEMINI_FRAMES_PER_HOUR_CAP", 200) or 0
        )
        self._frame_cap_records: list[tuple[float, int]] = []
        self._frame_cap_lock = threading.Lock()
        self._proxy_client = GeminiProxyClient(
            self._proxy_url,
            self._api_key,
            self._model,
            self._temperature,
            self._top_p,
            self._frequency_penalty,
            self._presence_penalty,
        )

    def _load_system_prompt_template(self) -> str:
        if self._prompt_template is not None:
            return self._prompt_template
        prompt_file = (self.config.get("MULTI_CAM_SYSTEM_PROMPT_FILE") or "").strip()
        if prompt_file and os.path.isfile(prompt_file):
            try:
                with open(prompt_file, encoding="utf-8") as f:
                    self._prompt_template = f.read()
                return self._prompt_template
            except Exception as e:
                logger.warning("Could not read prompt file %s: %s", prompt_file, e)
        default_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "ai_analyzer_system_prompt.txt"
        )
        if os.path.isfile(default_path):
            with open(default_path, encoding="utf-8") as f:
                self._prompt_template = f.read()
        else:
            self._prompt_template = (
                "You are a security recap assistant. Describe what happens in the "
                "provided camera frames. Output a JSON object with: title, scene, "
                "shortSummary, confidence, potential_threat_level."
            )
        return self._prompt_template

    def _load_quick_title_prompt(self) -> str:
        """Load quick-title prompt (single image; JSON with title and description)."""
        default_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "quick_title_prompt.txt"
        )
        if os.path.isfile(default_path):
            try:
                with open(default_path, encoding="utf-8") as f:
                    return f.read()
            except Exception as e:
                logger.warning("Could not read quick_title_prompt.txt: %s", e)
        return (
            "You are a security camera alert titler. Return valid JSON only with keys "
            '"title" (max 12 words) and "description" (max 2 sentences). No markdown.'
        )

    def _post_messages(
        self, messages: list[dict[str, Any]], timeout: int
    ) -> requests.Response | None:
        """Delegate to GeminiProxyClient. Returns response or None on failure."""
        return self._proxy_client.post_messages(messages, timeout)

    def _parse_response_content(self, data: dict[str, Any]) -> str | None:
        """Extract text from proxy response (Gemini or OpenAI format). None if empty."""
        content = None
        candidates = data.get("candidates") or []
        if candidates:
            first = candidates[0]
            if isinstance(first, dict):
                content_obj = first.get("content")
                if isinstance(content_obj, dict):
                    for part in content_obj.get("parts") or []:
                        if isinstance(part, dict) and "text" in part:
                            content = part.get("text")
                            break
                    if content is not None:
                        return str(content).strip() or None
        if content is None:
            content = (data.get("choices") or [{}])[0].get("message", {}).get("content")
        if not content or not str(content).strip():
            return None
        return str(content).strip()

    def _send_single_image_get_text(
        self, system_prompt: str, image_bgr: Any
    ) -> str | None:
        """POST one image to proxy and return raw message content (quick-title)."""
        user_content = [
            {
                "type": "text",
                "text": "Describe this image with the requested short title only.",
            },
            {
                "type": "image_url",
                "image_url": {"url": self._frame_to_base64_url(image_bgr)},
            },
        ]
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        resp = self._post_messages(messages, GEMINI_PROXY_QUICK_TITLE_TIMEOUT)
        if resp is None:
            return None
        try:
            data = resp.json()
        except (ValueError, TypeError) as e:
            logger.warning("Failed to parse quick-title response JSON: %s", e)
            return None
        return self._parse_response_content(data)

    def generate_quick_title(
        self, image: Any, camera: str, label: str
    ) -> dict[str, str] | None:
        """
        Send a single cropped/live frame to Gemini proxy; return title and description.
        image: numpy HWC BGR or torch.Tensor BCHW RGB; encoded via _frame_to_base64_url.
        Used for quick-title pipeline shortly after event start.
        Returns a dict with keys "title" and "description", or None on failure or empty.
        """
        if not self._proxy_url or not self._api_key:
            logger.warning("Gemini proxy_url or api_key not configured for quick title")
            return None
        system_prompt = self._load_quick_title_prompt()
        raw = self._send_single_image_get_text(system_prompt, image)
        if not raw:
            return None
        # Strip markdown code blocks if present
        if raw.startswith("```"):
            lines = raw.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            raw = "\n".join(lines)
        raw = raw.strip()
        if not raw:
            return None
        try:
            data = json.loads(raw)
        except (ValueError, TypeError) as e:
            logger.warning("Quick-title response is not valid JSON: %s", e)
            return None
        if not isinstance(data, dict):
            return None
        title = data.get("title")
        description = data.get("description", "")
        if not title or not str(title).strip():
            return None
        return {
            "title": str(title).strip()[:500],
            "description": str(description).strip()[:1000] if description else "",
        }

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
            template.replace("{image_count}", str(image_count))
            .replace("{global_event_camera_list}", camera_list)
            .replace("{first_image_number}", str(first_image_number))
            .replace("{last_image_number}", str(last_image_number))
            .replace("{current day and time}", activity_start_str)
            .replace("{duration of the event}", duration_str)
            .replace("{list of zones in global event, dont repeat zones}", zones_str)
            .replace("{list of labels and sub_labels tracked in scene}", labels_str)
        )

    def _frame_to_base64_url(self, frame: Any) -> str:
        """
        Encode a frame as JPEG base64 data URL.
        Accepts numpy HWC BGR (cv2) or torch.Tensor BCHW/CHW RGB (Phase 4 GPU pipeline).
        GPU-sourced frames (tensors) are always encoded via the tensor path to avoid CPU
        round-trip; only numpy HWC BGR uses the cv2 resize/encode path.
        """
        if is_tensor(frame):
            return self._frame_tensor_to_base64_url(frame)
        h, w = frame.shape[:2]
        if w > FRAME_MAX_WIDTH:
            scale = FRAME_MAX_WIDTH / w
            frame = cv2.resize(frame, (FRAME_MAX_WIDTH, int(h * scale)))
        _, buf = cv2.imencode(".jpg", frame)
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"

    def _frame_tensor_to_base64_url(self, t: Any) -> str:
        """Encode torch.Tensor BCHW/CHW RGB as JPEG base64; resize if > max width."""
        try:
            import torch
            from torch.nn import functional as F
            from torchvision.io import encode_jpeg
        except ImportError:
            logger.warning("torch/torchvision not available for tensor base64 encode")
            return "data:image/jpeg;base64,"
        if t.dim() == 4:
            t = t.squeeze(0)
        if t.dim() != 3 or t.shape[0] not in (1, 3):
            logger.warning(
                "_frame_tensor_to_base64_url expected CHW 1 or 3 channels, got %s",
                t.shape,
            )
            return "data:image/jpeg;base64,"
        h, w = int(t.shape[1]), int(t.shape[2])
        if w > FRAME_MAX_WIDTH:
            scale = FRAME_MAX_WIDTH / w
            new_h, new_w = int(h * scale), FRAME_MAX_WIDTH
            t = F.interpolate(
                t.unsqueeze(0).float(),
                size=(new_h, new_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            t = t.clamp(0.0, 255.0).round().to(torch.uint8)
        elif t.dtype != torch.uint8:
            t = (
                t.clamp(0.0, 255.0).round().to(torch.uint8)
                if t.is_floating_point()
                else t.to(torch.uint8)
            )
        t = t.cpu()
        try:
            jpeg_bytes = encode_jpeg(t, quality=90)
        except Exception as e:
            logger.warning("encode_jpeg failed in _frame_tensor_to_base64_url: %s", e)
            return "data:image/jpeg;base64,"
        b64 = base64.b64encode(jpeg_bytes.cpu().numpy().tobytes()).decode("ascii")  # type: ignore[union-attr]
        return f"data:image/jpeg;base64,{b64}"

    def _frame_cap_stats_and_check(self, frame_count: int) -> tuple[bool, int, int]:
        """
        Compute rolling-window stats, log them, check if request would exceed cap.
        Returns (allowed, current_frames_in_window, requests_in_window).
        Caller holds _frame_cap_lock when mutating _frame_cap_records; we acquire it.
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
            self._frame_cap_records = [
                (ts, n) for ts, n in self._frame_cap_records if now - ts <= window_sec
            ]
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
        """Record successful send for rolling cap. Lock only from send_to_proxy."""
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
        image_buffers: numpy HWC BGR or torch BCHW RGB; _frame_to_base64_url.
        Returns JSON (title, shortSummary, scene, confidence, threat_level) or None.
        """
        if not self._proxy_url or not self._api_key:
            logger.warning("Gemini proxy_url or api_key not configured")
            return None
        allowed, _, _ = self._frame_cap_stats_and_check(len(image_buffers))
        if not allowed:
            logger.warning(
                "Gemini API call skipped: hourly frame cap reached (cap=%d).",
                self._frame_cap_per_hour,
            )
            return None
        user_content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": "Analyze these camera frames; respond with requested JSON.",
            }
        ]
        for frame in image_buffers:
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": self._frame_to_base64_url(frame)},
                }
            )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        resp = self._post_messages(messages, GEMINI_PROXY_ANALYSIS_TIMEOUT)
        if resp is None:
            return None
        try:
            data = resp.json()
        except (ValueError, TypeError) as e:
            logger.warning("Failed to parse send_to_proxy response JSON: %s", e)
            return None
        content = self._parse_response_content(data)
        if not content:
            logger.warning("Empty response content from proxy")
            return None
        raw = content.strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            raw = "\n".join(lines)
        try:
            result = json.loads(raw)
        except json.JSONDecodeError as e:
            logger.exception("Failed to parse proxy JSON: %s", e)
            return None
        self._record_frames_sent(len(image_buffers))
        return result

    def send_text_prompt(self, system_prompt: str, user_prompt: str) -> str | None:
        """
        POST text-only messages to Gemini proxy (OpenAI-compatible). No images.
        Uses same GEMINI_PROXY_URL and GEMINI_API_KEY as send_to_proxy.
        Returns raw response content (e.g. Markdown) or None on failure.
        """
        if not self._proxy_url or not self._api_key:
            logger.warning("Gemini proxy_url or api_key not configured")
            return None
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        resp = self._post_messages(messages, GEMINI_PROXY_ANALYSIS_TIMEOUT)
        if resp is None:
            return None
        try:
            data = resp.json()
        except (ValueError, TypeError) as e:
            logger.warning("Failed to parse send_text_prompt response JSON: %s", e)
            return None
        content = self._parse_response_content(data)
        if not content:
            logger.warning("Empty response content from proxy (text prompt)")
            return None
        return content

    def _extract_and_prepare_ce_frames(
        self,
        ce_folder_path: str,
        ce_start_time: float = 0,
        primary_camera: str | None = None,
        log_callback: Callable[[str], None] | None = None,
    ) -> tuple[tuple[list[Any], str, list[Any]] | None, str | None]:
        """
        Extract frames from CE folder, add timestamp overlays, and build system prompt.
        Single place: config, extract_target_centric_frames, overlay, prompt build.
        Returns ((frames_raw, system_prompt, frames), None) or (None, error_message).
        """
        if not os.path.isdir(ce_folder_path):
            return (None, "CE folder not found")
        if log_callback is None:
            logger.info("Creating frame timeline")
        frames_raw = extract_target_centric_frames(
            ce_folder_path,
            max_frames_sec=self.max_frames_sec,
            max_frames_min=self.max_frames_min,
            crop_width=self.crop_width,
            crop_height=self.crop_height,
            tracking_target_frame_percent=int(
                self.config.get("TRACKING_TARGET_FRAME_PERCENT", 40)
            ),
            primary_camera=primary_camera,
            decode_second_camera_cpu_only=bool(
                self.config.get("DECODE_SECOND_CAMERA_CPU_ONLY", False)
            ),
            log_callback=log_callback,
            config=self.config,
            camera_timeline_analysis_multiplier=float(
                self.config.get("CAMERA_TIMELINE_ANALYSIS_MULTIPLIER", 2)
            ),
            camera_timeline_ema_alpha=float(
                self.config.get("CAMERA_TIMELINE_EMA_ALPHA", 0.4)
            ),
            camera_timeline_primary_bias_multiplier=float(
                self.config.get("CAMERA_TIMELINE_PRIMARY_BIAS_MULTIPLIER", 1.2)
            ),
            camera_switch_min_segment_frames=int(
                self.config.get("CAMERA_SWITCH_MIN_SEGMENT_FRAMES", 5)
            ),
            camera_switch_hysteresis_margin=float(
                self.config.get("CAMERA_SWITCH_HYSTERESIS_MARGIN", 1.15)
            ),
            camera_timeline_final_yolo_drop_no_person=bool(
                self.config.get("CAMERA_TIMELINE_FINAL_YOLO_DROP_NO_PERSON", False)
            ),
        )
        if not frames_raw:
            return (None, "No frames extracted (check clips and sidecars).")
        image_count = len(frames_raw)
        cameras_str = ", ".join(sorted({ef.camera for ef in frames_raw}))
        for i, ef in enumerate(frames_raw):
            ts = (
                ce_start_time + ef.timestamp_sec
                if ce_start_time > 0
                else ef.timestamp_sec
            )
            time_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
            person_area = (
                ef.metadata.get("person_area")
                if self.config.get("PERSON_AREA_DEBUG")
                else None
            )
            ef.frame = crop_utils.draw_timestamp_overlay(
                ef.frame,
                time_str,
                ef.camera,
                i + 1,
                image_count,
                person_area=person_area,
            )
        frames = [ef.frame for ef in frames_raw]
        first_ts = frames_raw[0].timestamp_sec if frames_raw else 0
        activity_start_str = (
            datetime.fromtimestamp(ce_start_time + first_ts).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            if ce_start_time > 0
            else datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        if log_callback is None:
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
        return ((frames_raw, system_prompt, frames), None)

    def analyze_multi_clip_ce(
        self,
        ce_id: str,
        ce_folder_path: str,
        ce_start_time: float = 0,
        primary_camera: str | None = None,
    ) -> dict[str, Any] | None:
        """
        Target-centric multi-clip analysis for consolidated events.
        Extracts frames via object detection, sends to Gemini, saves to CE root.
        Handles CEs with one or more cameras; same pipeline.
        primary_camera: optional initiator (first-camera bias in selection).
        """
        if not self._enabled:
            logger.debug("Gemini analysis disabled, skipping CE %s", ce_id)
            return None
        if not self._proxy_url or not self._api_key:
            logger.debug("Gemini proxy not configured, skipping CE %s", ce_id)
            return None
        try:
            data, err = self._extract_and_prepare_ce_frames(
                ce_folder_path,
                ce_start_time,
                primary_camera=primary_camera,
                log_callback=None,
            )
            if err or data is None:
                if err == "No frames extracted (check clips and sidecars).":
                    logger.warning("No frames extracted from CE %s", ce_id)
                return None
            frames_raw, system_prompt, frames = data
            logger.info("Sending to proxy")
            result = self.send_to_proxy(system_prompt, frames)
            if result and isinstance(result, dict):
                out_path = os.path.join(ce_folder_path, "analysis_result.json")
                try:
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(result, f, indent=2, ensure_ascii=False)
                    logger.debug(
                        "Saved analysis_result.json for CE %s to %s", ce_id, out_path
                    )
                except OSError as e:
                    logger.warning(
                        "Failed to write analysis_result.json for CE %s: %s", ce_id, e
                    )
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
        Build same payload as analyze_multi_clip_ce (prompt + user content with frames)
        and write frame files to ce_folder_path, but do not call send_to_proxy.

        Used by event_test orchestrator to produce system_prompt.txt and ai_request.html
        with download links to the written frame files.

        Returns (result, err). result = (system_prompt, user_content, frame_paths);
        or None; error_message when something failed (no frames, exception).
        When log_messages is provided, appends progress strings for SSE.
        """
        if not os.path.isdir(ce_folder_path):
            logger.warning("CE folder not found: %s", ce_folder_path)
            return (None, "CE folder not found")
        try:

            def _log(msg: str) -> None:
                if log_messages is not None:
                    log_messages.append(msg)

            camera_subdirs: list[str] = []
            try:
                with os.scandir(ce_folder_path) as it:
                    camera_subdirs = [
                        e.name for e in it if e.is_dir() and not e.name.startswith(".")
                    ]
            except OSError:
                pass
            all_have_sidecar = True
            missing: list[str] = []
            for cam in camera_subdirs:
                sidecar_path = os.path.join(
                    ce_folder_path, cam, _DETECTION_SIDECAR_FILENAME
                )
                if not os.path.isfile(sidecar_path):
                    all_have_sidecar = False
                    missing.append(cam)
            if all_have_sidecar and camera_subdirs:
                _log("Frame selection: detection sidecars (best camera per moment).")
            else:
                part = f" (missing: {', '.join(missing)})" if missing else ""
                msg = (
                    "Frame selection: sidecars missing; skipping multi-clip extract."
                    f"{part}"
                )
                _log(msg)

            data, err = self._extract_and_prepare_ce_frames(
                ce_folder_path,
                ce_start_time,
                primary_camera=None,
                log_callback=_log if log_messages is not None else None,
            )
            if err or data is None:
                return (None, err or "Preparation failed")
            frames_raw, system_prompt, frames = data
            cameras_str = ", ".join(sorted({ef.camera for ef in frames_raw}))
            _log(f"Extracted {len(frames_raw)} frames from clips ({cameras_str}).")
            _log(f"Organizing timeline (saving {len(frames_raw)} frames).")
            save_frames = bool(self.config.get("SAVE_AI_FRAMES", True))
            write_ai_frame_analysis_multi_cam(
                ce_folder_path,
                frames_raw,
                write_manifest=True,
                create_zip_flag=False,
                save_frames=save_frames,
            )
            _log("Building payload for preview")
            user_content: list[Any] = [
                {
                    "type": "text",
                    "text": "Analyze these camera frames; respond with requested JSON.",
                }
            ]
            for frame in frames:
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": self._frame_to_base64_url(frame)},
                    }
                )
            frame_relative_paths: list[str] = []
            frames_dir = os.path.join(ce_folder_path, "ai_frame_analysis", "frames")
            if os.path.isdir(frames_dir):
                for name in sorted(os.listdir(frames_dir)):
                    if name.startswith("frame_") and name.endswith(".jpg"):
                        frame_relative_paths.append(
                            os.path.join("ai_frame_analysis", "frames", name)
                        )
            return ((system_prompt, user_content, frame_relative_paths), None)
        except Exception as e:
            logger.exception(
                "build_multi_cam_payload_for_preview failed for %s: %s",
                ce_folder_path,
                e,
            )
            return (None, str(e))
