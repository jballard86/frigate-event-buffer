"""File operations: folder creation, downloads, transcoding, cleanup."""

import os
import re
import json
import time
import shutil
import logging
from datetime import datetime
from typing import List, Optional

import requests

from frigate_buffer.models import EventState
from frigate_buffer.services.video import VideoService

logger = logging.getLogger('frigate-buffer')


class FileManager:
    """Handles file operations: folder creation, downloads, transcoding, cleanup."""

    def __init__(self, storage_path: str, frigate_url: str, retention_days: int,
                 video_service: VideoService):
        self.storage_path = storage_path
        self.frigate_url = frigate_url
        self.retention_days = retention_days
        self.video_service = video_service

        # Ensure storage directory exists
        os.makedirs(storage_path, exist_ok=True)
        logger.info(f"FileManager initialized: {storage_path}")
        logger.debug(f"Retention: {retention_days} days")

    def sanitize_camera_name(self, camera: str) -> str:
        """Sanitize camera name for filesystem use."""
        # Lowercase, replace spaces with underscores, remove special chars
        sanitized = camera.lower().replace(' ', '_')
        sanitized = re.sub(r'[^a-z0-9_]', '', sanitized)
        return sanitized or 'unknown'

    def create_event_folder(self, event_id: str, camera: str, timestamp: float) -> str:
        """Create folder for event: {camera}/{timestamp}_{event_id} (legacy)"""
        sanitized_camera = self.sanitize_camera_name(camera)
        folder_name = f"{int(timestamp)}_{event_id}"
        camera_path = os.path.join(self.storage_path, sanitized_camera)
        folder_path = os.path.join(camera_path, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        logger.info(f"Created folder: {sanitized_camera}/{folder_name}")
        return folder_path

    def create_consolidated_event_folder(self, folder_name: str) -> str:
        """Create folder for consolidated event: events/{folder_name}"""
        events_dir = os.path.join(self.storage_path, "events")
        folder_path = os.path.join(events_dir, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        logger.info(f"Created consolidated folder: events/{folder_name}")
        return folder_path

    def ensure_consolidated_camera_folder(self, ce_folder_path: str, camera: str) -> str:
        """Ensure events/{ce_id}/{camera}/ exists. Returns the camera folder path."""
        sanitized = self.sanitize_camera_name(camera)
        camera_path = os.path.join(ce_folder_path, sanitized)
        os.makedirs(camera_path, exist_ok=True)
        return camera_path

    def download_snapshot(self, event_id: str, folder_path: str) -> bool:
        """Download snapshot from Frigate API."""
        url = f"{self.frigate_url}/api/events/{event_id}/snapshot.jpg"
        logger.debug(f"Downloading snapshot from {url}")

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            snapshot_path = os.path.join(folder_path, "snapshot.jpg")
            with open(snapshot_path, 'wb') as f:
                f.write(response.content)

            logger.info(f"Downloaded snapshot for {event_id} ({len(response.content)} bytes)")
            return True
        except requests.exceptions.Timeout:
            logger.error(f"Timeout downloading snapshot for {event_id}")
            return False
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error downloading snapshot for {event_id}: {e}")
            return False
        except Exception as e:
            logger.exception(f"Failed to download snapshot for {event_id}: {e}")
            return False

    def export_and_transcode_clip(
        self,
        event_id: str,
        folder_path: str,
        camera: str,
        start_time: float,
        end_time: float,
        export_buffer_before: int,
        export_buffer_after: int,
    ) -> dict:
        """
        Request clip export from Frigate's Export API for the full event duration
        (with buffer), then download and transcode to H.264.

        Uses buffer-assigned event time range: start_ts = start_time - buffer_before,
        end_ts = end_time + buffer_after. Falls back to events API if export fails.

        Returns dict with keys: success (bool), frigate_response (dict), optionally fallback (str).
        """
        # Compute export time range (Unix epoch seconds)
        start_ts = int(start_time - export_buffer_before)
        end_ts = int(end_time + export_buffer_after)

        # Frigate Export API expects camera name as used in config (e.g. "Doorbell")
        export_url = f"{self.frigate_url}/api/export/{camera}/start/{start_ts}/end/{end_ts}"
        exports_list_url = f"{self.frigate_url}/api/exports"

        temp_path = os.path.join(folder_path, "clip_original.mp4")
        final_path = os.path.join(folder_path, "clip.mp4")
        frigate_response: Optional[dict] = None

        try:
            # 1. Trigger export via POST (Frigate/FastAPI requires JSON body with playback, name)
            payload = {
                "playback": "realtime",
                "name": f"export_{event_id}"[:256],
            }
            logger.info(f"Requesting clip export: {export_url}")
            resp = requests.post(
                export_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=60
            )
            # Capture response for timeline debugging before raise_for_status
            if resp.headers.get("content-type", "").startswith("application/json"):
                try:
                    frigate_response = resp.json()
                except Exception:
                    frigate_response = {"status_code": resp.status_code, "raw": resp.text[:500] if resp.text else ""}
            elif resp.text:
                frigate_response = {"status_code": resp.status_code, "raw": resp.text[:500]}

            # Check if export failed (status not OK or success: false)
            if not resp.ok or (frigate_response and isinstance(frigate_response, dict) and frigate_response.get("success") is False):
                logger.warning(f"Export failed for {event_id}. Status: {resp.status_code}. Response: {resp.text}")

            resp.raise_for_status()

            # 2. Get export filename or export_id from response
            export_filename = None
            export_id = None
            if frigate_response:
                export_filename = frigate_response.get("export") or frigate_response.get("filename") or frigate_response.get("name")
                export_id = frigate_response.get("export_id")

            # 3. Poll exports list until our export appears (match by export_id or use newest)
            poll_count = 0
            poll_max = 36  # 36 * 2.5s = 90s max
            while not export_filename and poll_count < poll_max:
                time.sleep(2.5)
                poll_count += 1
                try:
                    list_resp = requests.get(exports_list_url, timeout=15)
                    list_resp.raise_for_status()
                    exports = list_resp.json() if list_resp.content else []
                    if isinstance(exports, list) and exports:
                        # Prefer match by export_id if we have it
                        if export_id:
                            for e in exports:
                                if e.get("export_id") == export_id or e.get("id") == export_id:
                                    export_filename = e.get("export") or e.get("filename") or e.get("name") or e.get("path")
                                    if export_filename:
                                        break
                        if not export_filename:
                            # Fallback: use newest export (likely ours)
                            newest = max(
                                exports,
                                key=lambda e: e.get("created", 0) or e.get("start_time", 0) or e.get("modified", 0)
                            )
                            export_filename = newest.get("export") or newest.get("filename") or newest.get("name") or newest.get("path")
                    elif isinstance(exports, dict):
                        export_filename = exports.get("export") or exports.get("filename") or exports.get("name")
                    if export_filename:
                        break
                except Exception as e:
                    logger.debug(f"Exports poll: {e}")

            if not export_filename:
                logger.warning("Could not determine export filename, falling back to events API")
                fallback_ok = self.download_and_transcode_clip(event_id, folder_path)
                return {
                    "success": fallback_ok,
                    "frigate_response": frigate_response,
                    "fallback": "events_api",
                }

            # 4. Download from Frigate exports (web server path, not /api/)
            # Handle path that may include camera prefix (e.g. "Doorbell/xxx.mp4" or "Doorbell_xxx.mp4")
            download_path = export_filename.lstrip("/").split("/")[-1] if "/" in export_filename else export_filename
            download_url = f"{self.frigate_url.rstrip('/')}/exports/{download_path}"
            logger.info(f"Downloading export clip from {download_url}")
            dl_resp = requests.get(download_url, timeout=180, stream=True)
            dl_resp.raise_for_status()

            bytes_downloaded = 0
            with open(temp_path, "wb") as f:
                for chunk in dl_resp.iter_content(chunk_size=8192):
                    f.write(chunk)
                    bytes_downloaded += len(chunk)

            logger.info(f"Downloaded export clip for {event_id} ({bytes_downloaded} bytes), transcoding...")

            # 4. Transcode
            transcode_ok = self.video_service.transcode_clip_to_h264(event_id, temp_path, final_path)
            return {"success": transcode_ok, "frigate_response": frigate_response}

        except requests.exceptions.RequestException as e:
            logger.warning(f"Export API failed for {event_id}: {e}, falling back to events API")
            fallback_ok = self.download_and_transcode_clip(event_id, folder_path)
            return {
                "success": fallback_ok,
                "frigate_response": frigate_response or {"error": str(e)},
                "fallback": "events_api",
            }
        except Exception as e:
            logger.exception(f"Export clip failed for {event_id}: {e}")
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
            return {"success": False, "frigate_response": frigate_response or {"error": str(e)}}

    def download_and_transcode_clip(self, event_id: str, folder_path: str) -> bool:
        """Download clip from Frigate events API and transcode to H.264 (fallback when Export API fails)."""
        temp_path = os.path.join(folder_path, "clip_original.mp4")
        final_path = os.path.join(folder_path, "clip.mp4")

        try:
            # Download original clip (retry on HTTP 400 — Frigate may not have clip ready yet)
            url = f"{self.frigate_url}/api/events/{event_id}/clip.mp4"

            download_success = False
            for attempt in range(1, 4):
                logger.debug(f"Downloading clip from {url} (attempt {attempt}/3)")
                try:
                    with requests.get(url, timeout=120, stream=True) as response:
                        response.raise_for_status()

                        bytes_downloaded = 0
                        with open(temp_path, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                f.write(chunk)
                                bytes_downloaded += len(chunk)

                        logger.debug(f"Downloaded clip for {event_id} ({bytes_downloaded} bytes), starting transcode...")
                        download_success = True
                        break

                except requests.exceptions.HTTPError as e:
                    if e.response is not None:
                        if e.response.status_code == 404:
                            logger.warning(f"No recording available for event {event_id}")
                            return False

                        if e.response.status_code == 400 and attempt < 3:
                            logger.warning(f"Clip not ready for {event_id} (HTTP 400), retrying in 5s ({attempt}/3)")
                            time.sleep(5)
                            continue
                    raise

            if not download_success:
                return False

            return self.video_service.transcode_clip_to_h264(event_id, temp_path, final_path)

        except requests.exceptions.Timeout:
            logger.error(f"Timeout downloading clip for {event_id}")
            return False
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error downloading clip for {event_id}: {e}")
            return False
        except Exception as e:
            logger.exception(f"Failed to download/transcode clip for {event_id}: {e}")
            return False
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    def write_summary(self, folder_path: str, event: EventState) -> bool:
        """Write summary.txt with event metadata."""
        try:
            summary_path = os.path.join(folder_path, "summary.txt")

            timestamp_str = time.strftime(
                '%Y-%m-%d %H:%M:%S',
                time.localtime(event.created_at)
            )

            lines = [
                f"Event ID: {event.event_id}",
                f"Camera: {event.camera}",
                f"Label: {event.label}",
                f"Timestamp: {timestamp_str}",
                f"Phase: {event.phase.name}",
                "",
            ]

            if event.genai_title:
                lines.append(f"Title: {event.genai_title}")

            if event.genai_description:
                lines.append(f"Description: {event.genai_description}")

            if event.genai_scene:
                lines.append(f"Scene: {event.genai_scene}")
            elif event.ai_description:
                lines.append(f"AI Description: {event.ai_description}")

            if event.severity:
                lines.append(f"Severity: {event.severity}")

            if event.threat_level > 0:
                level_names = {0: "Normal", 1: "Suspicious", 2: "Critical"}
                lines.append(f"Threat Level: {event.threat_level} ({level_names.get(event.threat_level, 'Unknown')})")

            if event.review_summary:
                lines.append("")
                lines.append("Review Summary: See review_summary.md")

            with open(summary_path, 'w') as f:
                f.write('\n'.join(lines))

            logger.debug(f"Written summary for {event.event_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to write summary: {e}")
            return False

    def write_review_summary(self, folder_path: str, summary_markdown: str) -> bool:
        """Write review_summary.md with the Frigate review summary."""
        try:
            summary_path = os.path.join(folder_path, "review_summary.md")
            with open(summary_path, 'w') as f:
                f.write(summary_markdown)
            logger.debug(f"Written review summary to {folder_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to write review summary: {e}")
            return False

    def append_timeline_entry(self, folder_path: str, entry: dict) -> None:
        """Append an entry to notification_timeline.json in the event folder."""
        timeline_path = os.path.join(folder_path, "notification_timeline.json")
        entry = dict(entry)
        entry["ts"] = entry.get("ts") or datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        try:
            data = {"event_id": None, "entries": []}
            if os.path.exists(timeline_path):
                with open(timeline_path, 'r') as f:
                    data = json.load(f)
            if not data.get("event_id"):
                folder_name = os.path.basename(folder_path)
                parts = folder_name.split("_", 1)
                if len(parts) > 1:
                    data["event_id"] = parts[1]
                elif entry.get("data", {}).get("event_id"):
                    data["event_id"] = entry["data"]["event_id"]
            data["entries"] = data.get("entries", [])
            data["entries"].append(entry)
            with open(timeline_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.debug(f"Failed to append timeline entry: {e}")

    def write_metadata_json(self, folder_path: str, event: EventState) -> bool:
        """Write machine-readable metadata.json for the event."""
        try:
            meta_path = os.path.join(folder_path, "metadata.json")
            metadata = {
                "event_id": event.event_id,
                "camera": event.camera,
                "label": event.label,
                "created_at": event.created_at,
                "end_time": event.end_time,
                "threat_level": event.threat_level,
                "severity": event.severity,
                "genai_title": event.genai_title,
                "genai_description": event.genai_description,
                "genai_scene": event.genai_scene,
                "phase": event.phase.name,
            }
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Failed to write metadata.json: {e}")
            return False

    def fetch_review_summary(self, start_ts: float, end_ts: float,
                             padding_before: float, padding_after: float) -> Optional[str]:
        """Fetch review summary from Frigate API with time padding."""
        padded_start = int(start_ts - padding_before)
        padded_end = int(end_ts + padding_after)

        url = f"{self.frigate_url}/api/review/summarize/start/{padded_start}/end/{padded_end}"
        logger.info(f"Fetching review summary: {url}")

        try:
            response = requests.post(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            summary = data.get("summary", "")
            if summary:
                logger.info(f"Review summary received ({len(summary)} chars)")
                return summary
            else:
                logger.warning("Review summary API returned empty summary")
                return None
        except requests.exceptions.Timeout:
            logger.error("Timeout fetching review summary")
            return None
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error fetching review summary: {e}")
            return None
        except Exception as e:
            logger.exception(f"Failed to fetch review summary: {e}")
            return None

    def cleanup_old_events(self, active_event_ids: List[str],
                          active_ce_folder_names: Optional[List[str]] = None) -> int:
        """Delete folders older than retention period. Returns count deleted.
        active_ce_folder_names: folder names of active consolidated events (e.g. 1771003190_abc) to skip."""
        now = time.time()
        cutoff = now - (self.retention_days * 86400)
        deleted_count = 0
        active_ce = set(active_ce_folder_names or [])

        logger.debug(f"Running cleanup: cutoff={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(cutoff))}")

        try:
            # Iterate through camera subdirectories and events/
            for camera_dir in os.listdir(self.storage_path):
                camera_path = os.path.join(self.storage_path, camera_dir)

                if not os.path.isdir(camera_path):
                    continue

                # Check if this is a camera folder (contains event folders)
                # or a legacy flat event folder (for migration period)
                first_item = camera_dir.split('_', 1)
                if len(first_item) > 1 and first_item[0].isdigit():
                    # Legacy flat structure: {timestamp}_{event_id}
                    try:
                        ts = float(first_item[0])
                        event_id = first_item[1]

                        if event_id and event_id in active_event_ids:
                            continue

                        if ts < cutoff:
                            shutil.rmtree(camera_path)
                            logger.info(f"Cleaned up legacy event: {camera_dir}")
                            deleted_count += 1
                    except (ValueError, IndexError):
                        pass
                    continue

                # New structure: iterate through event folders in camera dir
                for event_dir in os.listdir(camera_path):
                    event_path = os.path.join(camera_path, event_dir)

                    if not os.path.isdir(event_path):
                        continue

                    try:
                        parts = event_dir.split('_', 1)
                        ts = float(parts[0])
                        event_id = parts[1] if len(parts) > 1 else None

                        # Skip if event is still active (legacy) or CE is active (events/ folder)
                        if event_id and event_id in active_event_ids:
                            logger.debug(f"Skipping active event: {camera_dir}/{event_dir}")
                            continue
                        if camera_dir == "events" and event_dir in active_ce:
                            logger.debug(f"Skipping active consolidated event: {camera_dir}/{event_dir}")
                            continue

                        # Delete if older than cutoff
                        if ts < cutoff:
                            shutil.rmtree(event_path)
                            logger.info(f"Cleaned up old event: {camera_dir}/{event_dir}")
                            deleted_count += 1

                    except (ValueError, IndexError):
                        logger.debug(f"Skipping malformed folder: {camera_dir}/{event_dir}")
                        continue

        except Exception as e:
            logger.error(f"Cleanup error: {e}")

        return deleted_count

    def compute_storage_stats(self) -> dict:
        """Compute storage usage by camera and type. Returns bytes."""
        clips = 0
        snapshots = 0
        descriptions = 0
        by_camera = {}

        try:
            for camera_dir in os.listdir(self.storage_path):
                camera_path = os.path.join(self.storage_path, camera_dir)

                if not os.path.isdir(camera_path):
                    continue
                if camera_dir.split('_')[0].isdigit():
                    continue

                cam_clips = cam_snapshots = cam_descriptions = 0

                for event_dir in os.listdir(camera_path):
                    event_path = os.path.join(camera_path, event_dir)
                    if not os.path.isdir(event_path):
                        continue

                    clip_path = os.path.join(event_path, 'clip.mp4')
                    snapshot_path = os.path.join(event_path, 'snapshot.jpg')
                    for f in ('summary.txt', 'review_summary.md', 'metadata.json'):
                        p = os.path.join(event_path, f)
                        if os.path.exists(p):
                            cam_descriptions += os.path.getsize(p)
                    if os.path.exists(clip_path):
                        cam_clips += os.path.getsize(clip_path)
                    if os.path.exists(snapshot_path):
                        cam_snapshots += os.path.getsize(snapshot_path)

                cam_total = cam_clips + cam_snapshots + cam_descriptions
                if cam_total > 0:
                    by_camera[camera_dir] = {
                        'clips': cam_clips,
                        'snapshots': cam_snapshots,
                        'descriptions': cam_descriptions,
                        'total': cam_total
                    }
                clips += cam_clips
                snapshots += cam_snapshots
                descriptions += cam_descriptions

        except Exception as e:
            logger.error(f"Error computing storage stats: {e}")

        return {
            'clips': clips,
            'snapshots': snapshots,
            'descriptions': descriptions,
            'total': clips + snapshots + descriptions,
            'by_camera': by_camera
        }
