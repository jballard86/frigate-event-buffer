"""
Download Service - Handles network downloads and API interactions with Frigate.
"""

import os
import time
import logging
from typing import Optional

import requests

from frigate_buffer.services.video import VideoService

logger = logging.getLogger('frigate-buffer')


class DownloadService:
    """Handles network downloads and API interactions with Frigate."""

    def __init__(self, frigate_url: str, video_service: VideoService):
        self.frigate_url = frigate_url
        self.video_service = video_service
        logger.info(f"DownloadService initialized with Frigate URL: {frigate_url}")

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

    def post_event_description(self, event_id: str, description: str) -> bool:
        """POST event description to Frigate API so it is stored in Frigate's database."""
        url = f"{self.frigate_url}/api/events/{event_id}/description"
        try:
            response = requests.post(
                url,
                json={"description": description},
                headers={"Content-Type": "application/json"},
                timeout=15,
            )
            if response.ok:
                logger.info(f"Posted description to Frigate for event {event_id}")
                return True
            logger.warning(f"Frigate description API returned {response.status_code} for {event_id}")
            return False
        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to post description to Frigate for {event_id}: {e}")
            return False
