"""
Download Service - Handles network downloads and API interactions with Frigate.
"""

import os
import shutil
import time
import logging

import requests

from frigate_buffer.services.video import VideoService

logger = logging.getLogger('frigate-buffer')


# Default timeouts (seconds); can be overridden via config for slow Frigate exports
DEFAULT_EXPORT_DOWNLOAD_TIMEOUT = 360
DEFAULT_EVENTS_CLIP_TIMEOUT = 300


class DownloadService:
    """Handles network downloads and API interactions with Frigate."""

    def __init__(
        self,
        frigate_url: str,
        video_service: VideoService,
        export_download_timeout: int = DEFAULT_EXPORT_DOWNLOAD_TIMEOUT,
        events_clip_timeout: int = DEFAULT_EVENTS_CLIP_TIMEOUT,
    ):
        self.frigate_url = frigate_url
        self.video_service = video_service
        self.export_download_timeout = export_download_timeout
        self.events_clip_timeout = events_clip_timeout
        logger.info(
            f"DownloadService initialized with Frigate URL: {frigate_url}, "
            f"export_download_timeout={export_download_timeout}s, events_clip_timeout={events_clip_timeout}s"
        )

    def _request_export(self, export_url: str, body: dict) -> requests.Response:
        """
        POST to Frigate export URL with a JSON body. Frigate 0.17+ requires the body
        object to exist for validation; do not remove json={} even if it appears redundant.
        On non-200, response.text is logged so Pydantic validation errors are visible.
        """
        resp = requests.post(
            export_url,
            json=body,
            headers={"Content-Type": "application/json"},
            timeout=60,
        )
        if resp.status_code != 200:
            logger.warning(
                f"Export API non-200: status={resp.status_code} response.text={resp.text or '(empty)'}"
            )
        return resp

    def download_snapshot(self, event_id: str, folder_path: str) -> bool:
        """Download snapshot from Frigate API (streamed to file to avoid loading full response into memory)."""
        url = f"{self.frigate_url}/api/events/{event_id}/snapshot.jpg"
        logger.debug(f"Downloading snapshot from {url}")

        try:
            response = requests.get(url, timeout=30, stream=True)
            response.raise_for_status()

            snapshot_path = os.path.join(folder_path, "snapshot.jpg")
            bytes_downloaded = 0
            with open(snapshot_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
                        bytes_downloaded += len(chunk)

            logger.info(f"Downloaded snapshot for {event_id} ({bytes_downloaded} bytes)")
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
        # Compute export time range (Unix epoch seconds); Frigate 0.17 expects integers
        start_ts = int(start_time - export_buffer_before)
        end_ts = int(end_time + export_buffer_after)

        # Frigate Export API expects camera name as used in config (e.g. "Doorbell")
        export_url = f"{self.frigate_url}/api/export/{camera}/start/{start_ts}/end/{end_ts}"
        exports_list_url = f"{self.frigate_url}/api/exports"

        temp_path = os.path.join(folder_path, "clip_original.mp4")
        final_path = os.path.join(folder_path, "clip.mp4")
        frigate_response: dict | None = None

        # Body must always be a dict; Frigate 0.17+ requires it for validation (json={} minimum)
        body = {}
        body["name"] = f"export_{event_id}"[:256]
        body["playback"] = "realtime"
        body["source"] = "recordings"

        try:
            logger.info(f"Requesting clip export: {export_url}")
            resp = self._request_export(export_url, body)

            # Capture response for timeline debugging before raise_for_status
            if resp.headers.get("content-type", "").startswith("application/json"):
                try:
                    frigate_response = resp.json()
                except Exception:
                    frigate_response = {"status_code": resp.status_code, "raw": (resp.text or "")[:500]}
            else:
                if resp.text or resp.status_code != 200:
                    frigate_response = {"status_code": resp.status_code, "raw": (resp.text or "")[:500]}

            # Log failure payload if 200 but success: false (already logged non-200 in _request_export)
            if resp.ok and frigate_response and isinstance(frigate_response, dict) and frigate_response.get("success") is False:
                logger.warning(f"Export failed for {event_id}. Status: {resp.status_code}. Response: {resp.text or '(empty)'}")

            resp.raise_for_status()

            # 2. Get export filename or export_id from response
            export_filename = None
            export_id = None
            if frigate_response:
                export_filename = frigate_response.get("export") or frigate_response.get("filename") or frigate_response.get("name")
                export_id = frigate_response.get("export_id")

            # 3. Poll exports list until our export is completed (in_progress false or missing)
            # Do not download until completed; Frigate muxes the file asynchronously (10-20s for 4K).
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
                        matched = None
                        if export_id:
                            for e in exports:
                                if e.get("export_id") == export_id or e.get("id") == export_id:
                                    matched = e
                                    break
                        if not matched:
                            # Fallback: use newest export (likely ours)
                            newest = max(
                                exports,
                                key=lambda e: e.get("created", 0) or e.get("start_time", 0) or e.get("modified", 0)
                            )
                            if not export_id or newest.get("export_id") == export_id or newest.get("id") == export_id:
                                matched = newest
                        if matched:
                            # Only proceed when in_progress is false or missing (export completed)
                            if not matched.get("in_progress", False):
                                export_filename = (
                                    matched.get("video_path")
                                    or matched.get("export")
                                    or matched.get("filename")
                                    or matched.get("name")
                                    or matched.get("path")
                                )
                                if export_filename:
                                    # Use the export id from GET /exports so the watchdog can DELETE with the correct id
                                    if isinstance(frigate_response, dict):
                                        frigate_response["export_id"] = (
                                            matched.get("id")
                                            or matched.get("export_id")
                                            or frigate_response.get("export_id")
                                        )
                                    break
                            else:
                                logger.debug("Export still processing..., waiting for in_progress false")
                    elif isinstance(exports, dict):
                        e = exports
                        if not e.get("in_progress", False):
                            export_filename = e.get("video_path") or e.get("export") or e.get("filename") or e.get("name")
                            if export_filename and isinstance(frigate_response, dict):
                                frigate_response["export_id"] = (
                                    e.get("id") or e.get("export_id") or frigate_response.get("export_id")
                                )
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

            # Ensure we have the export id from GET /exports for the watchdog (DELETE expects list id).
            # If we got export_filename from POST we may have skipped the poll loop and never set it.
            if isinstance(frigate_response, dict):
                try:
                    list_resp = requests.get(exports_list_url, timeout=15)
                    list_resp.raise_for_status()
                    exports = list_resp.json() if list_resp.content else []
                    if isinstance(exports, list) and exports:
                        list_id = frigate_response.get("export_id") or frigate_response.get("id")
                        matched = None
                        for e in exports:
                            e_path = (
                                e.get("video_path")
                                or e.get("export")
                                or e.get("filename")
                                or e.get("name")
                                or e.get("path")
                            )
                            if e_path:
                                tail = export_filename.lstrip("/").split("/")[-1]
                                if export_filename in e_path or e_path.endswith(tail) or tail in e_path:
                                    matched = e
                                    break
                            if list_id and (e.get("id") == list_id or e.get("export_id") == list_id):
                                matched = e
                                break
                        if matched:
                            frigate_response["export_id"] = matched.get("id") or matched.get("export_id") or frigate_response.get("export_id")
                except Exception as e:
                    logger.debug("Could not sync export_id from exports list: %s", e)

            # 4. Download from Frigate exports (web server path, not /api/)
            # Handle path that may include camera prefix (e.g. "Doorbell/xxx.mp4" or "Doorbell_xxx.mp4")
            download_path = export_filename.lstrip("/").split("/")[-1] if "/" in export_filename else export_filename
            download_url = f"{self.frigate_url.rstrip('/')}/exports/{download_path}"
            logger.info(f"Downloading export clip from {download_url}")
            dl_resp = requests.get(download_url, timeout=self.export_download_timeout, stream=True)
            dl_resp.raise_for_status()

            bytes_downloaded = 0
            with open(temp_path, "wb") as f:
                for chunk in dl_resp.iter_content(chunk_size=8192):
                    f.write(chunk)
                    bytes_downloaded += len(chunk)

            logger.info(f"Downloaded export clip for {event_id} ({bytes_downloaded} bytes), transcoding...")

            # 4. Transcode
            transcode_ok, _ = self.video_service.transcode_clip_to_h264(event_id, temp_path, final_path)
            return {"success": transcode_ok, "frigate_response": frigate_response}

        except requests.exceptions.RequestException as e:
            if getattr(e, "response", None) is not None and e.response.text:
                logger.warning(f"Export API request failed for {event_id}: response.text={e.response.text}")
            else:
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

    def transcode_temp_to_final(
        self,
        event_id: str,
        temp_path: str,
        final_path: str,
        detection_sidecar_path: str | None = None,
        detection_model: str | None = None,
        detection_device: str | None = None,
) -> bool:
        """
        Transcode from temp file to final H.264 clip and remove temp.
        Used by the multi-cam pipeline so transcode can run in a bounded pool while the next clip downloads.
        When detection_sidecar_path is set (multi-cam), transcode will run ultralytics per frame and write detection.json.
        """
        try:
            ok, _ = self.video_service.transcode_clip_to_h264(
                event_id,
                temp_path,
                final_path,
                detection_sidecar_path=detection_sidecar_path,
                detection_model=detection_model,
                detection_device=detection_device or None,
            )
            return ok
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    def transcode_existing_clip(
        self,
        event_id: str,
        camera_folder_path: str,
        detection_sidecar_path: str | None = None,
        detection_model: str | None = None,
        detection_device: str | None = None,
    ) -> tuple[bool, str]:
        """
        Transcode an existing clip.mp4 in place (copy to temp, transcode to clip.mp4, remove temp).
        Returns (success, backend_string) for SSE logging (e.g. 'GPU' or 'CPU: NVENC unavailable').
        Used by the event_test orchestrator to run the post-download transcode step on copied event data.
        When detection_sidecar_path is set, transcode writes detection.json (e.g. for multi-cam frame extraction).
        """
        clip_path = os.path.join(camera_folder_path, "clip.mp4")
        if not os.path.isfile(clip_path):
            logger.warning("No clip.mp4 at %s", camera_folder_path)
            return (False, "no clip")
        temp_path = os.path.join(camera_folder_path, "clip_transcode_src.mp4")
        try:
            shutil.copy2(clip_path, temp_path)
            ok, backend = self.video_service.transcode_clip_to_h264(
                event_id,
                temp_path,
                clip_path,
                detection_sidecar_path=detection_sidecar_path,
                detection_model=detection_model,
                detection_device=detection_device or None,
            )
            return (ok, backend)
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    def download_clip_to_temp(self, event_id: str, folder_path: str) -> dict:
        """
        Download clip from Frigate events API to temp file only (no transcode).
        Returns dict with success (bool), temp_path (str | None). Caller must transcode and delete temp.
        """
        temp_path = os.path.join(folder_path, "clip_original.mp4")
        url = f"{self.frigate_url}/api/events/{event_id}/clip.mp4"
        try:
            for attempt in range(1, 4):
                logger.debug(f"Downloading clip from {url} (attempt {attempt}/3)")
                try:
                    with requests.get(url, timeout=self.events_clip_timeout, stream=True) as response:
                        response.raise_for_status()
                        bytes_downloaded = 0
                        with open(temp_path, "wb") as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                                    bytes_downloaded += len(chunk)
                        logger.debug(f"Downloaded clip for {event_id} ({bytes_downloaded} bytes)")
                        return {"success": True, "temp_path": temp_path}
                except requests.exceptions.HTTPError as e:
                    if e.response is not None:
                        if e.response.status_code == 404:
                            logger.warning(f"No recording available for event {event_id}")
                            return {"success": False, "temp_path": None}
                        if e.response.status_code == 400 and attempt < 3:
                            logger.warning(f"Clip not ready for {event_id} (HTTP 400), retrying in 5s ({attempt}/3)")
                            time.sleep(5)
                            continue
                    raise
            return {"success": False, "temp_path": None}
        except requests.exceptions.Timeout:
            logger.error(f"Timeout downloading clip for {event_id}")
            return {"success": False, "temp_path": None}
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error downloading clip for {event_id}: {e}")
            return {"success": False, "temp_path": None}
        except Exception as e:
            logger.exception(f"Failed to download clip for {event_id}: {e}")
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
            return {"success": False, "temp_path": None}

    def export_and_download_clip(
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
        Request clip export from Frigate, poll until ready, and download to temp file only.
        Does not transcode; caller can run transcode_temp_to_final (e.g. in a thread pool).
        Returns dict: success (bool), temp_path (str | None), frigate_response (dict), optionally fallback (str).
        On fallback to events API, downloads to temp and returns temp_path so pipeline can transcode in pool.
        """
        start_ts = int(start_time - export_buffer_before)
        end_ts = int(end_time + export_buffer_after)
        export_url = f"{self.frigate_url}/api/export/{camera}/start/{start_ts}/end/{end_ts}"
        exports_list_url = f"{self.frigate_url}/api/exports"
        temp_path = os.path.join(folder_path, "clip_original.mp4")
        frigate_response: dict | None = None
        body = {}
        body["name"] = f"export_{event_id}"[:256]
        body["playback"] = "realtime"
        body["source"] = "recordings"

        try:
            logger.info(f"Requesting clip export: {export_url}")
            resp = self._request_export(export_url, body)
            if resp.headers.get("content-type", "").startswith("application/json"):
                try:
                    frigate_response = resp.json()
                except Exception:
                    frigate_response = {"status_code": resp.status_code, "raw": (resp.text or "")[:500]}
            else:
                if resp.text or resp.status_code != 200:
                    frigate_response = {"status_code": resp.status_code, "raw": (resp.text or "")[:500]}
            if resp.ok and frigate_response and isinstance(frigate_response, dict) and frigate_response.get("success") is False:
                logger.warning(f"Export failed for {event_id}. Status: {resp.status_code}. Response: {resp.text or '(empty)'}")
            resp.raise_for_status()

            export_filename = None
            export_id = None
            if frigate_response:
                export_filename = frigate_response.get("export") or frigate_response.get("filename") or frigate_response.get("name")
                export_id = frigate_response.get("export_id")
            poll_count = 0
            poll_max = 36
            while not export_filename and poll_count < poll_max:
                time.sleep(2.5)
                poll_count += 1
                try:
                    list_resp = requests.get(exports_list_url, timeout=15)
                    list_resp.raise_for_status()
                    exports = list_resp.json() if list_resp.content else []
                    if isinstance(exports, list) and exports:
                        matched = None
                        if export_id:
                            for e in exports:
                                if e.get("export_id") == export_id or e.get("id") == export_id:
                                    matched = e
                                    break
                        if not matched:
                            newest = max(
                                exports,
                                key=lambda e: e.get("created", 0) or e.get("start_time", 0) or e.get("modified", 0),
                            )
                            if not export_id or newest.get("export_id") == export_id or newest.get("id") == export_id:
                                matched = newest
                        if matched and not matched.get("in_progress", False):
                            export_filename = (
                                matched.get("video_path")
                                or matched.get("export")
                                or matched.get("filename")
                                or matched.get("name")
                                or matched.get("path")
                            )
                            if export_filename and isinstance(frigate_response, dict):
                                frigate_response["export_id"] = matched.get("id") or matched.get("export_id") or frigate_response.get("export_id")
                            if export_filename:
                                break
                        else:
                            logger.debug("Export still processing..., waiting for in_progress false")
                    elif isinstance(exports, dict):
                        e = exports
                        if not e.get("in_progress", False):
                            export_filename = e.get("video_path") or e.get("export") or e.get("filename") or e.get("name")
                            if export_filename and isinstance(frigate_response, dict):
                                frigate_response["export_id"] = e.get("id") or e.get("export_id") or frigate_response.get("export_id")
                except Exception as e:
                    logger.debug(f"Exports poll: {e}")

            if not export_filename:
                logger.warning("Could not determine export filename, falling back to events API")
                fallback_result = self.download_clip_to_temp(event_id, folder_path)
                return {
                    "success": fallback_result.get("success", False),
                    "temp_path": fallback_result.get("temp_path"),
                    "frigate_response": frigate_response,
                    "fallback": "events_api",
                }

            if isinstance(frigate_response, dict):
                try:
                    list_resp = requests.get(exports_list_url, timeout=15)
                    list_resp.raise_for_status()
                    exports = list_resp.json() if list_resp.content else []
                    if isinstance(exports, list) and exports:
                        list_id = frigate_response.get("export_id") or frigate_response.get("id")
                        matched = None
                        for e in exports:
                            e_path = (
                                e.get("video_path") or e.get("export") or e.get("filename") or e.get("name") or e.get("path")
                            )
                            if e_path:
                                tail = export_filename.lstrip("/").split("/")[-1]
                                if export_filename in e_path or e_path.endswith(tail) or tail in e_path:
                                    matched = e
                                    break
                            if list_id and (e.get("id") == list_id or e.get("export_id") == list_id):
                                matched = e
                                break
                        if matched:
                            frigate_response["export_id"] = matched.get("id") or matched.get("export_id") or frigate_response.get("export_id")
                except Exception as e:
                    logger.debug("Could not sync export_id from exports list: %s", e)

            download_path = export_filename.lstrip("/").split("/")[-1] if "/" in export_filename else export_filename
            download_url = f"{self.frigate_url.rstrip('/')}/exports/{download_path}"
            logger.info(f"Downloading export clip from {download_url}")
            dl_resp = requests.get(download_url, timeout=self.export_download_timeout, stream=True)
            dl_resp.raise_for_status()
            bytes_downloaded = 0
            with open(temp_path, "wb") as f:
                for chunk in dl_resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        bytes_downloaded += len(chunk)
            logger.info(f"Downloaded export clip for {event_id} ({bytes_downloaded} bytes)")
            return {"success": True, "temp_path": temp_path, "frigate_response": frigate_response}

        except requests.exceptions.RequestException as e:
            if getattr(e, "response", None) is not None and e.response.text:
                logger.warning(f"Export API request failed for {event_id}: response.text={e.response.text}")
            else:
                logger.warning(f"Export API failed for {event_id}: {e}, falling back to events API")
            fallback_result = self.download_clip_to_temp(event_id, folder_path)
            return {
                "success": fallback_result.get("success", False),
                "temp_path": fallback_result.get("temp_path"),
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
            return {"success": False, "temp_path": None, "frigate_response": frigate_response or {"error": str(e)}}

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
                    with requests.get(url, timeout=self.events_clip_timeout, stream=True) as response:
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

            ok, _ = self.video_service.transcode_clip_to_h264(event_id, temp_path, final_path)
            return ok

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
                             padding_before: float, padding_after: float) -> str | None:
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
