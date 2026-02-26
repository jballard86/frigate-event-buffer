"""
Download Service - Handles network downloads and API interactions with Frigate.

Clips are saved with dynamic names: {camera}-{5_random_digits}.mp4
(e.g. doorbell-82749.mp4). No transcoding; raw H.264/H.265 from Frigate is used as-is.
"""

import logging
import os
import random
import re
import time

import requests

from frigate_buffer.constants import HTTP_DOWNLOAD_CHUNK_SIZE, HTTP_STREAM_CHUNK_SIZE
from frigate_buffer.services.video import VideoService

logger = logging.getLogger("frigate-buffer")


def _dynamic_clip_basename(camera_name: str) -> str:
    """Return clip filename: {camera}-{5_digits}.mp4 (e.g. doorbell-82749.mp4)."""
    safe = re.sub(r"[^\w\-]", "_", (camera_name or "camera").strip().lower())
    safe = safe[:64] or "camera"
    return f"{safe}-{random.randint(10000, 99999):05d}.mp4"


def _remove_other_mp4_in_folder(folder_path: str, keep_path: str) -> None:
    """Delete any .mp4 in folder_path that is not keep_path (orphan GC after write)."""
    keep_abs = os.path.normpath(os.path.abspath(keep_path))
    try:
        for name in os.listdir(folder_path):
            if name.lower().endswith(".mp4"):
                fp = os.path.normpath(os.path.abspath(os.path.join(folder_path, name)))
                if fp != keep_abs and os.path.isfile(fp):
                    try:
                        os.remove(fp)
                        logger.debug("Removed orphan clip %s", fp)
                    except OSError as e:
                        logger.warning("Could not remove orphan clip %s: %s", fp, e)
    except OSError as e:
        logger.debug("Could not list folder for orphan GC %s: %s", folder_path, e)


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
        config: dict | None = None,
    ):
        self.frigate_url = frigate_url
        self.video_service = video_service
        self.export_download_timeout = export_download_timeout
        self.events_clip_timeout = events_clip_timeout
        self.config = config or {}
        logger.info(
            f"DownloadService initialized with Frigate URL: {frigate_url}, "
            f"export_download_timeout={export_download_timeout}s, "
            f"events_clip_timeout={events_clip_timeout}s",
        )

    def _request_export(self, export_url: str, body: dict) -> requests.Response:
        """
        POST to Frigate export URL with a JSON body. Frigate 0.17+ requires the body
        to exist for validation; do not remove json={} even if it appears redundant.
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
                "Export API non-200: file_name_requested=%s status=%s "
                "frigate_response=%s",
                body.get("name", "(unknown)"),
                resp.status_code,
                resp.text or "(empty)",
            )
        return resp

    def download_snapshot(self, event_id: str, folder_path: str) -> bool:
        """Download snapshot from Frigate API (streamed to file to avoid loading
        full response into memory)."""
        url = f"{self.frigate_url}/api/events/{event_id}/snapshot.jpg"
        logger.debug(f"Downloading snapshot from {url}")

        try:
            response = requests.get(url, timeout=30, stream=True)
            response.raise_for_status()

            snapshot_path = os.path.join(folder_path, "snapshot.jpg")
            bytes_downloaded = 0
            with open(snapshot_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=HTTP_DOWNLOAD_CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
                        bytes_downloaded += len(chunk)

            logger.info(
                f"Downloaded snapshot for {event_id} ({bytes_downloaded} bytes)"
            )
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
        Request clip export from Frigate's Export API, then download directly
        to dynamic path (no transcode).

        Uses buffer-assigned event time range. Saves as {camera}-{unix_timestamp}.mp4
        and removes any other .mp4 in the folder (orphan GC). Falls back to events
        API if export fails.

        Returns dict: success (bool), frigate_response (dict), clip_path (str | None),
        optionally fallback (str).
        """
        start_ts = int(start_time - export_buffer_before)
        end_ts = int(end_time + export_buffer_after)
        export_url = (
            f"{self.frigate_url}/api/export/{camera}/start/{start_ts}/end/{end_ts}"
        )
        exports_list_url = f"{self.frigate_url}/api/exports"
        final_path = os.path.join(folder_path, _dynamic_clip_basename(camera))
        frigate_response: dict | None = None

        body = {}
        body["name"] = f"export_{event_id}"[:256]
        body["playback"] = "realtime"
        body["source"] = "recordings"

        try:
            logger.info("Requesting clip export: %s", export_url)
            resp = self._request_export(export_url, body)

            if resp.headers.get("content-type", "").startswith("application/json"):
                try:
                    frigate_response = resp.json()
                except Exception:
                    frigate_response = {
                        "status_code": resp.status_code,
                        "raw": (resp.text or "")[:500],
                    }
            else:
                if resp.text or resp.status_code != 200:
                    frigate_response = {
                        "status_code": resp.status_code,
                        "raw": (resp.text or "")[:500],
                    }

            if (
                resp.ok
                and frigate_response
                and isinstance(frigate_response, dict)
                and frigate_response.get("success") is False
            ):
                logger.warning(
                    "Export failed for %s: file_name_requested=%s status=%s "
                    "frigate_response=%s",
                    event_id,
                    body.get("name", "(unknown)"),
                    resp.status_code,
                    resp.text or "(empty)",
                )

            resp.raise_for_status()

            export_filename = None
            export_id = None
            if frigate_response:
                export_filename = (
                    frigate_response.get("export")
                    or frigate_response.get("filename")
                    or frigate_response.get("name")
                )
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
                                if (
                                    e.get("export_id") == export_id
                                    or e.get("id") == export_id
                                ):
                                    matched = e
                                    break
                        # Frigate DELETE expects list "id", not "name".
                        # Match by name/path so we persist correct id for watchdog.
                        if not matched and export_filename:
                            tail = export_filename.lstrip("/").split("/")[-1]
                            for e in exports:
                                if (
                                    e.get("name") == export_filename
                                    or e.get("name") == tail
                                ):
                                    matched = e
                                    break
                                e_path = (
                                    e.get("video_path")
                                    or e.get("export")
                                    or e.get("filename")
                                    or e.get("name")
                                    or e.get("path")
                                )
                                if e_path and (
                                    export_filename in e_path
                                    or e_path.endswith(tail)
                                    or tail in e_path
                                ):
                                    matched = e
                                    break
                        if not matched and body.get("name"):
                            for e in exports:
                                if e.get("name") == body.get("name"):
                                    matched = e
                                    break
                        if not matched:
                            newest = max(
                                exports,
                                key=lambda e: (
                                    e.get("created", 0)
                                    or e.get("start_time", 0)
                                    or e.get("modified", 0)
                                ),
                            )
                            if (
                                not export_id
                                or newest.get("export_id") == export_id
                                or newest.get("id") == export_id
                            ):
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
                                # Persist list "id" for watchdog DELETE.
                                frigate_response["export_id"] = (
                                    matched.get("id")
                                    or matched.get("export_id")
                                    or frigate_response.get("export_id")
                                )
                            if export_filename:
                                break
                        else:
                            logger.debug(
                                "Export still processing..., "
                                "waiting for in_progress false",
                            )
                    elif isinstance(exports, dict):
                        e = exports
                        if not e.get("in_progress", False):
                            export_filename = (
                                e.get("video_path")
                                or e.get("export")
                                or e.get("filename")
                                or e.get("name")
                            )
                            if export_filename and isinstance(frigate_response, dict):
                                frigate_response["export_id"] = (
                                    e.get("id")
                                    or e.get("export_id")
                                    or frigate_response.get("export_id")
                                )
                except Exception as e:
                    logger.debug("Exports poll: %s", e)

            if not export_filename:
                logger.warning(
                    "Could not determine export filename, falling back to "
                    "events API: file_name_requested=%s frigate_response=%s",
                    body.get("name", "(unknown)"),
                    frigate_response,
                )
                fallback_result = self._download_clip_events_api_to_path(
                    event_id, folder_path, camera
                )
                return {
                    "success": fallback_result.get("success", False),
                    "frigate_response": frigate_response,
                    "clip_path": fallback_result.get("clip_path"),
                    "fallback": "events_api",
                }

            if isinstance(frigate_response, dict):
                try:
                    list_resp = requests.get(exports_list_url, timeout=15)
                    list_resp.raise_for_status()
                    exports = list_resp.json() if list_resp.content else []
                    if isinstance(exports, list) and exports:
                        list_id = frigate_response.get(
                            "export_id"
                        ) or frigate_response.get("id")
                        matched = None
                        tail = export_filename.lstrip("/").split("/")[-1]
                        # Prefer match by name so we persist list "id" for DELETE.
                        for e in exports:
                            if (
                                e.get("name") == export_filename
                                or e.get("name") == tail
                            ):
                                matched = e
                                break
                        if not matched:
                            for e in exports:
                                e_path = (
                                    e.get("video_path")
                                    or e.get("export")
                                    or e.get("filename")
                                    or e.get("name")
                                    or e.get("path")
                                )
                                if e_path:
                                    if (
                                        export_filename in e_path
                                        or e_path.endswith(tail)
                                        or tail in e_path
                                    ):
                                        matched = e
                                        break
                                if list_id and (
                                    e.get("id") == list_id
                                    or e.get("export_id") == list_id
                                ):
                                    matched = e
                                    break
                        if matched:
                            frigate_response["export_id"] = (
                                matched.get("id")
                                or matched.get("export_id")
                                or frigate_response.get("export_id")
                            )
                except Exception as e:
                    logger.debug("Could not sync export_id from exports list: %s", e)

            download_path = (
                export_filename.lstrip("/").split("/")[-1]
                if "/" in export_filename
                else export_filename
            )
            download_url = f"{self.frigate_url.rstrip('/')}/exports/{download_path}"
            logger.info("Downloading export clip from %s", download_url)
            dl_resp = requests.get(
                download_url, timeout=self.export_download_timeout, stream=True
            )
            dl_resp.raise_for_status()
            bytes_downloaded = 0
            with open(final_path, "wb") as f:
                for chunk in dl_resp.iter_content(chunk_size=HTTP_STREAM_CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
                        bytes_downloaded += len(chunk)
            _remove_other_mp4_in_folder(folder_path, final_path)
            logger.info(
                "Downloaded export clip for %s (%s bytes) to %s",
                event_id,
                bytes_downloaded,
                final_path,
            )
            return {
                "success": True,
                "frigate_response": frigate_response,
                "clip_path": final_path,
            }

        except requests.exceptions.RequestException as e:
            resp = getattr(e, "response", None)
            file_name = body.get("name", "(unknown)") if body else "(unknown)"
            if resp is not None and resp.text:
                logger.warning(
                    "Export API request failed: file_name_requested=%s "
                    "frigate_response=%s",
                    file_name,
                    resp.text,
                )
            else:
                logger.warning(
                    "Export API failed, falling back to events API: "
                    "file_name_requested=%s frigate_response=%s",
                    file_name,
                    e,
                )
            fallback_result = self._download_clip_events_api_to_path(
                event_id, folder_path, camera
            )
            return {
                "success": fallback_result.get("success", False),
                "frigate_response": frigate_response or {"error": str(e)},
                "clip_path": fallback_result.get("clip_path"),
                "fallback": "events_api",
            }
        except Exception as e:
            logger.exception(
                "Export clip failed: file_name_requested=%s frigate_response=%s",
                body.get("name", "(unknown)") if body else "(unknown)",
                frigate_response or str(e),
            )
            if os.path.exists(final_path):
                try:
                    os.remove(final_path)
                except OSError:
                    pass
            return {
                "success": False,
                "frigate_response": frigate_response or {"error": str(e)},
                "clip_path": None,
            }

    def _download_clip_events_api_to_path(
        self, event_id: str, folder_path: str, camera: str
    ) -> dict:
        """Download clip from Frigate events API to dynamic path; run orphan GC.
        Returns success, clip_path."""
        final_path = os.path.join(folder_path, _dynamic_clip_basename(camera))
        url = f"{self.frigate_url}/api/events/{event_id}/clip.mp4"
        try:
            for attempt in range(1, 4):
                logger.debug("Downloading clip from %s (attempt %s/3)", url, attempt)
                try:
                    with requests.get(
                        url, timeout=self.events_clip_timeout, stream=True
                    ) as response:
                        response.raise_for_status()
                        bytes_downloaded = 0
                        with open(final_path, "wb") as f:
                            for chunk in response.iter_content(
                                chunk_size=HTTP_STREAM_CHUNK_SIZE
                            ):
                                if chunk:
                                    f.write(chunk)
                                    bytes_downloaded += len(chunk)
                        _remove_other_mp4_in_folder(folder_path, final_path)
                        logger.debug(
                            "Downloaded clip for %s (%s bytes) to %s",
                            event_id,
                            bytes_downloaded,
                            final_path,
                        )
                        return {"success": True, "clip_path": final_path}
                except requests.exceptions.HTTPError as e:
                    if e.response is not None:
                        if e.response.status_code == 404:
                            logger.warning(
                                "No recording available for event %s", event_id
                            )
                            return {"success": False, "clip_path": None}
                        if e.response.status_code == 400 and attempt < 3:
                            logger.warning(
                                "Clip not ready for %s (HTTP 400), "
                                "retrying in 5s (%s/3)",
                                event_id,
                                attempt,
                            )
                            time.sleep(5)
                            continue
                    raise
            return {"success": False, "clip_path": None}
        except requests.exceptions.Timeout:
            logger.error("Timeout downloading clip for %s", event_id)
            return {"success": False, "clip_path": None}
        except requests.exceptions.HTTPError as e:
            logger.error("HTTP error downloading clip for %s: %s", event_id, e)
            return {"success": False, "clip_path": None}
        except Exception as e:
            logger.exception("Failed to download clip for %s: %s", event_id, e)
            if os.path.exists(final_path):
                try:
                    os.remove(final_path)
                except OSError:
                    pass
            return {"success": False, "clip_path": None}

    def download_clip_to_temp(
        self, event_id: str, folder_path: str, camera: str
    ) -> dict:
        """
        Download clip from Frigate events API to dynamic path (no transcode).
        Returns dict: success (bool), clip_path (str | None). Orphan GC after write.
        """
        return self._download_clip_events_api_to_path(event_id, folder_path, camera)

    def download_and_transcode_clip(
        self, event_id: str, folder_path: str, camera: str
    ) -> bool:
        """Download clip from Frigate events API to dynamic path (fallback when
        Export API fails). No transcode."""
        result = self._download_clip_events_api_to_path(event_id, folder_path, camera)
        return result.get("success", False)

    def fetch_review_summary(
        self,
        start_ts: float,
        end_ts: float,
        padding_before: float,
        padding_after: float,
    ) -> str | None:
        """Fetch review summary from Frigate API with time padding."""
        padded_start = int(start_ts - padding_before)
        padded_end = int(end_ts + padding_after)

        url = (
            f"{self.frigate_url}/api/review/summarize/"
            f"start/{padded_start}/end/{padded_end}"
        )
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
        """POST event description to Frigate API for storage in Frigate's DB."""
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
            logger.warning(
                f"Frigate description API returned {response.status_code} "
                f"for {event_id}",
            )
            return False
        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to post description to Frigate for {event_id}: {e}")
            return False
