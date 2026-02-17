"""
Frigate Export Watchdog - Removes completed exports from Frigate's export list
after verifying clips are in event folders, and verifies download links.
"""

import json
import logging
import os
import re
from typing import Any, Literal
from urllib.parse import urlparse, urlunparse

import requests

from frigate_buffer.services.query import read_timeline_merged

logger = logging.getLogger("frigate-buffer")

# Timeout for DELETE /api/export/{id} so the watchdog cannot hang on slow/stuck Frigate or network
DELETE_EXPORT_TIMEOUT = 10

# Cap link verification to avoid unbounded HEAD requests per run
MAX_LINK_CHECK_FOLDERS = 20
MAX_HEAD_REQUESTS = 120

# Outcome of a single delete request for run summary
DeleteOutcome = Literal["success", "already_gone", "failure"]


def _sanitize_camera_name(camera: str) -> str:
    """Sanitize camera name for filesystem (match FileManager)."""
    s = camera.lower().replace(" ", "_")
    s = re.sub(r"[^a-z0-9_]", "", s)
    return s or "unknown"


def _parse_export_response_entries(
    folder_path: str,
) -> list[tuple[str, str | None]]:
    """
    Read merged timeline (base + append JSONL) and return list of (export_id, camera_for_clip).
    camera_for_clip is None for single-camera event (clip at root); else camera name for consolidated (clip at camera/clip.mp4).
    """
    result: list[tuple[str, str | None]] = []
    try:
        data = read_timeline_merged(folder_path)
    except (OSError, json.JSONDecodeError) as e:
        logger.debug(f"Watchdog: could not read timeline in {folder_path}: {e}")
        return result

    entries = data.get("entries") or []
    for entry in entries:
        if entry.get("source") != "frigate_api" or entry.get("direction") != "in":
            continue
        label = entry.get("label") or ""
        if "Clip export response" not in label:
            continue
        frigate_response = (entry.get("data") or {}).get("frigate_response") or {}
        export_id = frigate_response.get("export_id") or frigate_response.get("id")
        if not export_id:
            continue

        # Consolidated: "Clip export response for {camera}"
        camera_for_clip: str | None = None
        if "Clip export response for " in label:
            camera_for_clip = label.replace("Clip export response for ", "").strip()

        result.append((str(export_id), camera_for_clip))
    return result


def _clip_path_for_entry(
    folder_path: str,
    camera_for_clip: str | None,
) -> str:
    """Path to clip.mp4 for this entry (single-cam or consolidated)."""
    if camera_for_clip is None:
        return os.path.join(folder_path, "clip.mp4")
    sub = _sanitize_camera_name(camera_for_clip)
    return os.path.join(folder_path, sub, "clip.mp4")


def _rel_path_from_storage(folder_path: str, storage_path: str) -> tuple[str, str] | None:
    """Return (camera, subdir) relative to storage_path, or None if not under storage."""
    try:
        folder_real = os.path.realpath(folder_path)
        storage_real = os.path.realpath(storage_path)
        if not folder_real.startswith(storage_real):
            return None
        rel = os.path.relpath(folder_real, storage_real)
        parts = rel.split(os.sep)
        if len(parts) >= 2:
            return (parts[0], parts[1])
        if len(parts) == 1:
            return (parts[0], "")
        return None
    except (ValueError, IndexError):
        return None


def _event_files_list(folder_path: str) -> list[str]:
    """Build same file list as timeline page: root files + camera_subdir/clip.mp4 etc."""
    files: list[str] = []
    try:
        for name in os.listdir(folder_path):
            fp = os.path.join(folder_path, name)
            if os.path.isfile(fp):
                files.append(name)
        for sub in os.listdir(folder_path):
            sub_fp = os.path.join(folder_path, sub)
            if os.path.isdir(sub_fp) and not sub.startswith("."):
                for sf in ("clip.mp4", "snapshot.jpg", "metadata.json", "summary.txt", "review_summary.md"):
                    if os.path.isfile(os.path.join(sub_fp, sf)):
                        files.append(f"{sub}/{sf}")
        files.sort()
    except OSError:
        pass
    return files


def _redact_url(url: str) -> str:
    """Return URL with password redacted for safe logging."""
    try:
        parsed = urlparse(url)
        if parsed.password is not None:
            safe_netloc = f"{parsed.username or ''}:***@{parsed.hostname or ''}"
            if parsed.port is not None:
                safe_netloc += f":{parsed.port}"
            parsed = parsed._replace(netloc=safe_netloc)
        return urlunparse(parsed)
    except Exception:
        return "(url parse error)"


def _response_body_for_log(resp: requests.Response, max_len: int = 500) -> str:
    """Return a safe, truncated representation of the response body for logging."""
    if not resp.text:
        return ""
    try:
        if resp.headers.get("content-type", "").startswith("application/json"):
            obj = resp.json()
            raw = json.dumps(obj) if isinstance(obj, dict) else str(obj)
        else:
            raw = resp.text
        return (raw[:max_len] + "...") if len(raw) > max_len else raw
    except Exception:
        return (resp.text[:max_len] + "...") if len(resp.text) > max_len else resp.text


def _delete_export_from_frigate(
    frigate_url: str,
    export_id: str,
) -> DeleteOutcome:
    """
    Call Frigate DELETE /api/export/{export_id}. Log success, already-removed, or failure.
    Returns outcome so run_once can aggregate counts for the run summary.
    """
    url = f"{frigate_url.rstrip('/')}/api/export/{export_id}"
    logger.debug("Frigate export delete: url=%s export_id=%s", _redact_url(url), export_id)
    try:
        resp = requests.delete(url, timeout=DELETE_EXPORT_TIMEOUT)
        body_str = _response_body_for_log(resp)
        if resp.status_code == 200:
            logger.info("Frigate export removed: export_id=%s success", export_id)
            if body_str:
                logger.debug("Frigate export delete response (200): %s", body_str)
            return "success"
        if resp.status_code in (404, 422):
            logger.info(
                "Frigate export already removed: export_id=%s (status=%s)",
                export_id,
                resp.status_code,
            )
            if body_str:
                logger.debug("Frigate export delete response (%s): %s", resp.status_code, body_str)
            return "already_gone"
        # Failure: extract reason and log warning
        reason = ""
        try:
            if resp.headers.get("content-type", "").startswith("application/json"):
                body = resp.json()
                if isinstance(body, dict):
                    reason = body.get("message") or body.get("detail")
                    if isinstance(reason, list):
                        reason = "; ".join(str(x) for x in reason)
                    reason = str(reason) if reason else ""
        except Exception:
            pass
        if not reason and resp.text:
            reason = resp.text[:200]
        logger.warning(
            "Frigate export delete error: export_id=%s status=%s%s",
            export_id,
            resp.status_code,
            f" reason={reason}" if reason else "",
        )
        if body_str:
            logger.debug("Frigate export delete response (%s): %s", resp.status_code, body_str)
        return "failure"
    except requests.exceptions.RequestException as e:
        logger.warning("Frigate export delete request failed: export_id=%s error=%s", export_id, e)
        return "failure"


def run_once(config: dict[str, Any]) -> None:
    """
    Run one pass of the export watchdog: discover exports from timeline data,
    verify clips are in event folders, delete from Frigate (with logging), verify download links.
    Logs a run summary (succeeded / failed / already removed) at info.
    """
    storage_path = config.get("STORAGE_PATH") or ""
    frigate_url = config.get("FRIGATE_URL") or ""
    buffer_ip = config.get("BUFFER_IP") or ""
    flask_port = config.get("FLASK_PORT", 5055)
    base_url = f"http://{buffer_ip}:{flask_port}" if buffer_ip else ""

    if not storage_path or not os.path.isdir(storage_path):
        logger.debug("Export watchdog: STORAGE_PATH missing or not a directory, skipping")
        return
    if not frigate_url:
        logger.debug("Export watchdog: FRIGATE_URL missing, skipping")
        return

    logger.debug("Export watchdog run started: frigate_url=%s", _redact_url(frigate_url))

    seen_export_ids: set[str] = set()
    events_checked_for_links: list[tuple[str, str, str]] = []  # (folder_path, camera, subdir)
    folders_with_timeline = 0
    export_entries_found = 0
    succeeded = 0
    already_gone = 0
    failed = 0

    try:
        with os.scandir(storage_path) as it:
            for camera_entry in it:
                if not camera_entry.is_dir():
                    continue
                camera_dir = camera_entry.name
                camera_path = camera_entry.path

                # Skip legacy flat event folders (timestamp_eventid at top level)
                first_part = camera_dir.split("_", 1)[0] if camera_dir else ""
                if first_part.isdigit():
                    continue

                try:
                    with os.scandir(camera_path) as it_events:
                        for event_entry in it_events:
                            if not event_entry.is_dir():
                                continue
                            event_dir = event_entry.name
                            event_path = event_entry.path
                            timeline_path = os.path.join(event_path, "notification_timeline.json")
                            if not os.path.isfile(timeline_path):
                                continue

                            folders_with_timeline += 1
                            pairs = _parse_export_response_entries(event_path)
                            export_entries_found += len(pairs)
                            for export_id, camera_for_clip in pairs:
                                if export_id in seen_export_ids:
                                    continue
                                clip_path = _clip_path_for_entry(event_path, camera_for_clip)
                                if not os.path.isfile(clip_path):
                                    continue
                                seen_export_ids.add(export_id)
                                outcome = _delete_export_from_frigate(frigate_url, export_id)
                                if outcome == "success":
                                    succeeded += 1
                                elif outcome == "already_gone":
                                    already_gone += 1
                                else:
                                    failed += 1

                            # Remember for link verification (any folder with timeline and at least one clip)
                            if pairs and any(
                                os.path.isfile(_clip_path_for_entry(event_path, c))
                                for _, c in pairs
                            ):
                                rel = _rel_path_from_storage(event_path, storage_path)
                                if rel:
                                    events_checked_for_links.append((event_path, rel[0], rel[1]))

                except OSError as e:
                    logger.debug("Export watchdog: scan %s: %s", camera_path, e)

    except OSError as e:
        logger.warning("Export watchdog: scan storage: %s", e)
        return

    logger.debug(
        "Export watchdog: folders_with_timeline=%s export_entries_found=%s deletes_attempted=%s",
        folders_with_timeline,
        export_entries_found,
        succeeded + already_gone + failed,
    )
    total_deletes = succeeded + already_gone + failed
    if total_deletes == 0:
        logger.info("Export watchdog complete: no exports to delete")
    else:
        logger.info(
            "Export watchdog complete: %s succeeded, %s failed, %s already removed",
            succeeded,
            failed,
            already_gone,
        )

    # Verify download links (capped to avoid unbounded HEAD requests)
    if base_url and events_checked_for_links:
        head_count = 0
        for folder_path, camera, subdir in events_checked_for_links[:MAX_LINK_CHECK_FOLDERS]:
            if head_count >= MAX_HEAD_REQUESTS:
                break
            for f in _event_files_list(folder_path):
                if head_count >= MAX_HEAD_REQUESTS:
                    break
                url = f"{base_url}/files/{camera}/{subdir}/{f}"
                try:
                    r = requests.head(url, timeout=5)
                    if r.status_code != 200:
                        logger.warning(f"Export watchdog: download link check failed: {url} status={r.status_code}")
                except requests.exceptions.RequestException as e:
                    logger.debug(f"Export watchdog: link check {url}: {e}")
                finally:
                    head_count += 1


if __name__ == "__main__":
    from frigate_buffer.config import load_config

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    config = load_config()
    run_once(config)
    logger.info("Export watchdog run complete.")
