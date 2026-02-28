"""
Event Query Service - Handles reading and parsing event data from the filesystem.
"""

import json
import logging
import os
import re
import time
from collections import OrderedDict
from typing import Any

from frigate_buffer.constants import NON_CAMERA_DIRS

logger = logging.getLogger("frigate-buffer")


def resolve_clip_in_folder(folder_path: str) -> str | None:
    """
    Return the basename of the clip file in folder_path, or None if no clip.

    Lists *.mp4 in folder_path; if multiple, returns the newest by mtime
    (most recently modified).
    """
    try:
        candidates = [
            os.path.join(folder_path, n)
            for n in os.listdir(folder_path)
            if n.lower().endswith(".mp4")
            and os.path.isfile(os.path.join(folder_path, n))
        ]
    except OSError:
        return None
    if not candidates:
        return None
    if len(candidates) == 1:
        return os.path.basename(candidates[0])
    newest = max(candidates, key=lambda p: os.path.getmtime(p))
    return os.path.basename(newest)


def read_timeline_merged(folder_path: str) -> dict[str, Any]:
    """
    Read timeline from folder: merge notification_timeline.json (if present)
    with notification_timeline_append.jsonl (append-only log). Returns
    dict with 'event_id' and 'entries' (list).
    """
    data = {"event_id": None, "entries": []}
    base_path = os.path.join(folder_path, "notification_timeline.json")
    append_path = os.path.join(folder_path, "notification_timeline_append.jsonl")
    if os.path.exists(base_path):
        try:
            with open(base_path) as f:
                base = json.load(f)
            data["event_id"] = base.get("event_id")
            data["entries"] = list(base.get("entries") or [])
        except (OSError, json.JSONDecodeError):
            pass
    if data["event_id"] is None:
        folder_name = os.path.basename(folder_path)
        parts = folder_name.split("_", 1)
        data["event_id"] = parts[1] if len(parts) > 1 else folder_name
    if os.path.exists(append_path):
        try:
            with open(append_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        data["entries"].append(entry)
                        if data["event_id"] is None and entry.get("data", {}).get(
                            "event_id"
                        ):
                            data["event_id"] = entry["data"]["event_id"]
                    except json.JSONDecodeError:
                        continue
        except OSError:
            pass
    return data


class EventQueryService:
    """Service for querying events from the filesystem."""

    def __init__(
        self, storage_path: str, cache_ttl: int = 5, event_cache_max: int = 500
    ):
        self.storage_path = storage_path
        self._cache = {}
        self._cache_ttl = cache_ttl
        self._event_cache: OrderedDict = OrderedDict()  # LRU keyed by folder
        self._event_cache_max = event_cache_max

    def _get_cached(self, key: str) -> Any | None:
        if key in self._cache:
            entry = self._cache[key]
            if time.monotonic() - entry["timestamp"] < self._cache_ttl:
                return entry["data"]
        return None

    def _set_cache(self, key: str, data: Any):
        self._cache[key] = {"timestamp": time.monotonic(), "data": data}

    def evict_cache(self, key: str) -> None:
        """Remove a key from the list cache so the next request refetches.
        Used for test_events after Send prompt to AI."""
        self._cache.pop(key, None)

    def _get_event_cached(self, folder_path: str, mtime: float) -> dict[str, Any]:
        """Get parsed event data from cache if valid, otherwise parse and cache
        (LRU eviction when over cap)."""
        if folder_path in self._event_cache:
            self._event_cache.move_to_end(folder_path)
            entry = self._event_cache[folder_path]
            if entry["mtime"] == mtime:
                return entry["data"]

        # Cache miss or invalid
        data = self._parse_event_files(folder_path)
        self._event_cache[folder_path] = {"mtime": mtime, "data": data}
        if len(self._event_cache) > self._event_cache_max:
            self._event_cache.popitem(last=False)
        return data

    def _parse_event_files(self, folder_path: str) -> dict[str, Any]:
        """Read all event files in one go and return data dict."""
        data = {
            "summary_text": "Analysis pending...",
            "metadata": {},
            "timeline": {},
            "review_summary": None,
            "has_clip": False,
            "has_snapshot": False,
            "viewed": False,
        }

        try:
            with open(os.path.join(folder_path, "summary.txt")) as f:
                data["summary_text"] = f.read().strip()
        except (OSError, FileNotFoundError):
            pass

        try:
            with open(os.path.join(folder_path, "metadata.json")) as f:
                data["metadata"] = json.load(f)
        except (OSError, FileNotFoundError, json.JSONDecodeError):
            pass

        try:
            data["timeline"] = read_timeline_merged(folder_path)
        except (OSError, FileNotFoundError, json.JSONDecodeError):
            pass

        try:
            with open(os.path.join(folder_path, "review_summary.md")) as f:
                data["review_summary"] = f.read().strip()
        except (OSError, FileNotFoundError):
            pass

        clip_basename = resolve_clip_in_folder(folder_path)
        data["has_clip"] = clip_basename is not None
        data["clip_basename"] = clip_basename
        data["has_snapshot"] = os.path.exists(os.path.join(folder_path, "snapshot.jpg"))
        data["viewed"] = os.path.exists(os.path.join(folder_path, ".viewed"))

        # Scan subdirectories (for consolidated events)
        subdirs_map = {}
        try:
            with os.scandir(folder_path) as it:
                for entry in it:
                    if entry.is_dir() and not entry.name.startswith("."):
                        sub_clip_basename = resolve_clip_in_folder(entry.path)
                        sub_snap = os.path.join(entry.path, "snapshot.jpg")
                        sub_meta = os.path.join(entry.path, "metadata.json")

                        sub_data = {
                            "has_clip": sub_clip_basename is not None,
                            "clip_basename": sub_clip_basename,
                            "has_snapshot": os.path.exists(sub_snap),
                            "metadata": None,
                        }

                        if os.path.exists(sub_meta):
                            try:
                                with open(sub_meta) as f:
                                    sub_data["metadata"] = json.load(f)
                            except (OSError, FileNotFoundError, json.JSONDecodeError):
                                pass

                        subdirs_map[entry.name] = sub_data
        except OSError:
            pass

        data["subdirs"] = subdirs_map
        return data

    def _parse_summary(self, summary_text: str) -> dict[str, str]:
        """Parse key-value pairs from summary.txt format."""
        parsed = {}
        for line in summary_text.split("\n"):
            if ":" in line:
                key, _, value = line.partition(":")
                parsed[key.strip()] = value.strip()
        return parsed

    def _extract_genai_entries(
        self, timeline_data: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Extract GenAI metadata entries from notification_timeline.json.
        Identical descriptions (same title, shortSummary, scene) are deduplicated
        so the AI Analysis section shows each unique analysis once; the raw timeline
        file is unchanged for debugging."""
        entries = []
        seen = set()
        data = timeline_data

        for e in data.get("entries", []):
            payload = (e.get("data") or {}).get("payload") or {}
            if payload.get("type") != "genai":
                continue
            after = payload.get("after") or {}
            meta = (after.get("data") or {}).get("metadata")
            if not meta:
                continue
            title = meta.get("title") or ""
            scene = meta.get("scene") or ""
            short_summary = meta.get("shortSummary") or meta.get("description") or ""
            # Skip boilerplate "no concerns/activity" entries to avoid noise
            lower = (title + " " + short_summary + " " + scene).lower()
            if "no concerns" in lower or "no activity" in lower:
                if not title and not scene and len(short_summary) < 80:
                    continue
            content_key = (title.strip(), short_summary.strip(), scene.strip())
            if content_key in seen:
                continue
            seen.add(content_key)
            entries.append(
                {
                    "title": title,
                    "scene": scene,
                    "shortSummary": short_summary,
                    "time": meta.get("time"),
                    "potential_threat_level": meta.get("potential_threat_level", 0),
                }
            )
        return entries

    def _event_ended_in_timeline(self, timeline_data: dict[str, Any]) -> bool:
        """Check if event has ended based on timeline (Event end from Frigate,
        or end_time set)."""
        data = timeline_data

        for e in data.get("entries", []):
            label = (e.get("label") or "").lower()
            if "event end" in label:
                return True
            payload = (e.get("data") or {}).get("payload") or {}
            if payload.get("type") == "end":
                return True
            after = payload.get("after") or {}
            if after.get("end_time") is not None:
                return True
        return False

    def _extract_end_timestamp_from_timeline(
        self, timeline_data: dict[str, Any]
    ) -> float | None:
        """Return the latest end_time from timeline entries
        (payload.after.end_time for Frigate, or data.end_time for test_ai_prompt
        entries). Used so the player can show event end time and duration. We use
        the maximum so consolidated events and multiple event-end MQTT messages
        show the true span. Regular events only have Frigate entries; test events
        may have test_ai_prompt."""
        end_times: list[float] = []
        for e in (timeline_data or {}).get("entries", []):
            # Frigate event end (regular events only)
            payload = (e.get("data") or {}).get("payload") or {}
            after = payload.get("after") or {}
            end_time = after.get("end_time")
            if end_time is not None:
                try:
                    end_times.append(float(end_time))
                except (TypeError, ValueError):
                    pass
            # Test-event-only: Send prompt to AI writes this source
            if e.get("source") == "test_ai_prompt":
                end_time = (e.get("data") or {}).get("end_time")
                if end_time is not None:
                    try:
                        end_times.append(float(end_time))
                    except (TypeError, ValueError):
                        pass
        return max(end_times) if end_times else None

    def _extract_cameras_zones_from_timeline(
        self, timeline_data: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Extract cameras and zones from frigate_mqtt entries in
        notification_timeline.json."""
        data = timeline_data

        camera_zones = {}
        for e in data.get("entries", []):
            if e.get("source") != "frigate_mqtt":
                continue
            payload = (e.get("data") or {}).get("payload") or {}
            after = payload.get("after") or payload.get("before") or {}
            camera = after.get("camera")
            if not camera:
                continue
            zones = camera_zones.setdefault(camera, set())
            for z in after.get("entered_zones") or []:
                if z:
                    zones.add(z)
            for z in after.get("current_zones") or []:
                if z:
                    zones.add(z)
        return [
            {"camera": cam, "zones": sorted(zones)}
            for cam, zones in sorted(camera_zones.items())
        ]

    def _get_consolidated_events(self) -> list[dict[str, Any]]:
        """Get consolidated events from events/{ce_id}/{camera}/ structure."""
        events_dir = os.path.join(self.storage_path, "events")
        events_list = []
        if not os.path.isdir(events_dir):
            return events_list

        entries = []
        try:
            with os.scandir(events_dir) as it:
                entries = sorted(
                    [e for e in it if e.is_dir()], key=lambda e: e.name, reverse=True
                )
        except OSError:
            pass

        for entry in entries:
            ce_id = entry.name
            # Consolidated event folders must follow pattern: {timestamp}_{id}
            # Strictly enforce to prevent non-event folders appearing as events
            parts = ce_id.split("_", 1)
            if len(parts) < 2 or not parts[0].isdigit():
                continue
            # Skip test folders (test1, test2, etc.)
            if re.match(r"^test\d+$", ce_id):
                continue

            # Cache key uses content mtime so adding a clip (*.mp4) in any
            # camera subdir invalidates cache
            content_mtime = entry.stat().st_mtime
            try:
                with os.scandir(entry.path) as it:
                    for child in it:
                        if child.is_dir() and not child.name.startswith("."):
                            try:
                                content_mtime = max(
                                    content_mtime, child.stat().st_mtime
                                )
                            except OSError:
                                pass
            except OSError:
                pass
            data = self._get_event_cached(entry.path, content_mtime)

            parts = ce_id.split("_", 1)
            ts = parts[0] if len(parts) > 0 else "0"

            summary_text = data.get("summary_text", "Analysis pending...")
            metadata = data.get("metadata", {}) or {}
            timeline = data.get("timeline", {})
            review_summary = data.get("review_summary")
            viewed = data.get("viewed", False)
            subdirs = data.get("subdirs", {})

            parsed = self._parse_summary(summary_text)

            cameras = sorted(subdirs.keys())

            # Consolidated: metadata may live in primary camera folder
            if not metadata and cameras:
                for cam in cameras:
                    cam_meta = subdirs.get(cam, {}).get("metadata")
                    if cam_meta:
                        metadata = cam_meta
                        break

            genai_entries = self._extract_genai_entries(timeline)

            primary_clip = primary_snapshot = None
            hosted_clips: list[dict[str, str]] = []
            for cam in cameras:
                cam_data = subdirs.get(cam, {})
                clip_basename = cam_data.get("clip_basename")
                if cam_data.get("has_clip") and clip_basename:
                    url = f"/files/events/{ce_id}/{cam}/{clip_basename}"
                    if primary_clip is None:
                        primary_clip = url
                    hosted_clips.append({"camera": cam, "url": url})
            for cam in cameras:
                cam_data = subdirs.get(cam, {})
                if cam_data.get("has_snapshot"):
                    primary_snapshot = f"/files/events/{ce_id}/{cam}/snapshot.jpg"
                    break

            # When the edited summary video exists in the CE root, add it to
            # the dropdown and make it primary.
            summary_basename = f"{ce_id}_summary.mp4"
            summary_path = os.path.join(entry.path, summary_basename)
            if os.path.isfile(summary_path):
                summary_url = f"/files/events/{ce_id}/{summary_basename}"
                hosted_clips.append({"camera": "Summary video", "url": summary_url})
                primary_clip = summary_url

            has_clip = any(subdirs.get(cam, {}).get("has_clip") for cam in cameras)
            has_summary = os.path.isfile(summary_path)
            if has_summary:
                has_clip = True
            has_snapshot = any(
                subdirs.get(cam, {}).get("has_snapshot") for cam in cameras
            )

            # hosted_clip: only set when we have an actual clip (camera or
            # summary); no fallback to non-existent file
            hosted_clip_url = primary_clip if has_clip else None
            hosted_snapshot_url = (
                primary_snapshot
                if has_snapshot
                else (
                    f"/files/events/{ce_id}/{cameras[0]}/snapshot.jpg"
                    if cameras
                    else None
                )
            )

            # "Not ongoing": event ended in timeline, or has clip, or older than 90 min
            event_ended = self._event_ended_in_timeline(timeline)
            try:
                ts_float = float(ts)
                age_seconds = time.time() - ts_float
                ongoing = (
                    not has_clip and age_seconds < 5400 and not event_ended
                )  # 90 min
            except (ValueError, TypeError):
                ongoing = not has_clip and not event_ended

            cameras_with_zones = self._extract_cameras_zones_from_timeline(timeline)
            if not cameras_with_zones and cameras:
                cameras_with_zones = [{"camera": c, "zones": []} for c in cameras]

            end_ts = self._extract_end_timestamp_from_timeline(
                timeline
            ) or metadata.get("end_time")
            event_dict = {
                "event_id": ce_id,
                "camera": "events",
                "subdir": ce_id,
                "timestamp": ts,
                "summary": summary_text,
                "title": metadata.get("genai_title") or parsed.get("Title"),
                "description": metadata.get("genai_description")
                or parsed.get("Description")
                or parsed.get("AI Description"),
                "scene": metadata.get("genai_scene") or parsed.get("Scene"),
                "label": metadata.get("label") or parsed.get("Label", "unknown"),
                "severity": metadata.get("severity") or parsed.get("Severity"),
                "threat_level": metadata.get("threat_level", 0),
                "review_summary": review_summary,
                "has_clip": has_clip,
                "has_snapshot": has_snapshot,
                "viewed": viewed,
                "hosted_clip": hosted_clip_url,
                "hosted_snapshot": hosted_snapshot_url,
                "hosted_clips": hosted_clips,
                "cameras": cameras,
                "cameras_with_zones": cameras_with_zones,
                "consolidated": True,
                "ongoing": ongoing,
                "genai_entries": genai_entries,
            }
            if end_ts is not None:
                event_dict["end_timestamp"] = end_ts
            events_list.append(event_dict)

        return events_list

    def _get_camera_events(self, camera_name: str) -> list[dict[str, Any]]:
        """Helper to get events for a specific camera."""
        camera_path = os.path.join(self.storage_path, camera_name)
        events = []

        if not os.path.isdir(camera_path):
            return events

        entries = []
        try:
            with os.scandir(camera_path) as it:
                entries = sorted(
                    [e for e in it if e.is_dir()], key=lambda e: e.name, reverse=True
                )
        except OSError:
            pass

        for entry in entries:
            subdir = entry.name

            # Get cached data
            data = self._get_event_cached(entry.path, entry.stat().st_mtime)

            parts = subdir.split("_", 1)
            ts = parts[0] if len(parts) > 0 else "0"
            eid = parts[1] if len(parts) > 1 else subdir

            summary_text = data.get("summary_text", "Analysis pending...")
            metadata = data.get("metadata", {}) or {}
            timeline = data.get("timeline", {})
            review_summary = data.get("review_summary")

            has_clip = data.get("has_clip", False)
            has_snapshot = data.get("has_snapshot", False)
            viewed = data.get("viewed", False)

            parsed = self._parse_summary(summary_text)

            event_ended = self._event_ended_in_timeline(timeline)
            try:
                ts_float = float(ts)
                age_seconds = time.time() - ts_float
                ongoing = not has_clip and age_seconds < 5400 and not event_ended
            except (ValueError, TypeError):
                ongoing = not has_clip and not event_ended

            genai_entries = self._extract_genai_entries(timeline)

            cameras_with_zones = self._extract_cameras_zones_from_timeline(timeline)
            if not cameras_with_zones:
                cameras_with_zones = [{"camera": camera_name, "zones": []}]

            end_ts = self._extract_end_timestamp_from_timeline(
                timeline
            ) or metadata.get("end_time")
            clip_basename = data.get("clip_basename")
            clip_url = (
                f"/files/{camera_name}/{subdir}/{clip_basename}"
                if (has_clip and clip_basename)
                else None
            )
            legacy_hosted_clips = (
                [{"camera": camera_name, "url": clip_url}] if has_clip else []
            )
            event_dict = {
                "event_id": eid,
                "camera": camera_name,
                "subdir": subdir,
                "timestamp": ts,
                "summary": summary_text,
                "title": metadata.get("genai_title") or parsed.get("Title"),
                "description": metadata.get("genai_description")
                or parsed.get("Description")
                or parsed.get("AI Description"),
                "scene": metadata.get("genai_scene") or parsed.get("Scene"),
                "label": metadata.get("label") or parsed.get("Label", "unknown"),
                "severity": metadata.get("severity") or parsed.get("Severity"),
                "threat_level": metadata.get("threat_level", 0),
                "review_summary": review_summary,
                "has_clip": has_clip,
                "has_snapshot": has_snapshot,
                "viewed": viewed,
                "hosted_clip": clip_url,
                "hosted_snapshot": f"/files/{camera_name}/{subdir}/snapshot.jpg"
                if has_snapshot
                else None,
                "hosted_clips": legacy_hosted_clips,
                "cameras_with_zones": cameras_with_zones,
                "ongoing": ongoing,
                "genai_entries": genai_entries,
            }
            if end_ts is not None:
                event_dict["end_timestamp"] = end_ts
            events.append(event_dict)

        return events

    def get_events(self, camera_name: str) -> list[dict[str, Any]]:
        """Get events for a specific camera or 'events' for consolidated events."""
        cache_key = f"events_{camera_name}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        if camera_name == "events":
            events = self._get_consolidated_events()
        else:
            events = self._get_camera_events(camera_name)

        self._set_cache(cache_key, events)
        return events

    def get_all_events(self) -> tuple[list[dict[str, Any]], list[str]]:
        """Get all events across all cameras (global view)."""
        cache_key = "all_events"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        all_events = []
        cameras_found = []

        try:
            for camera_dir in os.listdir(self.storage_path):
                camera_path = os.path.join(self.storage_path, camera_dir)
                if not os.path.isdir(camera_path):
                    continue
                if camera_dir.split("_")[0].isdigit():
                    continue
                if camera_dir in NON_CAMERA_DIRS:
                    continue

                cameras_found.append(camera_dir)
                events = self.get_events(camera_dir)
                all_events.extend(events)

            all_events.sort(key=lambda x: x["timestamp"], reverse=True)

        except Exception as e:
            logger.error(f"Error listing events: {e}")
            raise

        result = (all_events, sorted(cameras_found))
        self._set_cache(cache_key, result)
        return result

    def get_saved_events(self, camera: str | None = None) -> list[dict[str, Any]]:
        """
        List events under storage_path/saved/ (user-kept events).
        Optional camera filters to saved/camera only. Each event dict has saved=True
        and file URLs under /files/saved/<camera>/<subdir>/...
        """
        cache_key = "saved_events" if camera is None else f"saved_events_{camera}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        saved_path = os.path.join(self.storage_path, "saved")
        all_events: list[dict[str, Any]] = []

        if not os.path.isdir(saved_path):
            self._set_cache(cache_key, all_events)
            return all_events

        try:
            top_dirs = os.listdir(saved_path)
            if camera is not None:
                top_dirs = [d for d in top_dirs if d == camera]
            for camera_name in top_dirs:
                camera_path = os.path.join(saved_path, camera_name)
                if not os.path.isdir(camera_path):
                    continue
                entries = []
                try:
                    with os.scandir(camera_path) as it:
                        entries = sorted(
                            [e for e in it if e.is_dir()],
                            key=lambda e: e.name,
                            reverse=True,
                        )
                except OSError:
                    continue
                for entry in entries:
                    subdir = entry.name
                    content_mtime = entry.stat().st_mtime
                    data = self._get_event_cached(entry.path, content_mtime)
                    subdirs_map = data.get("subdirs", {})
                    if camera_name == "events" and subdirs_map:
                        event_dict = self._build_saved_consolidated_event_dict(
                            entry.path, subdir, data, content_mtime
                        )
                    else:
                        event_dict = self._build_saved_single_cam_event_dict(
                            camera_name, subdir, data
                        )
                    if event_dict is not None:
                        event_dict["saved"] = True
                        all_events.append(event_dict)
        except Exception as e:
            logger.error("Error listing saved events: %s", e)

        all_events.sort(key=lambda x: x["timestamp"], reverse=True)
        self._set_cache(cache_key, all_events)
        return all_events

    def _build_saved_consolidated_event_dict(
        self, folder_path: str, ce_id: str, data: dict[str, Any], content_mtime: float
    ) -> dict[str, Any] | None:
        """Build event dict for a saved consolidated event (saved/events/ce_id)."""
        subdirs = data.get("subdirs", {})
        cameras = sorted(subdirs.keys())
        if not cameras:
            return None
        parts = ce_id.split("_", 1)
        ts = parts[0] if len(parts) > 0 else "0"
        summary_text = data.get("summary_text", "Analysis pending...")
        metadata = data.get("metadata", {}) or {}
        timeline = data.get("timeline", {})
        review_summary = data.get("review_summary")
        viewed = data.get("viewed", False)
        parsed = self._parse_summary(summary_text)
        genai_entries = self._extract_genai_entries(timeline)
        if not metadata and cameras:
            for cam in cameras:
                cam_meta = subdirs.get(cam, {}).get("metadata")
                if cam_meta:
                    metadata = cam_meta
                    break
        files_prefix = "saved/events"
        primary_clip = primary_snapshot = None
        hosted_clips: list[dict[str, str]] = []
        for cam in cameras:
            cam_data = subdirs.get(cam, {})
            clip_basename = cam_data.get("clip_basename")
            if cam_data.get("has_clip") and clip_basename:
                url = f"/files/{files_prefix}/{ce_id}/{cam}/{clip_basename}"
                if primary_clip is None:
                    primary_clip = url
                hosted_clips.append({"camera": cam, "url": url})
        for cam in cameras:
            if subdirs.get(cam, {}).get("has_snapshot"):
                primary_snapshot = f"/files/{files_prefix}/{ce_id}/{cam}/snapshot.jpg"
                break
        summary_basename = f"{ce_id}_summary.mp4"
        summary_path = os.path.join(folder_path, summary_basename)
        if os.path.isfile(summary_path):
            summary_url = f"/files/{files_prefix}/{ce_id}/{summary_basename}"
            hosted_clips.append({"camera": "Summary video", "url": summary_url})
            primary_clip = summary_url
        has_clip = any(
            subdirs.get(c, {}).get("has_clip") for c in cameras
        ) or os.path.isfile(summary_path)
        has_snapshot = any(subdirs.get(c, {}).get("has_snapshot") for c in cameras)
        hosted_clip_url = primary_clip if has_clip else None
        hosted_snapshot_url = (
            primary_snapshot
            if has_snapshot
            else (
                f"/files/{files_prefix}/{ce_id}/{cameras[0]}/snapshot.jpg"
                if cameras
                else None
            )
        )
        event_ended = self._event_ended_in_timeline(timeline)
        try:
            ts_float = float(ts)
            ongoing = (
                not has_clip and (time.time() - ts_float) < 5400 and not event_ended
            )
        except (ValueError, TypeError):
            ongoing = not has_clip and not event_ended
        cameras_with_zones = self._extract_cameras_zones_from_timeline(timeline)
        if not cameras_with_zones and cameras:
            cameras_with_zones = [{"camera": c, "zones": []} for c in cameras]
        end_ts = self._extract_end_timestamp_from_timeline(timeline) or metadata.get(
            "end_time"
        )
        event_dict: dict[str, Any] = {
            "event_id": ce_id,
            "camera": "events",
            "subdir": ce_id,
            "timestamp": ts,
            "summary": summary_text,
            "title": metadata.get("genai_title") or parsed.get("Title"),
            "description": metadata.get("genai_description")
            or parsed.get("Description")
            or parsed.get("AI Description"),
            "scene": metadata.get("genai_scene") or parsed.get("Scene"),
            "label": metadata.get("label") or parsed.get("Label", "unknown"),
            "severity": metadata.get("severity") or parsed.get("Severity"),
            "threat_level": metadata.get("threat_level", 0),
            "review_summary": review_summary,
            "has_clip": has_clip,
            "has_snapshot": has_snapshot,
            "viewed": viewed,
            "hosted_clip": hosted_clip_url,
            "hosted_snapshot": hosted_snapshot_url,
            "hosted_clips": hosted_clips,
            "cameras": cameras,
            "cameras_with_zones": cameras_with_zones,
            "consolidated": True,
            "ongoing": ongoing,
            "genai_entries": genai_entries,
        }
        if end_ts is not None:
            event_dict["end_timestamp"] = end_ts
        return event_dict

    def _build_saved_single_cam_event_dict(
        self, camera_name: str, subdir: str, data: dict[str, Any]
    ) -> dict[str, Any]:
        """Build event dict for a saved CE folder with one camera
        (saved/camera/subdir). Pipeline is always CE; folder layout only."""
        parts = subdir.split("_", 1)
        ts = parts[0] if len(parts) > 0 else "0"
        eid = parts[1] if len(parts) > 1 else subdir
        summary_text = data.get("summary_text", "Analysis pending...")
        metadata = data.get("metadata", {}) or {}
        timeline = data.get("timeline", {})
        review_summary = data.get("review_summary")
        has_clip = data.get("has_clip", False)
        has_snapshot = data.get("has_snapshot", False)
        viewed = data.get("viewed", False)
        parsed = self._parse_summary(summary_text)
        event_ended = self._event_ended_in_timeline(timeline)
        try:
            ts_float = float(ts)
            ongoing = (
                not has_clip and (time.time() - ts_float) < 5400 and not event_ended
            )
        except (ValueError, TypeError):
            ongoing = not has_clip and not event_ended
        genai_entries = self._extract_genai_entries(timeline)
        cameras_with_zones = self._extract_cameras_zones_from_timeline(timeline)
        if not cameras_with_zones:
            cameras_with_zones = [{"camera": camera_name, "zones": []}]
        end_ts = self._extract_end_timestamp_from_timeline(timeline) or metadata.get(
            "end_time"
        )
        clip_basename = data.get("clip_basename")
        files_prefix = f"saved/{camera_name}"
        clip_url = (
            f"/files/{files_prefix}/{subdir}/{clip_basename}"
            if (has_clip and clip_basename)
            else None
        )
        legacy_hosted_clips = (
            [{"camera": camera_name, "url": clip_url}] if has_clip else []
        )
        event_dict: dict[str, Any] = {
            "event_id": eid,
            "camera": camera_name,
            "subdir": subdir,
            "timestamp": ts,
            "summary": summary_text,
            "title": metadata.get("genai_title") or parsed.get("Title"),
            "description": metadata.get("genai_description")
            or parsed.get("Description")
            or parsed.get("AI Description"),
            "scene": metadata.get("genai_scene") or parsed.get("Scene"),
            "label": metadata.get("label") or parsed.get("Label", "unknown"),
            "severity": metadata.get("severity") or parsed.get("Severity"),
            "threat_level": metadata.get("threat_level", 0),
            "review_summary": review_summary,
            "has_clip": has_clip,
            "has_snapshot": has_snapshot,
            "viewed": viewed,
            "hosted_clip": clip_url,
            "hosted_snapshot": f"/files/{files_prefix}/{subdir}/snapshot.jpg"
            if has_snapshot
            else None,
            "hosted_clips": legacy_hosted_clips,
            "cameras_with_zones": cameras_with_zones,
            "ongoing": ongoing,
            "genai_entries": genai_entries,
        }
        if end_ts is not None:
            event_dict["end_timestamp"] = end_ts
        return event_dict

    def get_test_events(self) -> list[dict[str, Any]]:
        """
        List only test run events (events/test1, events/test2, ...).
        Same event dict shape as consolidated events; used when filter=test_events.
        """
        cache_key = "test_events"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        events_dir = os.path.join(self.storage_path, "events")
        events_list: list[dict[str, Any]] = []
        if not os.path.isdir(events_dir):
            self._set_cache(cache_key, events_list)
            return events_list

        entries = []
        try:
            with os.scandir(events_dir) as it:
                entries = sorted(
                    [e for e in it if e.is_dir() and re.match(r"^test\d+$", e.name)],
                    key=lambda e: e.stat().st_mtime,
                    reverse=True,
                )
        except OSError:
            pass

        for entry in entries:
            ce_id = entry.name
            content_mtime = entry.stat().st_mtime
            try:
                with os.scandir(entry.path) as it:
                    for child in it:
                        if child.is_dir() and not child.name.startswith("."):
                            try:
                                content_mtime = max(
                                    content_mtime, child.stat().st_mtime
                                )
                            except OSError:
                                pass
            except OSError:
                pass
            data = self._get_event_cached(entry.path, content_mtime)
            subdirs = data.get("subdirs", {})
            cameras = sorted(subdirs.keys())
            if not cameras:
                continue
            # Use folder mtime as event date so test events show correct
            # date/length (regular events unchanged)
            ts = str(int(content_mtime))
            summary_text = data.get("summary_text", "Analysis pending...")
            metadata = data.get("metadata", {}) or {}
            timeline = data.get("timeline", {})
            review_summary = data.get("review_summary")
            viewed = data.get("viewed", False)
            parsed = self._parse_summary(summary_text)
            genai_entries = self._extract_genai_entries(timeline)
            if not metadata and cameras:
                for cam in cameras:
                    cam_meta = subdirs.get(cam, {}).get("metadata")
                    if cam_meta:
                        metadata = cam_meta
                        break
            primary_clip = primary_snapshot = None
            hosted_clips = []
            for cam in cameras:
                cam_data = subdirs.get(cam, {})
                clip_basename = cam_data.get("clip_basename")
                if cam_data.get("has_clip") and clip_basename:
                    url = f"/files/events/{ce_id}/{cam}/{clip_basename}"
                    if primary_clip is None:
                        primary_clip = url
                    hosted_clips.append({"camera": cam, "url": url})
            for cam in cameras:
                if subdirs.get(cam, {}).get("has_snapshot"):
                    primary_snapshot = f"/files/events/{ce_id}/{cam}/snapshot.jpg"
                    break
            summary_basename = f"{ce_id}_summary.mp4"
            summary_path = os.path.join(entry.path, summary_basename)
            if os.path.isfile(summary_path):
                hosted_clips.append(
                    {
                        "camera": "Summary video",
                        "url": f"/files/events/{ce_id}/{summary_basename}",
                    }
                )
                primary_clip = f"/files/events/{ce_id}/{summary_basename}"
            has_clip = any(
                subdirs.get(c, {}).get("has_clip") for c in cameras
            ) or os.path.isfile(summary_path)
            has_snapshot = any(subdirs.get(c, {}).get("has_snapshot") for c in cameras)
            hosted_clip_url = primary_clip if has_clip else None
            hosted_snapshot_url = (
                primary_snapshot
                if has_snapshot
                else (
                    f"/files/events/{ce_id}/{cameras[0]}/snapshot.jpg"
                    if cameras
                    else None
                )
            )
            event_ended = self._event_ended_in_timeline(timeline)
            ongoing = not has_clip and not event_ended
            cameras_with_zones = self._extract_cameras_zones_from_timeline(timeline)
            if not cameras_with_zones and cameras:
                cameras_with_zones = [{"camera": c, "zones": []} for c in cameras]
            end_ts = self._extract_end_timestamp_from_timeline(
                timeline
            ) or metadata.get("end_time")
            event_dict = {
                "event_id": ce_id,
                "camera": "events",
                "subdir": ce_id,
                "timestamp": ts,
                "summary": summary_text,
                "title": metadata.get("genai_title") or parsed.get("Title"),
                "description": metadata.get("genai_description")
                or parsed.get("Description")
                or parsed.get("AI Description"),
                "scene": metadata.get("genai_scene") or parsed.get("Scene"),
                "label": metadata.get("label") or parsed.get("Label", "unknown"),
                "severity": metadata.get("severity") or parsed.get("Severity"),
                "threat_level": metadata.get("threat_level", 0),
                "review_summary": review_summary,
                "has_clip": has_clip,
                "has_snapshot": has_snapshot,
                "viewed": viewed,
                "hosted_clip": hosted_clip_url,
                "hosted_snapshot": hosted_snapshot_url,
                "hosted_clips": hosted_clips,
                "cameras": cameras,
                "cameras_with_zones": cameras_with_zones,
                "consolidated": True,
                "ongoing": ongoing,
                "genai_entries": genai_entries,
            }
            if end_ts is not None:
                event_dict["end_timestamp"] = end_ts
            events_list.append(event_dict)

        self._set_cache(cache_key, events_list)
        return events_list

    def get_cameras(self) -> list[str]:
        """List available cameras from storage."""
        cache_key = "cameras"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        active_cameras = []
        try:
            for item in os.listdir(self.storage_path):
                item_path = os.path.join(self.storage_path, item)
                if os.path.isdir(item_path):
                    # Skip if it starts with a numeric prefix (event folder pattern)
                    if item.split("_")[0].isdigit():
                        continue
                    # Skip known non-camera directories
                    if item in NON_CAMERA_DIRS:
                        continue
                    active_cameras.append(item)
        except Exception as e:
            logger.error(f"Error listing cameras: {e}")

        if (
            os.path.isdir(os.path.join(self.storage_path, "events"))
            and "events" not in active_cameras
        ):
            active_cameras.append("events")

        result = sorted(active_cameras)
        self._set_cache(cache_key, result)
        return result

    def get_unread_count(self) -> int:
        """
        Count event folders that do not have a .viewed file (lightweight scan).

        Uses os.scandir only; does not run _parse_event_files. Result is cached
        with the same TTL as list caches to avoid disk thrashing when the
        mobile app polls frequently.
        """
        cached = self._get_cached("unread_count")
        if cached is not None:
            return cached

        count = 0
        try:
            with os.scandir(self.storage_path) as it:
                for entry in it:
                    if not entry.is_dir():
                        continue
                    if entry.name in NON_CAMERA_DIRS:
                        continue
                    if entry.name.split("_")[0].isdigit():
                        continue
                    if entry.name == "events":
                        try:
                            with os.scandir(entry.path) as sub_it:
                                for sub in sub_it:
                                    if not sub.is_dir():
                                        continue
                                    if re.match(r"^test\d+$", sub.name):
                                        continue
                                    if not os.path.exists(
                                        os.path.join(sub.path, ".viewed")
                                    ):
                                        count += 1
                        except OSError:
                            pass
                    else:
                        try:
                            with os.scandir(entry.path) as sub_it:
                                for sub in sub_it:
                                    if sub.is_dir() and not os.path.exists(
                                        os.path.join(sub.path, ".viewed")
                                    ):
                                        count += 1
                        except OSError:
                            pass
        except OSError as e:
            logger.error("Error scanning for unread count: %s", e)

        self._set_cache("unread_count", count)
        return count
