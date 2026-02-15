"""
Event Query Service - Handles reading and parsing event data from the filesystem.
"""

import os
import json
import time
import logging
from typing import List, Dict, Optional, Tuple, Any

logger = logging.getLogger('frigate-buffer')

class EventQueryService:
    """Service for querying events from the filesystem."""

    def __init__(self, storage_path: str, cache_ttl: int = 5):
        self.storage_path = storage_path
        self._cache = {}
        self._cache_ttl = cache_ttl

    def _get_cached(self, key: str) -> Optional[Any]:
        if key in self._cache:
            entry = self._cache[key]
            if time.monotonic() - entry['timestamp'] < self._cache_ttl:
                return entry['data']
        return None

    def _set_cache(self, key: str, data: Any):
        self._cache[key] = {
            'timestamp': time.monotonic(),
            'data': data
        }

    def _parse_summary(self, summary_text: str) -> Dict[str, str]:
        """Parse key-value pairs from summary.txt format."""
        parsed = {}
        for line in summary_text.split('\n'):
            if ':' in line:
                key, _, value = line.partition(':')
                parsed[key.strip()] = value.strip()
        return parsed

    def _extract_genai_entries(self, folder_path: str) -> List[Dict[str, Any]]:
        """Extract GenAI metadata entries from notification_timeline.json.
        Identical descriptions (same title, shortSummary, scene) are deduplicated
        so the AI Analysis section shows each unique analysis once; the raw timeline
        file is unchanged for debugging."""
        entries = []
        seen = set()
        timeline_path = os.path.join(folder_path, 'notification_timeline.json')
        if not os.path.exists(timeline_path):
            return entries
        try:
            with open(timeline_path, 'r') as f:
                data = json.load(f)
        except Exception:
            return entries
        for e in data.get('entries', []):
            payload = (e.get('data') or {}).get('payload') or {}
            if payload.get('type') != 'genai':
                continue
            after = payload.get('after') or {}
            meta = (after.get('data') or {}).get('metadata')
            if not meta:
                continue
            title = meta.get('title') or ''
            scene = meta.get('scene') or ''
            short_summary = meta.get('shortSummary') or meta.get('description') or ''
            # Skip boilerplate "no concerns/activity" entries to avoid noise
            lower = (title + ' ' + short_summary + ' ' + scene).lower()
            if 'no concerns' in lower or 'no activity' in lower:
                if not title and not scene and len(short_summary) < 80:
                    continue
            content_key = (title.strip(), short_summary.strip(), scene.strip())
            if content_key in seen:
                continue
            seen.add(content_key)
            entries.append({
                'title': title,
                'scene': scene,
                'shortSummary': short_summary,
                'time': meta.get('time'),
                'potential_threat_level': meta.get('potential_threat_level', 0),
            })
        return entries

    def _event_ended_in_timeline(self, folder_path: str) -> bool:
        """Check if event has ended based on timeline (Event end from Frigate, or end_time set)."""
        timeline_path = os.path.join(folder_path, 'notification_timeline.json')
        if not os.path.exists(timeline_path):
            return False
        try:
            with open(timeline_path, 'r') as f:
                data = json.load(f)
        except Exception:
            return False
        for e in data.get('entries', []):
            label = (e.get('label') or '').lower()
            if 'event end' in label:
                return True
            payload = (e.get('data') or {}).get('payload') or {}
            if payload.get('type') == 'end':
                return True
            after = payload.get('after') or {}
            if after.get('end_time') is not None:
                return True
        return False

    def _extract_cameras_zones_from_timeline(self, folder_path: str) -> List[Dict[str, Any]]:
        """Extract cameras and zones from frigate_mqtt entries in notification_timeline.json."""
        timeline_path = os.path.join(folder_path, 'notification_timeline.json')
        if not os.path.exists(timeline_path):
            return []
        try:
            with open(timeline_path, 'r') as f:
                data = json.load(f)
        except Exception:
            return []
        camera_zones = {}
        for e in data.get('entries', []):
            if e.get('source') != 'frigate_mqtt':
                continue
            payload = (e.get('data') or {}).get('payload') or {}
            after = payload.get('after') or payload.get('before') or {}
            camera = after.get('camera')
            if not camera:
                continue
            zones = camera_zones.setdefault(camera, set())
            for z in (after.get('entered_zones') or []):
                if z:
                    zones.add(z)
            for z in (after.get('current_zones') or []):
                if z:
                    zones.add(z)
        return [{"camera": cam, "zones": sorted(zones)} for cam, zones in sorted(camera_zones.items())]

    def _get_consolidated_events(self) -> List[Dict[str, Any]]:
        """Get consolidated events from events/{ce_id}/{camera}/ structure."""
        events_dir = os.path.join(self.storage_path, "events")
        events_list = []
        if not os.path.isdir(events_dir):
            return events_list

        for ce_id in sorted(os.listdir(events_dir), reverse=True):
            ce_path = os.path.join(events_dir, ce_id)
            if not os.path.isdir(ce_path):
                continue
            if '.' in ce_id and not ce_id.split('_')[0].isdigit():
                continue

            parts = ce_id.split('_', 1)
            ts = parts[0] if len(parts) > 0 else "0"
            summary_path = os.path.join(ce_path, 'summary.txt')
            metadata_path = os.path.join(ce_path, 'metadata.json')
            review_summary_path = os.path.join(ce_path, 'review_summary.md')
            viewed_path = os.path.join(ce_path, '.viewed')

            summary_text = "Analysis pending..."
            parsed = {}
            if os.path.exists(summary_path):
                with open(summary_path, 'r') as f:
                    summary_text = f.read().strip()
                parsed = self._parse_summary(summary_text)

            cameras = [d for d in os.listdir(ce_path)
                      if os.path.isdir(os.path.join(ce_path, d)) and not d.startswith('.')]

            metadata = {}
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                except Exception:
                    pass
            # Consolidated: metadata may live in primary camera folder
            if not metadata and cameras:
                for cam in cameras:
                    cam_meta = os.path.join(ce_path, cam, 'metadata.json')
                    if os.path.exists(cam_meta):
                        try:
                            with open(cam_meta, 'r') as f:
                                metadata = json.load(f)
                            break
                        except Exception:
                            pass

            genai_entries = self._extract_genai_entries(ce_path)

            review_summary = None
            if os.path.exists(review_summary_path):
                with open(review_summary_path, 'r') as f:
                    review_summary = f.read().strip()
            primary_clip = primary_snapshot = None
            for cam in cameras:
                cam_path = os.path.join(ce_path, cam)
                if os.path.exists(os.path.join(cam_path, 'clip.mp4')):
                    primary_clip = f"/files/events/{ce_id}/{cam}/clip.mp4"
                    break
            for cam in cameras:
                cam_path = os.path.join(ce_path, cam)
                if os.path.exists(os.path.join(cam_path, 'snapshot.jpg')):
                    primary_snapshot = f"/files/events/{ce_id}/{cam}/snapshot.jpg"
                    break

            has_clip = any(os.path.exists(os.path.join(ce_path, cam, 'clip.mp4')) for cam in cameras)
            has_snapshot = any(os.path.exists(os.path.join(ce_path, cam, 'snapshot.jpg')) for cam in cameras)

            # "Not ongoing": event ended in timeline, or has clip, or older than 90 min
            event_ended = self._event_ended_in_timeline(ce_path)
            try:
                ts_float = float(ts)
                age_seconds = time.time() - ts_float
                ongoing = not has_clip and age_seconds < 5400 and not event_ended  # 90 min
            except (ValueError, TypeError):
                ongoing = not has_clip and not event_ended

            cameras_with_zones = self._extract_cameras_zones_from_timeline(ce_path)
            if not cameras_with_zones and cameras:
                cameras_with_zones = [{"camera": c, "zones": []} for c in cameras]

            events_list.append({
                "event_id": ce_id,
                "camera": "events",
                "subdir": ce_id,
                "timestamp": ts,
                "summary": summary_text,
                "title": metadata.get("genai_title") or parsed.get("Title"),
                "description": metadata.get("genai_description") or parsed.get("Description") or parsed.get("AI Description"),
                "scene": metadata.get("genai_scene") or parsed.get("Scene"),
                "label": metadata.get("label") or parsed.get("Label", "unknown"),
                "severity": metadata.get("severity") or parsed.get("Severity"),
                "threat_level": metadata.get("threat_level", 0),
                "review_summary": review_summary,
                "has_clip": has_clip,
                "has_snapshot": has_snapshot,
                "viewed": os.path.exists(viewed_path),
                "hosted_clip": primary_clip or (f"/files/events/{ce_id}/{cameras[0]}/clip.mp4" if cameras else None),
                "hosted_snapshot": primary_snapshot or (f"/files/events/{ce_id}/{cameras[0]}/snapshot.jpg" if cameras else None),
                "cameras": cameras,
                "cameras_with_zones": cameras_with_zones,
                "consolidated": True,
                "ongoing": ongoing,
                "genai_entries": genai_entries
            })

        return events_list

    def _get_camera_events(self, camera_name: str) -> List[Dict[str, Any]]:
        """Helper to get events for a specific camera."""
        camera_path = os.path.join(self.storage_path, camera_name)
        events = []

        if not os.path.isdir(camera_path):
            return events

        subdirs = sorted(
            [d for d in os.listdir(camera_path)
             if os.path.isdir(os.path.join(camera_path, d))],
            reverse=True
        )

        for subdir in subdirs:
            folder_path = os.path.join(camera_path, subdir)
            summary_path = os.path.join(folder_path, 'summary.txt')

            parts = subdir.split('_', 1)
            ts = parts[0] if len(parts) > 0 else "0"
            eid = parts[1] if len(parts) > 1 else subdir

            summary_text = "Analysis pending..."
            parsed = {}
            if os.path.exists(summary_path):
                with open(summary_path, 'r') as f:
                    summary_text = f.read().strip()
                parsed = self._parse_summary(summary_text)

            metadata_path = os.path.join(folder_path, 'metadata.json')
            metadata = {}
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                except Exception:
                    pass

            review_summary_path = os.path.join(folder_path, 'review_summary.md')
            review_summary = None
            if os.path.exists(review_summary_path):
                with open(review_summary_path, 'r') as f:
                    review_summary = f.read().strip()

            clip_path = os.path.join(folder_path, 'clip.mp4')
            snapshot_path = os.path.join(folder_path, 'snapshot.jpg')
            viewed_path = os.path.join(folder_path, '.viewed')
            has_clip = os.path.exists(clip_path)
            event_ended = self._event_ended_in_timeline(folder_path)
            try:
                ts_float = float(ts)
                age_seconds = time.time() - ts_float
                ongoing = not has_clip and age_seconds < 5400 and not event_ended
            except (ValueError, TypeError):
                ongoing = not has_clip and not event_ended
            genai_entries = self._extract_genai_entries(folder_path)

            cameras_with_zones = self._extract_cameras_zones_from_timeline(folder_path)
            if not cameras_with_zones:
                cameras_with_zones = [{"camera": camera_name, "zones": []}]

            events.append({
                "event_id": eid,
                "camera": camera_name,
                "subdir": subdir,
                "timestamp": ts,
                "summary": summary_text,
                "title": metadata.get("genai_title") or parsed.get("Title"),
                "description": metadata.get("genai_description") or parsed.get("Description") or parsed.get("AI Description"),
                "scene": metadata.get("genai_scene") or parsed.get("Scene"),
                "label": metadata.get("label") or parsed.get("Label", "unknown"),
                "severity": metadata.get("severity") or parsed.get("Severity"),
                "threat_level": metadata.get("threat_level", 0),
                "review_summary": review_summary,
                "has_clip": has_clip,
                "has_snapshot": os.path.exists(snapshot_path),
                "viewed": os.path.exists(viewed_path),
                "hosted_clip": f"/files/{camera_name}/{subdir}/clip.mp4",
                "hosted_snapshot": f"/files/{camera_name}/{subdir}/snapshot.jpg",
                "cameras_with_zones": cameras_with_zones,
                "ongoing": ongoing,
                "genai_entries": genai_entries
            })

        return events

    def get_events(self, camera_name: str) -> List[Dict[str, Any]]:
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

    def get_all_events(self) -> Tuple[List[Dict[str, Any]], List[str]]:
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
                if camera_dir.split('_')[0].isdigit():
                    continue

                cameras_found.append(camera_dir)
                events = self.get_events(camera_dir)
                all_events.extend(events)

            all_events.sort(key=lambda x: x['timestamp'], reverse=True)

        except Exception as e:
            logger.error(f"Error listing events: {e}")
            raise

        result = (all_events, sorted(cameras_found))
        self._set_cache(cache_key, result)
        return result

    def get_cameras(self) -> List[str]:
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
                    if not item.split('_')[0].isdigit():
                        active_cameras.append(item)
        except Exception as e:
            logger.error(f"Error listing cameras: {e}")

        if os.path.isdir(os.path.join(self.storage_path, "events")) and "events" not in active_cameras:
            active_cameras.append("events")

        result = sorted(active_cameras)
        self._set_cache(cache_key, result)
        return result
