"""File operations: folder creation, summaries, cleanup."""

import os
import re
import json
import time
import shutil
import logging
from datetime import datetime
from typing import List, Optional

from frigate_buffer.models import EventState

logger = logging.getLogger('frigate-buffer')


class FileManager:
    """Handles file operations: folder creation, summaries, cleanup."""

    def __init__(self, storage_path: str, retention_days: int):
        self.storage_path = storage_path
        self.retention_days = retention_days

        # Ensure storage directory exists
        os.makedirs(storage_path, exist_ok=True)
        logger.info(f"FileManager initialized: {storage_path}")
        logger.debug(f"Retention: {retention_days} days")

    def sanitize_camera_name(self, camera: str) -> str:
        """Sanitize camera name for filesystem use."""
        # Lowercase, replace spaces with underscores, remove special chars
        sanitized = camera.lower().replace(' ', '_')
        sanitized = re.sub(r'[^a-z0-9_]', '', sanitized)
        result = sanitized or 'unknown'

        # Verify path security
        final_path = os.path.realpath(os.path.join(self.storage_path, result))
        real_storage_path = os.path.realpath(self.storage_path)
        if os.path.commonpath([real_storage_path, final_path]) != real_storage_path:
            raise ValueError(f"Invalid camera name: {camera}")

        return result

    def create_event_folder(self, event_id: str, camera: str, timestamp: float) -> str:
        """Create folder for event: {camera}/{timestamp}_{event_id} (legacy)"""
        sanitized_camera = self.sanitize_camera_name(camera)
        folder_name = f"{int(timestamp)}_{event_id}"
        camera_path = os.path.join(self.storage_path, sanitized_camera)
        folder_path = os.path.join(camera_path, folder_name)

        # Verify path security
        real_storage_path = os.path.realpath(self.storage_path)
        real_folder_path = os.path.realpath(folder_path)
        if os.path.commonpath([real_storage_path, real_folder_path]) != real_storage_path:
            raise ValueError(f"Invalid event path: {folder_path}")

        os.makedirs(folder_path, exist_ok=True)
        logger.info(f"Created folder: {sanitized_camera}/{folder_name}")
        return folder_path

    def create_consolidated_event_folder(self, folder_name: str) -> str:
        """Create folder for consolidated event: events/{folder_name}"""
        events_dir = os.path.join(self.storage_path, "events")
        folder_path = os.path.join(events_dir, folder_name)

        # Verify path security
        real_storage_path = os.path.realpath(self.storage_path)
        real_folder_path = os.path.realpath(folder_path)
        if os.path.commonpath([real_storage_path, real_folder_path]) != real_storage_path:
            raise ValueError(f"Invalid consolidated event path: {folder_path}")

        os.makedirs(folder_path, exist_ok=True)
        logger.info(f"Created consolidated folder: events/{folder_name}")
        return folder_path

    def ensure_consolidated_camera_folder(self, ce_folder_path: str, camera: str) -> str:
        """Ensure events/{ce_id}/{camera}/ exists. Returns the camera folder path."""
        sanitized = self.sanitize_camera_name(camera)
        camera_path = os.path.join(ce_folder_path, sanitized)
        os.makedirs(camera_path, exist_ok=True)
        return camera_path

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
            # Get list of cameras safely
            try:
                camera_dirs = os.listdir(self.storage_path)
            except OSError:
                camera_dirs = []

            for camera_dir in camera_dirs:
                camera_path = os.path.join(self.storage_path, camera_dir)

                if not os.path.isdir(camera_path):
                    continue
                if camera_dir.split('_')[0].isdigit():
                    continue

                cam_clips = cam_snapshots = cam_descriptions = 0

                # Get list of events safely
                try:
                    event_dirs = os.listdir(camera_path)
                except OSError:
                    continue

                for event_dir in event_dirs:
                    event_path = os.path.join(camera_path, event_dir)
                    if not os.path.isdir(event_path):
                        continue

                    clip_path = os.path.join(event_path, 'clip.mp4')
                    snapshot_path = os.path.join(event_path, 'snapshot.jpg')

                    for f in ('summary.txt', 'review_summary.md', 'metadata.json'):
                        p = os.path.join(event_path, f)
                        try:
                            if os.path.exists(p):
                                cam_descriptions += os.path.getsize(p)
                        except OSError:
                            pass

                    try:
                        if os.path.exists(clip_path):
                            cam_clips += os.path.getsize(clip_path)
                    except OSError:
                        pass

                    try:
                        if os.path.exists(snapshot_path):
                            cam_snapshots += os.path.getsize(snapshot_path)
                    except OSError:
                        pass

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
