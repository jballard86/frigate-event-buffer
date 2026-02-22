"""File operations: folder creation, summaries, cleanup."""

import os
import re
import json
import time
import shutil
import zipfile
import logging
from datetime import datetime
from typing import Any, Union

from frigate_buffer.models import EventState

logger = logging.getLogger('frigate-buffer')

# Directories under storage root that are not cameras; skip in cleanup and storage stats.
_NON_CAMERA_DIRS = frozenset({"ultralytics", "yolo_models", "daily_reports", "daily_reviews"})

# Optional imports for AI frame stitched write (color + B/W)
try:
    import cv2
    import numpy as np
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False


def _is_tensor(x: Any) -> bool:
    """True if x is a torch.Tensor."""
    return type(x).__name__ == "Tensor"


def write_stitched_frame(frame_bgr: Union[Any, np.ndarray], filepath: str) -> bool:
    """
    Write a single color image to filepath.

    Accepts numpy ndarray HWC BGR (from overlay or legacy path) or torch.Tensor BCHW/CHW RGB.
    For tensor: uses torchvision.io.encode_jpeg and writes bytes. For numpy: uses cv2.imwrite.
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
    except OSError as e:
        logger.warning("Failed to create directory for %s: %s", filepath, e)
        return False

    if _is_tensor(frame_bgr):
        try:
            import torch
            from torchvision.io import encode_jpeg
        except ImportError as e:
            logger.warning("torch/torchvision not available for tensor frame write: %s", e)
            return False
        t = frame_bgr
        if t.dim() == 4:
            t = t.squeeze(0)
        if t.dim() != 3 or t.shape[0] not in (1, 3):
            logger.warning("write_stitched_frame tensor expected CHW with 1 or 3 channels, got shape %s", t.shape)
            return False
        if t.dtype == torch.float32 or t.dtype == torch.float64:
            t = (t.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
        elif t.dtype != torch.uint8:
            t = t.to(torch.uint8)
        # encode_jpeg expects CHW uint8 (RGB)
        try:
            jpeg_bytes = encode_jpeg(t, quality=90)
        except Exception as e:
            logger.warning("encode_jpeg failed for %s: %s", filepath, e)
            return False
        buf = jpeg_bytes.cpu().numpy().tobytes()
        try:
            with open(filepath, "wb") as f:
                f.write(buf)
            return True
        except OSError as e:
            logger.warning("Failed to write frame %s: %s", filepath, e)
            return False

    if not _CV2_AVAILABLE:
        logger.warning("cv2/numpy not available, cannot write frame")
        return False
    try:
        return cv2.imwrite(filepath, frame_bgr)
    except Exception as e:
        logger.warning("Failed to write frame %s: %s", filepath, e)
        return False


def _write_ai_analysis_internal(
    event_dir: str,
    frames_data: list[Any],
    camera_override: str | None = None,
    write_manifest: bool = True,
    create_zip_flag: bool = True,
    save_frames: bool = True,
    include_camera_in_filename: bool = True,
) -> None:
    """Internal helper to write AI frame analysis data."""
    if not save_frames or not frames_data:
        return
    if not _CV2_AVAILABLE:
        logger.warning("cv2/numpy not available, skipping AI frame analysis write")
        return

    frames_dir = os.path.join(event_dir, "ai_frame_analysis", "frames")
    os.makedirs(frames_dir, exist_ok=True)
    total = len(frames_data)
    manifest_entries = []

    for seq, item in enumerate(frames_data, start=1):
        # Extract fields from item (tuple/list or object with attributes)
        is_tuple = isinstance(item, (tuple, list))
        frame = getattr(item, "frame", item[0] if is_tuple and len(item) >= 1 else None)
        frame_time_sec = getattr(item, "timestamp_sec", item[1] if is_tuple and len(item) >= 2 else 0.0)
        camera = camera_override or getattr(item, "camera", item[2] if is_tuple and len(item) >= 3 else "")
        meta = getattr(item, "metadata", item[3] if is_tuple and len(item) >= 4 else {}) or {}

        if frame is None:
            continue

        camera_str = str(camera)
        if include_camera_in_filename and camera_str:
            fname = f"frame_{seq:03d}_{camera_str.replace(' ', '_')}.jpg"
        else:
            fname = f"frame_{seq:03d}.jpg"

        out_path = os.path.join(frames_dir, fname)
        if write_stitched_frame(frame, out_path):
            m = {
                "filename": fname,
                "timestamp_sec": frame_time_sec,
                "camera": camera,
                "seq": seq,
                "total": total,
            }
            if meta.get("is_full_frame_resize") is True:
                m["is_full_frame_resize"] = True
            manifest_entries.append(m)

    if write_manifest and manifest_entries:
        manifest_path = os.path.join(event_dir, "ai_frame_analysis", "manifest.json")
        try:
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest_entries, f, indent=2)
        except OSError as e:
            logger.warning("Failed to write ai_frame_analysis manifest: %s", e)

    if create_zip_flag:
        create_ai_analysis_zip(event_dir)


def write_ai_frame_analysis_multi_cam(
    event_dir: str,
    frames_with_time_and_camera: list[Any],
    write_manifest: bool = True,
    create_zip_flag: bool = True,
    save_frames: bool = True,
) -> None:
    """Write ai_frame_analysis/frames at CE root with multi-cam manifest.
    Accepts list of ExtractedFrame (or objects with .frame, .timestamp_sec, .camera, .metadata)."""
    _write_ai_analysis_internal(
        event_dir,
        frames_with_time_and_camera,
        write_manifest=write_manifest,
        create_zip_flag=create_zip_flag,
        save_frames=save_frames,
        include_camera_in_filename=True
    )


def create_ai_analysis_zip(event_dir: str) -> None:
    """Create event_dir/ai_analysis_debug.zip containing ai_frame_analysis/ and analysis_result.json."""
    zip_path = os.path.join(event_dir, "ai_analysis_debug.zip")
    analysis_root = os.path.join(event_dir, "analysis_result.json")
    ai_dir = os.path.join(event_dir, "ai_frame_analysis")
    if not os.path.isdir(ai_dir):
        logger.debug("No ai_frame_analysis dir, skipping zip")
        return
    try:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _dirs, files in os.walk(ai_dir):
                for f in files:
                    abspath = os.path.join(root, f)
                    arcname = os.path.relpath(abspath, event_dir)
                    zf.write(abspath, arcname)
            if os.path.isfile(analysis_root):
                zf.write(analysis_root, "analysis_result.json")
        logger.debug("Created %s", zip_path)
    except OSError as e:
        logger.warning("Failed to create ai_analysis_debug.zip: %s", e)


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
            logger.exception(f"Failed to write summary: {e}")
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
        """Append an entry to the event folder timeline (append-only JSONL for efficiency)."""
        entry = dict(entry)
        entry["ts"] = entry.get("ts") or datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        append_path = os.path.join(folder_path, "notification_timeline_append.jsonl")
        try:
            with open(append_path, 'a') as f:
                f.write(json.dumps(entry) + '\n')
        except Exception as e:
            logger.debug(f"Failed to append timeline entry: {e}")

    def write_ce_summary(self, folder_path: str, ce_id: str, title: str, description: str,
                         scene: str = "", threat_level: int = 0, label: str = "unknown",
                         start_time: float = 0) -> bool:
        """Write summary.txt for a consolidated event at CE root."""
        try:
            summary_path = os.path.join(folder_path, "summary.txt")
            timestamp_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)) if start_time else ""
            lines = [
                f"Event ID: {ce_id}",
                "Camera: events (consolidated)",
                f"Label: {label}",
                f"Timestamp: {timestamp_str}",
                "Phase: finalized",
                "",
            ]
            if title:
                lines.append(f"Title: {title}")
            if description:
                lines.append(f"Description: {description}")
            if scene:
                lines.append(f"Scene: {scene}")
            if threat_level > 0:
                level_names = {0: "Normal", 1: "Suspicious", 2: "Critical"}
                lines.append(f"Threat Level: {threat_level} ({level_names.get(threat_level, 'Unknown')})")
            lines.append("")
            lines.append("Review Summary: See review_summary.md")
            with open(summary_path, 'w') as f:
                f.write('\n'.join(lines))
            logger.debug("Written CE summary for %s", ce_id)
            return True
        except Exception as e:
            logger.exception("Failed to write CE summary: %s", e)
            return False

    def write_ce_metadata_json(self, folder_path: str, ce_id: str, title: str, description: str,
                               scene: str = "", threat_level: int = 0, label: str = "unknown",
                               camera: str = "events", start_time: float = 0, end_time: float | None = None) -> bool:
        """Write metadata.json for a consolidated event at CE root."""
        try:
            meta_path = os.path.join(folder_path, "metadata.json")
            metadata = {
                "event_id": ce_id,
                "camera": camera,
                "label": label,
                "created_at": start_time,
                "end_time": end_time,
                "threat_level": threat_level,
                "severity": "detection",
                "genai_title": title,
                "genai_description": description,
                "genai_scene": scene,
                "phase": "finalized",
            }
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            return True
        except Exception as e:
            logger.error("Failed to write CE metadata: %s", e)
            return False

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

    def delete_event_folder(self, folder_path: str) -> bool:
        """Delete a single event folder (or CE camera subfolder). Path must be under storage_path.
        Returns True if the folder was deleted, False if path invalid or not found."""
        try:
            base = os.path.realpath(self.storage_path)
            target = os.path.realpath(folder_path)
            if not target.startswith(base) or target == base:
                logger.warning(f"Refusing to delete path outside storage: {folder_path}")
                return False
            if not os.path.isdir(target):
                return False
            shutil.rmtree(target)
            logger.info(f"Deleted event folder: {folder_path}")
            return True
        except OSError as e:
            logger.warning(f"Failed to delete event folder {folder_path}: {e}")
            return False

    def rename_event_folder(self, folder_path: str, suffix: str = "-canceled") -> str:
        """Rename an event folder by appending suffix to its basename. Path must be under storage_path.
        Returns the new path. Raises ValueError if path is outside storage."""
        real_storage = os.path.realpath(self.storage_path)
        real_folder = os.path.realpath(folder_path)
        if os.path.commonpath([real_storage, real_folder]) != real_storage or real_folder == real_storage:
            raise ValueError(f"Invalid event path for rename: {folder_path}")
        if not os.path.isdir(real_folder):
            raise ValueError(f"Not a directory: {folder_path}")
        parent = os.path.dirname(real_folder)
        basename = os.path.basename(real_folder)
        if basename.endswith(suffix):
            return folder_path
        new_basename = basename + suffix
        new_path = os.path.join(parent, new_basename)
        if os.path.exists(new_path):
            raise ValueError(f"Target already exists: {new_path}")
        os.rename(real_folder, new_path)
        logger.info(f"Renamed event folder to {new_basename}")
        return new_path

    def write_canceled_summary(self, folder_path: str, reason: str = "Event exceeded max_event_length_seconds; AI analysis skipped.") -> bool:
        """Write summary.txt for a canceled event so the event view shows the cancel title and reason."""
        try:
            summary_path = os.path.join(folder_path, "summary.txt")
            lines = [
                "Title: Canceled event: max event length exceeded",
                f"Description: {reason}",
            ]
            with open(summary_path, 'w') as f:
                f.write('\n'.join(lines))
            logger.debug(f"Written canceled summary to {folder_path}")
            return True
        except Exception as e:
            logger.exception(f"Failed to write canceled summary to {folder_path}: {e}")
            return False

    def cleanup_old_events(self, active_event_ids: list[str],
                          active_ce_folder_names: list[str] | None = None) -> int:
        """Delete folders older than retention period. Returns count deleted.
        active_ce_folder_names: folder names of active consolidated events (e.g. 1771003190_abc) to skip."""
        now = time.time()
        cutoff = now - (self.retention_days * 86400)
        deleted_count = 0
        active_ce = set(active_ce_folder_names or [])

        logger.debug(f"Running cleanup: cutoff={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(cutoff))}")

        try:
            # Iterate through camera subdirectories and events/
            with os.scandir(self.storage_path) as it:
                for camera_entry in it:
                    if not camera_entry.is_dir():
                        continue

                    camera_dir = camera_entry.name
                    camera_path = camera_entry.path

                    # Check if this is a camera folder (contains event folders)
                    # or a legacy flat event folder (for migration period)
                    first_item = camera_dir.split('_', 1)
                    if len(first_item) > 1 and first_item[0].isdigit():
                        # Legacy flat structure: {timestamp}_{event_id}
                        try:
                            ts = float(first_item[0])
                            event_id = first_item[1]
                            base_id = (event_id[:-len('-canceled')] if (event_id and event_id.endswith('-canceled')) else event_id) if event_id else None

                            if base_id and base_id in active_event_ids:
                                continue

                            if ts < cutoff:
                                shutil.rmtree(camera_path)
                                logger.info(f"Cleaned up legacy event: {camera_dir}")
                                deleted_count += 1
                        except (ValueError, IndexError):
                            pass
                        continue

                    if camera_dir in _NON_CAMERA_DIRS:
                        continue

                    # New structure: iterate through event folders in camera dir
                    with os.scandir(camera_path) as it_events:
                        for event_entry in it_events:
                            if not event_entry.is_dir():
                                continue

                            event_dir = event_entry.name
                            event_path = event_entry.path

                            # Test run folders (test1, test2, ...): age-only by mtime
                            if camera_dir == "events" and re.match(r"^test\d+$", event_dir):
                                try:
                                    if os.path.getmtime(event_path) < cutoff:
                                        shutil.rmtree(event_path)
                                        logger.info(f"Cleaned up old test run: {camera_dir}/{event_dir}")
                                        deleted_count += 1
                                except OSError:
                                    pass
                                continue

                            try:
                                parts = event_dir.split('_', 1)
                                ts = float(parts[0])
                                event_id = parts[1] if len(parts) > 1 else None
                                # For -canceled folders, match by base id so we protect until event is removed; folder remains deletable when past retention
                                base_id = (event_id[:-len('-canceled')] if (event_id and event_id.endswith('-canceled')) else event_id) if event_id else None

                                # Skip if event is still active (legacy) or CE is active (events/ folder)
                                if base_id and base_id in active_event_ids:
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

    def _sum_event_folder(self, event_path: str) -> tuple:
        """Sum clip, snapshot, and description sizes in an event folder. Returns (clips, snapshots, descriptions) in bytes."""
        from frigate_buffer.services.query import resolve_clip_in_folder
        clip_basename = resolve_clip_in_folder(event_path)
        clip_path = os.path.join(event_path, clip_basename) if clip_basename else None
        snapshot_path = os.path.join(event_path, 'snapshot.jpg')
        desc_files = (
            'summary.txt', 'review_summary.md', 'metadata.json',
            'notification_timeline.json', 'analysis_result.json'
        )
        cam_clips = cam_snapshots = cam_descriptions = 0
        try:
            if clip_path and os.path.exists(clip_path):
                cam_clips = os.path.getsize(clip_path)
        except OSError:
            pass
        try:
            if os.path.exists(snapshot_path):
                cam_snapshots = os.path.getsize(snapshot_path)
        except OSError:
            pass
        for f in desc_files:
            p = os.path.join(event_path, f)
            try:
                if os.path.exists(p):
                    cam_descriptions += os.path.getsize(p)
            except OSError:
                pass
        return (cam_clips, cam_snapshots, cam_descriptions)

    def _dir_total_bytes(self, path: str) -> int:
        """Return total size in bytes of all files under path. Returns 0 if path does not exist or is not a dir."""
        if not os.path.isdir(path):
            return 0
        total = 0
        try:
            for root, _dirs, files in os.walk(path):
                for f in files:
                    try:
                        total += os.path.getsize(os.path.join(root, f))
                    except OSError:
                        pass
        except OSError:
            pass
        return total

    def compute_storage_stats(self) -> dict:
        """Compute storage usage by camera and type. Returns bytes.
        Covers: legacy camera/event folders, consolidated events/ce_id/camera/ and CE-root,
        and daily_reports/ and daily_reviews/ under storage_path.
        """
        clips = 0
        snapshots = 0
        descriptions = 0
        by_camera = {}

        try:
            camera_entries = []
            try:
                with os.scandir(self.storage_path) as it:
                    camera_entries = list(it)
            except OSError:
                pass

            for camera_entry in camera_entries:
                if not camera_entry.is_dir():
                    continue

                camera_dir = camera_entry.name
                camera_path = camera_entry.path

                if camera_dir.split('_')[0].isdigit():
                    continue
                if camera_dir in _NON_CAMERA_DIRS:
                    continue

                cam_clips = cam_snapshots = cam_descriptions = 0

                if camera_dir == "events":
                    # Consolidated layout: events/{ce_id}/ (CE root files) and events/{ce_id}/{camera}/ (event files)
                    try:
                        with os.scandir(camera_path) as it_ce:
                            for ce_entry in it_ce:
                                if not ce_entry.is_dir():
                                    continue
                                ce_path = ce_entry.path
                                # CE-root files (notification.gif, review_summary.md)
                                for f in ('notification.gif', 'review_summary.md'):
                                    p = os.path.join(ce_path, f)
                                    try:
                                        if os.path.exists(p) and os.path.isfile(p):
                                            cam_descriptions += os.path.getsize(p)
                                    except OSError:
                                        pass
                                # Camera subdirs: events/ce_id/camera/ with *.mp4, snapshot.jpg, etc.
                                try:
                                    with os.scandir(ce_path) as it_cam:
                                        for cam_entry in it_cam:
                                            if not cam_entry.is_dir() or cam_entry.name.startswith('.'):
                                                continue
                                            c, s, d = self._sum_event_folder(cam_entry.path)
                                            cam_clips += c
                                            cam_snapshots += s
                                            cam_descriptions += d
                                except OSError:
                                    pass
                    except OSError:
                        pass
                else:
                    # Legacy layout: camera/event_id/ with clip, snapshot, descriptions
                    try:
                        with os.scandir(camera_path) as it_events:
                            for event_entry in it_events:
                                if not event_entry.is_dir():
                                    continue
                                c, s, d = self._sum_event_folder(event_entry.path)
                                cam_clips += c
                                cam_snapshots += s
                                cam_descriptions += d
                    except OSError:
                        continue

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

            # Include daily_reports and daily_reviews in total and descriptions
            for subdir in ('daily_reports', 'daily_reviews'):
                path = os.path.join(self.storage_path, subdir)
                size = self._dir_total_bytes(path)
                if size > 0:
                    descriptions += size
            total = clips + snapshots + descriptions

        except Exception as e:
            logger.error(f"Error computing storage stats: {e}")
            total = clips + snapshots + descriptions

        return {
            'clips': clips,
            'snapshots': snapshots,
            'descriptions': descriptions,
            'total': total,
            'by_camera': by_camera
        }
