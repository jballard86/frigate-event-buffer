"""Filesystem reads for event folders: clips, timeline merge, full folder parse."""

from __future__ import annotations

import json
import os
from typing import Any


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
    data: dict[str, Any] = {"event_id": None, "entries": []}
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


def parse_event_folder(folder_path: str) -> dict[str, Any]:
    """Read all event files under folder_path and return the aggregated data dict."""
    data: dict[str, Any] = {
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

    subdirs_map: dict[str, dict[str, Any]] = {}
    try:
        with os.scandir(folder_path) as it:
            for entry in it:
                if entry.is_dir() and not entry.name.startswith("."):
                    sub_clip_basename = resolve_clip_in_folder(entry.path)
                    sub_snap = os.path.join(entry.path, "snapshot.jpg")
                    sub_meta = os.path.join(entry.path, "metadata.json")

                    sub_data: dict[str, Any] = {
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
