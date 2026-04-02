"""Pure transforms on summary text and merged timeline data (no filesystem I/O)."""

from __future__ import annotations

from typing import Any


def parse_summary(summary_text: str) -> dict[str, str]:
    """Parse key-value pairs from summary.txt format."""
    parsed: dict[str, str] = {}
    for line in summary_text.split("\n"):
        if ":" in line:
            key, _, value = line.partition(":")
            parsed[key.strip()] = value.strip()
    return parsed


def extract_genai_entries(timeline_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract GenAI metadata entries from notification_timeline.json.

    Identical descriptions (same title, shortSummary, scene) are deduplicated
    so the AI Analysis section shows each unique analysis once; the raw timeline
    file is unchanged for debugging.
    """
    entries: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
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


def event_ended_in_timeline(timeline_data: dict[str, Any]) -> bool:
    """True if timeline shows event end (label, end payload, or after.end_time)."""
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


def extract_end_timestamp_from_timeline(
    timeline_data: dict[str, Any] | None,
) -> float | None:
    """Return the latest end_time from timeline entries.

    Uses payload.after.end_time for Frigate, or data.end_time for test_ai_prompt
    entries. Maximum wins so consolidated events span the true interval.
    """
    end_times: list[float] = []
    for e in (timeline_data or {}).get("entries", []):
        payload = (e.get("data") or {}).get("payload") or {}
        after = payload.get("after") or {}
        end_time = after.get("end_time")
        if end_time is not None:
            try:
                end_times.append(float(end_time))
            except (TypeError, ValueError):
                pass
        if e.get("source") == "test_ai_prompt":
            end_time = (e.get("data") or {}).get("end_time")
            if end_time is not None:
                try:
                    end_times.append(float(end_time))
                except (TypeError, ValueError):
                    pass
    return max(end_times) if end_times else None


def extract_cameras_zones_from_timeline(
    timeline_data: dict[str, Any],
) -> list[dict[str, Any]]:
    """Cameras and zones from frigate_mqtt entries in the merged timeline."""
    data = timeline_data

    camera_zones: dict[str, set[str]] = {}
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
