"""Voluptuous fragment for YAML ``cameras`` (required top-level list)."""

from __future__ import annotations

from voluptuous import Optional, Required

# Single camera entry; wrapped in a list for Required("cameras"): [...].
CAMERA_ENTRY_SCHEMA = {
    # Frigate camera name; must match the camera key in Frigate config.
    Required("name"): str,
    # If set, only events with these object labels (e.g. person, car) are
    # processed; empty or omit = allow all.
    Optional("labels"): [str],
    # Smart zone filter: event creation gated by zone and exceptions
    # (omit for legacy: all events start immediately).
    Optional("event_filters"): {
        # Only create an event when the object enters one of these zones.
        Optional("tracked_zones"): [str],
        # Labels or sub_labels that start an event regardless of zone.
        Optional("exceptions"): [str],
    },
}

CAMERAS_LIST_SCHEMA = [CAMERA_ENTRY_SCHEMA]
