# Frigate Event Buffer

A state-aware orchestrator that listens to Frigate NVR events via MQTT, tracks them through their lifecycle, sends Ring-style sequential notifications to Home Assistant, and maintains a configurable rolling evidence locker (default 3 days). Includes a built-in event viewer and daily review summaries. Compatible with **Frigate 0.17+** with full GenAI review integration.

## Features

- **Frigate 0.17 Compatible**: Supports Frigate 0.17's new MQTT payload structure including `type: "genai"` review messages and `data.metadata` fields
- **MQTT Event Tracking**: Subscribes to Frigate's MQTT topics to detect and track events in real-time
- **Four-Phase Lifecycle**: Tracks events through NEW → DESCRIBED → FINALIZED → SUMMARIZED phases
- **Ring-Style Notifications**: Sends progressive updates to Home Assistant with phase-specific messages as event details emerge
- **GenAI Integration**: Captures Frigate's GenAI titles, descriptions, and threat levels (via `data.metadata` in reviews and `description` type in tracked object updates)
- **Review Summaries**: Fetches rich markdown security reports from Frigate's review summary API with cross-camera context, timeline, and assessments
- **Threat Level Alerts**: Three-tier threat classification (0=Normal, 1=Suspicious, 2=Critical) — Level 2 alerts bypass phone volume/DND and keep all follow-up notifications audible
- **Camera/Label Filtering**: Only process events from specific cameras or with specific labels
- **Smart Zone Filtering**: Optional per-camera tracked zones (create event only when object enters); exceptions (e.g., UPS, FedEx) trigger regardless of zone; Late Start when creation is triggered by a later update
- **Multi-Camera Support**: Handles events from multiple cameras simultaneously without state collision
- **Clip Export via Frigate API**: Requests clips from Frigate's Export API (POST with JSON body `playback`/`name` per Frigate schema; polls by `export_id` when async); full event duration with configurable buffer. For **consolidated events**, exports use per-camera time ranges and a representative event ID per camera (from sub-events) to avoid 404s and incorrect footage. Falls back to per-event clip via events API if export fails or times out; export failures log full raw response at WARNING for debugging; timeline includes full Frigate response; "Event ongoing" badge clears when clip is available, when Frigate signals event end (from timeline), or after 90 minutes
- **Auto-Transcoding**: Transcodes clips to H.264 for broad compatibility
- **Clip Download Retry**: Retries clip downloads up to 3 times on HTTP 400 (Frigate not ready), with 5-second delays when using events API fallback. HTTP 404 (no recording available) is not retried—logged once and returns immediately
- **FFmpeg Safety**: 60-second timeout with graceful termination prevents zombie processes
- **Rolling Retention**: Automatically cleans up events older than the retention period (default: 3 days)
- **Notification Rate Limiting**: Max 2 notifications per 5 seconds with queue overflow protection
- **Built-in Event Viewer**: Self-contained web page at `/player` with video playback, expandable AI analysis (each GenAI event from timeline with its own expand/collapse; cross-camera review; single-camera shows "Review Summary"), "Event ongoing" badge clears when clip available, when Frigate signals event end (timeline), or after 90 min, event navigation, reviewed/unreviewed filtering, download, and **View Timeline** (per-event data pipeline log with all files including clips/snapshots from camera subdirs) — embeddable as an HA iframe
- **Stats Dashboard**: Stats as a header button (like Daily Review) linking to `/stats-page`; standalone stats page with event counts (today/week/month), API Usage (Month to Date API cost and token usage from HA helpers when configured), storage by camera, recent errors, last cleanup, system info; configurable auto-refresh with manual Refresh button
- **Daily Review**: Frigate review summarize integration — scheduled fetch at 1am for previous day, 90-day retention (configurable), date selector, "Current Day Review" for midnight-to-now; markdown rendering
- **Event Review Tracking**: Mark events as reviewed with per-event or bulk "mark all" controls; defaults to showing unreviewed events
- **REST API**: Serves events, clips, and snapshots to your Home Assistant dashboard
- **Debug Logging**: Configurable log levels for troubleshooting

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│              State-Aware Orchestrator                    │
├─────────────────────────────────────────────────────────┤
│  MQTT Client          EventStateManager    Flask API    │
│  ├─ frigate/events    ├─ active_events     ├─ /player   │
│  │   (new,update,end) ├─ smart zone filter  ├─ /events   │
│  ├─ frigate/+/        ├─ phase tracking    ├─ /cameras  │
│  │   tracked_object   │   (NEW→DESCRIBED   ├─ /files    │
│  └─ frigate/reviews   │    →FINALIZED      ├─ /stats    │
│                       │    →SUMMARIZED)    ├─ /stats-   │
│                       │                    │   page     │
│                       │                    ├─ /daily-   │
│                       │                    │   review   │
│                       │                    └─ /status   │
└─────────────────────────────────────────────────────────┘
```

## Project Structure

The application is organized as a Python package `frigate_buffer/`:

| File / Directory | Description |
|------------------|-------------|
| `main.py` | Entry point. Loads config, sets up logging and signal handlers, starts the orchestrator. Run with `python -m frigate_buffer.main`. |
| `config.py` | Configuration loading. Merges YAML, env vars, and defaults. Searches config paths in order. |
| `logging_utils.py` | Error buffer and logging. `ErrorBuffer` stores recent errors for the stats dashboard; `setup_logging()` configures log level and handlers. |
| `models.py` | Data models: `EventPhase`, `EventState`, `ConsolidatedEvent`, plus helpers for consolidated IDs and "no concerns" detection. |
| `orchestrator.py` | `StateAwareOrchestrator` — MQTT subscription, event phase tracking, notification publishing, and coordination of managers. |
| `managers/` | Business logic modules: |
| `managers/file.py` | `FileManager` — clip/snapshot download, FFmpeg transcode, storage paths, cleanup. Export API uses JSON body (`playback`, `name`), polls by export_id, 90s poll timeout, 180s download timeout; returns rich result with Frigate response for timeline debugging. Export failures and `success: false` responses are logged at WARNING with full raw response. HTTP 404 on clip download is treated as "no recording available" and returns False without retries; retries remain for HTTP 400 (Not Ready) and other transient errors. |
| `managers/state.py` | `EventStateManager` — per-event state (phase, metadata) and active event tracking. |
| `managers/consolidation.py` | `ConsolidatedEventManager` — groups related Frigate events into consolidated events. |
| `managers/reviews.py` | `DailyReviewManager` — fetches and caches Frigate daily review summaries. |
| `services/notifier.py` | `NotificationPublisher` — publishes MQTT notifications to Home Assistant. |
| `web/server.py` | Flask app factory `create_app(orchestrator)`. Routes for player, events, files, stats, daily review, API. |
| `templates/` | Jinja2 templates (player, stats, daily review, timeline). Single location under `frigate_buffer/`; used by Flask at runtime. |
| `static/` | Static assets (marked.min.js, purify.min.js). Located under `frigate_buffer/`. |
| `Dockerfile` | Builds from `frigate_buffer/` and runs `python -m frigate_buffer.main`. |
| `docker-compose.example.yaml` | Template for Docker Compose — local build or token pull from private GitHub. |
| `config.example.yaml` | Example configuration for cameras, event_filters (Smart Zone Filtering), settings, network, optional HA integration. |

## Quick Start

### 1. Build the Docker Image

From the project root (cloned repo):

```bash
docker build -t frigate-buffer .
```

### 2. Run the Container

```bash
docker run -d \
  --name frigate-buffer \
  -p 5055:5055 \
  -v /mnt/user/appdata/frigate_buffer:/app/storage \
  frigate-buffer
```

### 3. Verify It's Running

```bash
curl http://localhost:5055/status
```

### Running without Docker

From the project root with dependencies installed:

```bash
python -m frigate_buffer.main
```

The app searches for `config.yaml` in order at: `/app/config.yaml`, `/app/storage/config.yaml`, `./config.yaml`, and `config.yaml`. No separate path configuration is needed.

## Configuration

Configuration is loaded from three sources (in order of priority):

1. **Environment variables** (highest priority)
2. **config.yaml** file (searched at `/app/config.yaml`, `/app/storage/config.yaml`, `./config.yaml`, and `config.yaml` in the current working directory)
3. **Default values** (lowest priority)

Place your `config.yaml` inside the storage volume (e.g. `/mnt/user/appdata/frigate_buffer/config.yaml`); the app finds it at `/app/storage/config.yaml` automatically. No separate bind mount is required. You can optionally bind-mount a config file to `/app/config.yaml` instead.

### config.yaml

Copy the example config and customize it:

```bash
cp config.example.yaml config.yaml
# Edit config.yaml with your values
```

The config file structure:

```yaml
# Camera configuration with per-camera label filtering
# Only events from listed cameras will be processed
# Omit a camera entirely to filter it out
cameras:
  # Doorbell - only person and package events
  - name: "Doorbell"
    labels:
      - "person"
      - "package"

  # Driveway - only vehicle events (with Smart Zone Filtering example)
  - name: "Front_Yard"
    labels:
      - "car"
      - "truck"
    # event_filters:     # Optional - omit for legacy behavior
    #   tracked_zones:   # Only create when object enters these zones
    #     - driveway
    #   exceptions:      # Create regardless of zone
    #     - "UPS"
    #     - "FedEx"

  # Backyard - allow ALL labels (empty list)
  - name: "Carport"
    labels: []

# Application settings
settings:
  retention_days: 3
  cleanup_interval_hours: 1
  ffmpeg_timeout_seconds: 60
  notification_delay_seconds: 2  # Delay before fetching snapshot after initial notification
  log_level: "DEBUG"  # DEBUG, INFO, WARNING, ERROR
  summary_padding_before: 15   # Seconds before event start for review summary
  summary_padding_after: 15    # Seconds after event end for review summary
  stats_refresh_seconds: 60    # Stats panel auto-refresh interval (seconds)
  daily_review_retention_days: 90   # How long to keep saved daily reviews (days)
  daily_review_schedule_hour: 1     # Hour (0-23) to fetch previous day's review
  event_gap_seconds: 120      # Seconds of no activity before new consolidated event
  export_buffer_before: 5     # Seconds before event start for clip export time range
  export_buffer_after: 30    # Seconds after event end for clip export time range

# Network configuration (REQUIRED - no defaults)
network:
  mqtt_broker: "YOUR_MQTT_BROKER_IP"
  mqtt_port: 1883
  frigate_url: "http://YOUR_FRIGATE_IP:5000"
  buffer_ip: "YOUR_BUFFER_IP"  # IP where this container is reachable
  flask_port: 5055
  storage_path: "/app/storage"

# Optional: Home Assistant REST API (for stats page API Usage display)
# ha:
#   base_url: "http://YOUR_HA_IP:8123/api"
#   token: "YOUR_LONG_LIVED_ACCESS_TOKEN"
#   gemini_cost_entity: "input_number.gemini_daily_cost"
#   gemini_tokens_entity: "input_number.gemini_total_tokens"
```

### Environment Variables

Environment variables override config.yaml values:

| Variable | Default | Description |
|----------|---------|-------------|
| `MQTT_BROKER` | *(required)* | MQTT broker IP address |
| `MQTT_PORT` | `1883` | MQTT broker port |
| `BUFFER_IP` | *(required)* | Buffer container's reachable IP (used in notification image/video URLs) |
| `HA_IP` | *(optional)* | Fallback for BUFFER_IP if set (for compatibility) |
| `FRIGATE_URL` | *(required)* | Frigate API base URL |
| `STORAGE_PATH` | `/app/storage` | Storage directory inside container |
| `RETENTION_DAYS` | `3` | Days to retain event folders |
| `FLASK_PORT` | `5055` | Flask server port |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `STATS_REFRESH_SECONDS` | `60` | Stats panel auto-refresh interval (seconds) |
| `DAILY_REVIEW_RETENTION_DAYS` | `90` | Days to retain saved daily reviews |
| `DAILY_REVIEW_SCHEDULE_HOUR` | `1` | Hour (0-23) to run daily review fetch for previous day |
| `EVENT_GAP_SECONDS` | `120` | Seconds without activity before new consolidated event |
| `EXPORT_BUFFER_BEFORE` | `5` | Seconds before event start for clip export time range |
| `EXPORT_BUFFER_AFTER` | `30` | Seconds after event end for clip export time range |
| `HA_URL` | *(optional)* | Home Assistant API base URL, e.g. `http://YOUR_HA_IP:8123/api` (for stats page API Usage display) |
| `HA_TOKEN` | *(optional)* | Home Assistant long-lived access token (for stats page API Usage display) |

## Daily Review

The Daily Review feature integrates with [Frigate's Review Summarize API](https://docs.frigate.video/configuration/genai/genai_review) to fetch and display AI-generated security summaries. The buffer stores these reviews separately from event clips (90-day retention by default).

### GET /stats-page

Standalone stats dashboard. Opens from the "Stats" button in the player header. Shows: API Usage (Month to Date cost/tokens from HA helpers when configured), Event Counts, Events by Camera, Storage, Recent Activity, Errors, System. Configurable auto-refresh (`stats_refresh_seconds`), manual Refresh button. "Back to Events" links to `/player`.

### GET /daily-review

Web page with date selector and formatted markdown review. Opens from the "Daily Review" button in the player header.

- **Date dropdown**: Select any available date; defaults to previous day
- **Load Review**: Fetch/cache review for selected date from Frigate
- **Current Day Review**: Fetch today's review (midnight to current time) from Frigate

### GET /api/daily-review/dates

List available cached review dates.

### GET /api/daily-review/{date}

Get review for date (YYYY-MM-DD). Fetches from Frigate if not cached. Use `?force=1` to re-fetch.

### GET /api/daily-review/current

Fetch current day review (midnight to now) from Frigate. Saves as partial review for today.

### Scheduled Job

At the configured hour (default 1am), the buffer fetches the previous day's review from Frigate and saves it. Old reviews are removed based on `daily_review_retention_days`.

## Camera/Label Filtering

The orchestrator filters events on a per-camera basis. Filter order: **camera/label first**, then **Smart Zone Filtering** (if `event_filters` is configured).

- **Camera must be listed**: Cameras not in the config are filtered out entirely
- **Per-camera labels**: Each camera can have its own label whitelist
- **Empty labels = allow all**: If `labels: []` is empty, all labels are allowed for that camera
- **Filtered events**: Events that don't match are silently ignored (visible in DEBUG logs)

Example: Different labels for different cameras:

```yaml
cameras:
  # Doorbell - only person and package events
  - name: "Doorbell"
    labels:
      - "person"
      - "package"

  # Driveway - only vehicle events
  - name: "Driveway"
    labels:
      - "car"

  # Backyard - allow ALL labels
  - name: "Backyard"
    labels: []
```

## Smart Zone Filtering

Optional per-camera `event_filters` reduce background noise by creating events **only** when an object enters a tracked zone (e.g., driveway). Exceptions create events regardless of zone.

**Flow:** The buffer listens for both `type: new` and `type: update` on `frigate/events`. For each message, if the event is not yet tracked, it runs the decision tree: (1) Does the label or sub_label match an exception? → Start immediately. (2) Has the object entered any zone in `tracked_zones`? → Start. (3) Otherwise → Defer. Each `update` re-evaluates until the event is created.

- **tracked_zones**: Frigate zone names. Create an event **only** when the object enters one of these zones. A car on the road is deferred until it enters the driveway.
- **exceptions**: Labels or sub_labels (e.g. `person`, `UPS`, `FedEx`) that create an event regardless of zone. Put both in the same list; matching is case-insensitive.
- **Late Start**: When an event is first created from an `update` message (e.g., car enters driveway), the original `start_time` from the payload is used so clips and review windows cover the full event.

Omit `event_filters` for legacy behavior (all events start immediately on `new`). At startup, the buffer logs `Smart Zone Filtering:` with per-camera `tracked_zones` and `exceptions` when configured.

```yaml
  - name: "Front_Yard"
    labels:
      - "car"
      - "truck"
    event_filters:
      tracked_zones:
        - driveway
      exceptions:
        - "person"
        - "UPS"
        - "FedEx"
```

## API Endpoints

### GET /events/<camera>/<subdir>/timeline

Per-event notification timeline page. Shows data received from Frigate (MQTT), clip export request/response (including full Frigate API response for debugging failures), Frigate Review Summarize API requests/responses, and payloads sent to Home Assistant. Rendered from `notification_timeline.json` in the event folder.

- **Back to Player** — links to `/player?camera=X&subdir=Y` so the player opens on the same event
- **Download Timeline** — downloads `notification_timeline.json`
- **Event Files** — download links for all files: root-level (notification_timeline.json, review_summary.md, summary.txt, metadata.json); and per-camera subdirs (clip.mp4, snapshot.jpg, metadata.json, etc.) for consolidated events

### GET /player

Built-in event viewer web page. Open in a browser or embed as an HA iframe card.

Features:
- Single-column responsive layout (scales to any device)
- HTML5 video player with snapshot poster
- Expandable AI analysis: each GenAI event from the timeline with its own expand/collapse (Expand button only shown when content is truncated); identical descriptions are deduplicated so each unique analysis is shown once (raw timeline keeps all entries for debugging); cross-camera review (or "Review Summary" when single camera); "No activity" boilerplate hidden when GenAI data is present
- Event Details (Cameras & Zones, label, timestamp): shows all cameras with affected zones from timeline (e.g., `Doorbell: Front_Porch, Front Yard` / `Carport: No Zones Indicated`)
- Camera filter dropdown
- Reviewed/Unreviewed/All filter (defaults to unreviewed)
- Stats button in header — links to `/stats-page` (standalone stats dashboard)
- "View most recent notification" link loads `/player?filter=all` so the most recent event is shown first (ignores reviewed/unreviewed filter)
- "Mark Reviewed" per-event and "Mark All Reviewed" bulk action
- Prev/Next event navigation
- Download and delete buttons
- **View Timeline** link — opens per-event page showing data from Frigate, Frigate API requests/responses, and HA notification payloads (saved as `notification_timeline.json`)
- Auto-refresh every 30 seconds (pauses during video playback)
- When no events: "No Events Found" with link to Stats page
- Dark theme optimized for HA dark mode

```
http://YOUR_BUFFER_IP:5055/player
```

### GET /cameras

List available cameras.

```bash
curl http://localhost:5055/cameras
```

Response:
```json
{
  "cameras": ["doorbell", "front_yard"],
  "default": "doorbell"
}
```

### GET /events

List all events across all cameras (global view).

**Query Parameters:**
- `?filter=unreviewed` (default) — only unreviewed events
- `?filter=reviewed` — only reviewed events
- `?filter=all` — all events

```bash
curl http://localhost:5055/events
curl http://localhost:5055/events?filter=all
```

Response:
```json
{
  "cameras": ["doorbell", "front_yard"],
  "total_count": 5,
  "events": [
    {
      "event_id": "1234567890.123-abcdef",
      "camera": "doorbell",
      "subdir": "1234567890_1234567890.123-abcdef",
      "timestamp": "1234567890",
      "label": "person",
      "title": "Person at Front Door",
      "description": "A person in blue jacket approaching the door",
      "severity": "alert",
      "threat_level": 0,
      "review_summary": "# Security Summary...",
      "summary": "Event ID: ...\nCamera: ...",
      "has_clip": true,
      "has_snapshot": true,
      "viewed": false,
      "ongoing": false,
      "hosted_clip": "/files/doorbell/1234567890_eventid/clip.mp4",
      "hosted_snapshot": "/files/doorbell/1234567890_eventid/snapshot.jpg",
      "cameras_with_zones": [{"camera": "doorbell", "zones": ["Front_Porch", "Front Yard"]}],
      "genai_entries": [{"title": "Person at door", "shortSummary": "...", "scene": "...", "time": "Friday, 2:30 PM"}]
    }
  ]
}
```

`genai_entries` are deduplicated by title, shortSummary, and scene so each unique analysis appears once; the raw `notification_timeline.json` still contains all timeline entries for debugging.

### GET /events/{camera}

List events for a specific camera.

**Query Parameters:**
- `?filter=unreviewed` (default) — only unreviewed events
- `?filter=reviewed` — only reviewed events
- `?filter=all` — all events

```bash
curl http://localhost:5055/events/doorbell
curl http://localhost:5055/events/doorbell?filter=all
```

### GET /files/{path}

Serve stored files (clips, snapshots). Clips are already transcoded to H.264.

```bash
curl http://localhost:5055/files/doorbell/1234567890_eventid/clip.mp4 --output clip.mp4
```

### POST /delete/{folder}

Manually delete an event folder.

```bash
curl -X POST http://localhost:5055/delete/doorbell/1234567890_eventid
```

### POST /viewed/{camera}/{subdir}

Mark a specific event as reviewed.

```bash
curl -X POST http://localhost:5055/viewed/doorbell/1234567890_eventid
```

### DELETE /viewed/{camera}/{subdir}

Remove the reviewed marker from an event.

```bash
curl -X DELETE http://localhost:5055/viewed/doorbell/1234567890_eventid
```

### POST /viewed/all

Mark all events across all cameras as reviewed.

```bash
curl -X POST http://localhost:5055/viewed/all
```

### GET /stats

Get statistics for the stats dashboard (events, storage, errors, system). Used by the Stats page. When HA is configured via `ha.base_url` and `ha.token` in config.yaml, optionally includes `ha_helpers` with `gemini_month_cost` and `gemini_month_tokens` from HA input_number helpers.

```bash
curl http://localhost:5055/stats
```

Response:
```json
{
  "events": {
    "today": 12,
    "this_week": 45,
    "this_month": 120,
    "total_reviewed": 80,
    "total_unreviewed": 40,
    "by_camera": {"doorbell": 60, "front_yard": 60}
  },
  "ha_helpers": {
    "gemini_month_cost": 0.00123,
    "gemini_month_tokens": 1234567
  },
  "storage": {
    "total_mb": 1250,
    "total_gb": 1.22,
    "by_camera": {"doorbell": {"mb": 600, "gb": null}, "front_yard": {"mb": null, "gb": 1.2}},
    "breakdown": {"clips_mb": 1100, "snapshots_mb": 80, "descriptions_mb": 2}
  },
  "errors": [
    {"ts": "2025-02-12 10:30:00", "level": "ERROR", "message": "FFmpeg timeout for event xyz"}
  ],
  "last_cleanup": {"at": "2025-02-12 09:00:00", "deleted": 5},
  "most_recent": {"event_id": "abc123", "camera": "doorbell", "url": "/player?filter=all", "timestamp": 1707742800},
  "system": {
    "uptime_seconds": 3600,
    "mqtt_connected": true,
    "active_events": 2,
    "retention_days": 3,
    "cleanup_interval_hours": 1,
    "stats_refresh_seconds": 60,
    "storage_path": "/app/storage"
  }
}
```

Errors are limited to the last 10 (see container logs for full history). Storage is shown in MB or GB (GB when over 1 GB per camera). The `most_recent.url` links to `/player?filter=all` so the most recent event is shown first when the page loads. The `ha_helpers` object is optional and only included when HA is configured via `ha.base_url` and `ha.token` in config.yaml (or `HA_URL` and `HA_TOKEN` env vars).

### GET /status

Get orchestrator health status (for monitoring).

```bash
curl http://localhost:5055/status
```

Response:
```json
{
  "online": true,
  "mqtt_connected": true,
  "uptime_seconds": 3661.5,
  "uptime": "1:01:01",
  "started_at": "2024-01-15 10:30:00",
  "active_events": {
    "total_active": 2,
    "by_phase": {"NEW": 1, "DESCRIBED": 1, "FINALIZED": 0, "SUMMARIZED": 0},
    "by_camera": {"Doorbell": 2}
  },
  "config": {
    "retention_days": 3,
    "log_level": "DEBUG",
    "ffmpeg_timeout": 60,
    "summary_padding_before": 15,
    "summary_padding_after": 15
  }
}
```

## MQTT Notifications

The orchestrator publishes notifications to `frigate/custom/notifications`:

```json
{
  "event_id": "1234567890.123-abcdef",
  "status": "new|described|finalized|clip_ready|summarized|overflow",
  "phase": "NEW|DESCRIBED|FINALIZED|SUMMARIZED|OVERFLOW",
  "camera": "Doorbell",
  "label": "person",
  "title": "Person at Front Door",
  "message": "A person in blue jacket approaching the door",
  "image_url": "http://YOUR_BUFFER_IP:5055/files/doorbell/1234567890_eventid/snapshot.jpg",
  "video_url": "http://YOUR_BUFFER_IP:5055/files/doorbell/1234567890_eventid/clip.mp4",
  "player_url": "http://YOUR_BUFFER_IP:5055/player",
  "tag": "frigate_1234567890.123-abcdef",
  "timestamp": 1234567890.123,
  "threat_level": 0,
  "critical": false
}
```

### Phase-Specific Messages

When GenAI descriptions are available (Frigate 0.17 with GenAI configured), they are used as the notification title and message. When not available, each phase shows distinct fallback text:

| Status | Message Content |
|--------|----------------|
| `new` | Best description available, or `"Person detected at Doorbell"` |
| `snapshot_ready` | Best description available, or `"Person detected at Doorbell"` |
| `described` | AI description (this IS the new content) |
| `clip_ready` | `"Video available. {best description}"` — combines status context with description |
| `finalized` | GenAI description (this IS the new content) |
| `summarized` | First meaningful line from review summary (truncated to 200 chars) — **skipped** when GenAI returns "No Concerns were found during this time period" |

Every notification includes an `image_url` that is always buffer-based for Companion app reachability: local snapshot once downloaded, or a proxy to Frigate (`/api/events/{event_id}/snapshot.jpg`). The `video_url` field is included once the clip is downloaded.

### Rate Limiting

Notifications are rate-limited to prevent notification flooding:

- **Max 2 notifications per 5 seconds** - Excess notifications are queued
- **Queue size limit: 10** - When exceeded, queue is cleared and an overflow notification is sent
- **Overflow notification** - Directs user to review events on the dashboard

The overflow notification has `status: "overflow"` and `event_id: "overflow_summary"`.

## Security

This application includes several security measures to protect against common vulnerabilities:

- **Path Traversal Protection**: The `/files` and `/delete` endpoints have been hardened to prevent path traversal attacks. All file and folder paths are resolved to their canonical form and checked to ensure they are within the designated storage directory.
- **Information Leakage Prevention**: The `/status` endpoint has been updated to remove sensitive configuration details, such as the MQTT broker and Frigate URL, from the JSON response. This reduces the risk of information leakage.
- **No Shell Injection**: The application uses `subprocess.Popen` without `shell=True`, which prevents shell injection attacks.

It is recommended to run this application in a containerized environment and to restrict access to the API endpoints to trusted users and services.

## Home Assistant Integration

### Binary Sensor for Monitoring

Create a REST binary sensor to monitor the orchestrator:

```yaml
binary_sensor:
  - platform: rest
    name: "Frigate Buffer Online"
    resource: http://YOUR_FRIGATE_BUFFER_IP:5055/status
    method: GET
    value_template: "{{ value_json.online and value_json.mqtt_connected }}"
    scan_interval: 60
    device_class: connectivity
```

### Notification Automation

This automation uses a dynamic phone target via a helper. Sound plays only on the initial detection; subsequent updates (clip, AI description) silently replace the notification. **Level 2 (critical) threats bypass phone volume/DND and keep all updates audible.**

```yaml
alias: "Frigate Orchestrator: Phone Notifications"
description: "Ring-style notifications with threat level support."
triggers:
  - trigger: mqtt
    topic: frigate/custom/notifications
actions:
  - variables:
      payload: "{{ trigger.payload_json }}"
      target_phone: "{{ states('input_text.notification_target_phone') }}"
      is_critical: "{{ payload.critical | default(false) }}"
      is_new: "{{ payload.status == 'new' }}"

  - condition: template
    value_template: "{{ target_phone | length > 0 }}"

  - action: "notify.{{ target_phone }}"
    data:
      title: "{{ payload.title }}"
      message: "{{ payload.message }}"
      data:
        tag: "{{ payload.tag }}"
        url: "{{ states('input_text.frigate_buffer_url') }}/player"
        clickAction: "{{ states('input_text.frigate_buffer_url') }}/player"
        image: >-
          {% if payload.image_url %}{{ payload.image_url }}{% endif %}
        video: >-
          {% if payload.video_url %}{{ payload.video_url }}{% endif %}
        # Critical (level 2): always audible, bypasses DND
        # Normal: only "new" gets sound
        importance: >-
          {% if is_critical %}max{% elif is_new %}high{% else %}low{% endif %}
        ttl: 0
        sound: >-
          {% if is_critical or is_new %}default{% else %}none{% endif %}
        channel: >-
          {% if is_critical %}alarm{% else %}general{% endif %}
        push:
          interruption-level: >-
            {% if is_critical %}critical{% elif is_new %}time-sensitive{% else %}passive{% endif %}
        sticky: true
        notification_icon: >-
          {% if is_critical %}mdi:shield-alert{% else %}mdi:shield-check{% endif %}

  - if:
      - condition: template
        value_template: "{{ payload.status in ['clip_ready', 'finalized', 'summarized'] }}"
    then:
      - action: homeassistant.update_entity
        target:
          entity_id: sensor.frigate_feed_raw
      - action: input_number.set_value
        target:
          entity_id: input_number.security_event_index
        data:
          value: 0
```

### Dashboard Card (iframe)

The simplest dashboard setup uses the built-in event viewer via an iframe card. The viewer handles camera selection, event navigation, AI summaries, video playback, and downloads — no HA helpers or sensors needed.

```yaml
type: iframe
url: "http://YOUR_BUFFER_IP:5055/player"
aspect_ratio: "16:9"
```

### Stats Page — API Usage (optional)

The Stats page (`/stats-page`) can display Month to Date API cost and token usage when Home Assistant helpers and REST API are configured. Add to `config.yaml`:

```yaml
ha:
  base_url: "http://YOUR_HA_IP:8123/api"
  token: "YOUR_LONG_LIVED_ACCESS_TOKEN"
  gemini_cost_entity: "input_number.gemini_daily_cost"
  gemini_tokens_entity: "input_number.gemini_total_tokens"
```

Or set `HA_URL` and `HA_TOKEN` environment variables. The cost is formatted to 5 decimal places.

### Required Helpers

Create at Settings > Devices & Services > Helpers.

**1. Notification Target Phone** (Text)
- **Entity ID**: `input_text.notification_target_phone`
- **Value**: Your phone's service name (e.g., `mobile_app_sm_s928u`)

**2. Frigate Buffer URL** (Text)
- **Entity ID**: `input_text.frigate_buffer_url`
- **Value**: Base URL of the buffer (e.g., `http://YOUR_BUFFER_IP:5055`) — **must be reachable from the Companion app** (VPN, Nabu Casa, or same network). Notification images use this base; the buffer proxies snapshots from Frigate when needed.

### Events Sensor (optional)

If you need event data in HA automations or template sensors, create a REST sensor:

```yaml
sensor:
  - platform: rest
    name: "Frigate Feed Raw"
    resource: http://YOUR_FRIGATE_BUFFER_IP:5055/events
    value_template: "{{ value_json.total_count }}"
    json_attributes:
      - events
      - cameras
    scan_interval: 60
```

## Docker Compose

**Option 1: Build from local clone** — Copy and customize the example:

```bash
cp docker-compose.example.yaml docker-compose.yaml
# Edit docker-compose.yaml: update paths, BUFFER_IP, FRIGATE_URL, MQTT_BROKER
docker compose up -d
```

**Option 2: Build from private GitHub (token pull)** — For Dockge, Portainer, or headless servers. Copy `docker-compose.example.yaml` to `docker-compose.yaml`, comment out `build: .`, and uncomment the `build.context` block with your GitHub username and PAT (repo scope):

```yaml
build:
  context: https://YOUR_GITHUB_USERNAME:YOUR_GITHUB_PAT@github.com/jballard86/frigate-event-buffer.git#main
```

See `docker-compose.example.yaml` for the full template with all env vars and optional config bind mount.

Place `config.yaml` in the storage volume — it is found at `/app/storage/config.yaml` automatically. Optional: bind-mount a config file to `/app/config.yaml`.

## Debug Logging

Enable debug logging to troubleshoot issues:

1. **Via config.yaml:**
   ```yaml
   settings:
     log_level: "DEBUG"
   ```

2. **Via environment variable:**
   ```bash
   docker run -e LOG_LEVEL=DEBUG ...
   ```

Debug output includes:
- MQTT messages received (topic, payload size)
- Filtering decisions (camera/label allow/deny, Smart Zone Filtering ignore when event deferred)
- Event state transitions
- File operations (download start/end, transcode progress)
- FFmpeg commands and timing

## FFmpeg Process Safety

The orchestrator includes safeguards against hung FFmpeg processes:

- **60-second timeout**: Configurable via `ffmpeg_timeout_seconds`
- **Graceful termination**: SIGTERM first, wait 5s, then SIGKILL
- **Zombie reaping**: Ensures child processes are properly cleaned up
- **Fallback**: If transcoding fails/times out, original clip is used

## Event Lifecycle

The buffer listens to **`frigate/events`** for `type: new`, `type: update`, and `type: end`. With Smart Zone Filtering enabled, event creation may be deferred: a `new` or `update` is ignored until the object enters a tracked zone or matches an exception. When creation is triggered by a later `update` (Late Start), the original `start_time` from the payload is used so clips cover the full event.

| Time | MQTT Topic | Phase | Action |
|------|------------|-------|--------|
| T+0s | `frigate/events` (type=new or update) | NEW | Create folder if not deferred by Smart Zone Filtering; send instant notification; fetch local snapshot after delay. With tracked_zones, may defer until object enters tracked zone (Late Start). |
| T+5s | `frigate/{camera}/tracked_object_update` (type=description) | DESCRIBED | Update with AI description |
| T+30s | `frigate/events` (type=end) | - | Download snapshot; request clip via Frigate Export API (POST with `playback`/`name` JSON body; poll by export_id). For consolidated events, each camera gets its own time range and representative event ID from sub-events. |
| T+35s | *(background processing)* | - | Transcode clip, write summary, send `clip_ready`; falls back to per-event clip if export fails or times out. Export failures log full response at WARNING; 404 on clip download is not retried. |
| T+45s | `frigate/reviews` (type=genai) | FINALIZED | Update with GenAI metadata (title, description, threat_level) from `data.metadata` |
| T+55s | Frigate Review Summary API | SUMMARIZED | Fetch cross-camera review summary with configurable time padding, write `review_summary.md` |

### Threat Level Behavior

Events with `potential_threat_level: 2` (Critical) receive special notification handling:

| Threat Level | Initial Alert | Follow-up Updates | DND/Volume |
|-------------|--------------|-------------------|------------|
| 0 (Normal) | Audible | Silent | Respects settings |
| 1 (Suspicious) | Audible | Silent | Respects settings |
| 2 (Critical) | Audible + alarm channel | **All audible** | **Bypasses DND** |

The threat level is determined by Frigate's GenAI analysis based on the `activity_context_prompt` in your Frigate configuration.

## Storage Structure

Events are organized by camera. Event subdir format: `{timestamp}_{frigate_event_id}` (e.g. Frigate event ID `1234567890.123-abcdef` becomes subdir `1234567890_1234567890.123-abcdef`).

```
/app/storage/
├── doorbell/
│   ├── 1234567890_1234567890.123-abcdef/
│   │   ├── clip.mp4                # H.264 transcoded video (from Frigate Export API; full event + buffer; downloadable from timeline)
│   │   ├── snapshot.jpg            # Event snapshot
│   │   ├── summary.txt             # Event metadata (human-readable)
│   │   ├── metadata.json           # Structured metadata (threat_level, etc.)
│   │   ├── review_summary.md       # Frigate review summary (markdown)
│   │   ├── notification_timeline.json  # Data pipeline log (Frigate MQTT, clip export request/response, review summarize, HA)
│   │   └── .viewed                 # Review marker (created when marked as reviewed)
│   └── 1234567891_1234567891.456-ghijkl/
│       └── ...
├── front_yard/
│   └── 1234567892_1234567892.789-mnopqr/
│       └── ...
```

Camera folder names are sanitized (lowercase, spaces to underscores).

## Troubleshooting

### MQTT Not Connecting

Check `/status` endpoint - `mqtt_connected` should be `true`. Verify:
- MQTT broker is accessible from the container
- No firewall blocking port 1883
- Correct `MQTT_BROKER` environment variable

### Events Not Being Processed

1. Enable debug logging: `LOG_LEVEL=DEBUG`
2. Check for filter messages in logs:
   ```
   Filtered out event from camera 'BackYard' (not configured)
   Filtered out 'car' on 'Doorbell' (allowed: ['person', 'package'])
   Ignoring <event_id> (smart zone filter: not in tracked zones, entered=['road'])
   ```
3. Verify camera names match exactly (case-sensitive)
4. Check that the camera is listed in the `cameras` config
5. **Smart Zone Filtering**: If `event_filters` is configured, events may be deferred until the object enters a tracked zone or matches an exception. Check `tracked_zones` and `exceptions` in your config.

### Screenshots Not Showing in Notifications

Notification images use buffer URLs (local file or `/api/events/{event_id}/snapshot.jpg` proxy). The Companion app must be able to reach `input_text.frigate_buffer_url` to fetch images. Ensure:
- **Same network**: Phone and buffer on same LAN, or
- **VPN / exit node**: Phone routes through a server that can reach the buffer, or
- **Nabu Casa / reverse proxy**: Buffer URL points to a publicly reachable address
- Enable `LOG_LEVEL: DEBUG` to verify `image_url` in published payloads

### Clips Not Downloading

The app uses Frigate's Export API first (`POST /api/export/<camera>/start/<start>/end/<end>` with `Content-Type: application/json` and body `{"playback": "realtime", "name": "export_<event_id>"}` per Frigate schema); it polls `/api/exports` by `export_id` for up to 90 seconds. For **consolidated events**, each camera is exported with that camera's own time range and a representative event ID (from sub-events), which reduces 404s and wrong footage. If export fails or returns `success: false`, the full raw response is logged at WARNING. If export fails or times out, it falls back to the per-event events API (placeholder clip for consolidated events). HTTP 404 on clip download is treated as "no recording available" and is not retried; retries (up to 3, 5s apart) apply only to HTTP 400 (Not Ready) and other transient errors. If clips still fail:
- Check logs for `Export failed for <event_id>. Status: ... Response: ...` or `No recording available for event ...` or `Clip not ready for ... (HTTP 400), retrying`
- Verify `FRIGATE_URL` is correct and accessible
- Verify Frigate API is responding: `curl http://your-frigate-ip:5000/api/events`
- Camera names must match Frigate config exactly (case-sensitive, e.g. `Doorbell` not `doorbell`)
- Under high server load, exports can timeout; the placeholder fallback provides a clip from the primary event
- The timeline page shows "Clip export request" (with `representative_id` for consolidated events), "Clip export response" (with full Frigate API response for debugging), and "Placeholder clip (events API fallback)" entries

### FFmpeg Timeouts

If transcoding is timing out:
- Increase `ffmpeg_timeout_seconds` in config.yaml
- Check Ryzen CPU load during transcoding
- Consider using a faster preset (already using 'fast')

### Check Container Logs

```bash
docker logs frigate-buffer
docker logs -f frigate-buffer  # Follow logs
```

## License

MIT
