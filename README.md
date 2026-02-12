# Frigate Event Buffer

A state-aware orchestrator that listens to Frigate NVR events via MQTT, tracks them through their lifecycle, sends Ring-style sequential notifications to Home Assistant, and maintains a rolling 3-day evidence locker. Compatible with **Frigate 0.17+** with full GenAI review integration.

## Features

- **Frigate 0.17 Compatible**: Supports Frigate 0.17's new MQTT payload structure including `type: "genai"` review messages and `data.metadata` fields
- **MQTT Event Tracking**: Subscribes to Frigate's MQTT topics to detect and track events in real-time
- **Four-Phase Lifecycle**: Tracks events through NEW → DESCRIBED → FINALIZED → SUMMARIZED phases
- **Ring-Style Notifications**: Sends progressive updates to Home Assistant with phase-specific messages as event details emerge
- **GenAI Integration**: Captures Frigate's GenAI titles, descriptions, and threat levels (via `data.metadata` in reviews and `description` type in tracked object updates)
- **Review Summaries**: Fetches rich markdown security reports from Frigate's review summary API with cross-camera context, timeline, and assessments
- **Threat Level Alerts**: Three-tier threat classification (0=Normal, 1=Suspicious, 2=Critical) — Level 2 alerts bypass phone volume/DND and keep all follow-up notifications audible
- **Camera/Label Filtering**: Only process events from specific cameras or with specific labels
- **Multi-Camera Support**: Handles events from multiple cameras simultaneously without state collision
- **Auto-Transcoding**: Downloads clips from Frigate and transcodes to H.264 for broad compatibility
- **Clip Download Retry**: Retries clip downloads up to 3 times on HTTP 400 (Frigate not ready), with 5-second delays
- **FFmpeg Safety**: 60-second timeout with graceful termination prevents zombie processes
- **Rolling Retention**: Automatically cleans up events older than the retention period (default: 3 days)
- **Notification Rate Limiting**: Max 2 notifications per 5 seconds with queue overflow protection
- **Built-in Event Viewer**: Self-contained web page at `/player` with video playback, AI analysis, event navigation, reviewed/unreviewed filtering, and download — embeddable as an HA iframe
- **Stats Dashboard**: "Stats" filter option and stats panel when no events — event counts (today/week/month), storage by camera, recent errors, last cleanup, system info; auto-refresh every 30s with manual Refresh button
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
│  ├─ frigate/+/        ├─ phase tracking    ├─ /events   │
│  │   tracked_object   │   (NEW→DESCRIBED   ├─ /cameras  │
│  └─ frigate/reviews   │    →FINALIZED      ├─ /files    │
│                       │    →SUMMARIZED)    ├─ /stats    │
│                       │                    └─ /status   │
└─────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Build the Docker Image

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

## Configuration

Configuration is loaded from three sources (in order of priority):

1. **Environment variables** (highest priority)
2. **config.yaml** file (searched at `/app/config.yaml`, `/app/storage/config.yaml`, `./config.yaml`)
3. **Default values** (lowest priority)

Place your `config.yaml` in the storage volume directory — it will be found automatically at `/app/storage/config.yaml` without needing a separate file bind mount.

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

  # Driveway - only vehicle events
  - name: "Front_Yard"
    labels:
      - "car"
      - "truck"

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

# Network configuration (REQUIRED - no defaults)
network:
  mqtt_broker: "YOUR_MQTT_BROKER_IP"
  mqtt_port: 1883
  frigate_url: "http://YOUR_FRIGATE_IP:5000"
  buffer_ip: "YOUR_BUFFER_IP"  # IP where this container is reachable
  flask_port: 5055
  storage_path: "/app/storage"
```

### Environment Variables

Environment variables override config.yaml values:

| Variable | Default | Description |
|----------|---------|-------------|
| `MQTT_BROKER` | *(required)* | MQTT broker IP address |
| `MQTT_PORT` | `1883` | MQTT broker port |
| `BUFFER_IP` | *(required)* | Buffer container's reachable IP (used in notification image/video URLs) |
| `FRIGATE_URL` | *(required)* | Frigate API base URL |
| `STORAGE_PATH` | `/app/storage` | Storage directory inside container |
| `RETENTION_DAYS` | `3` | Days to retain event folders |
| `FLASK_PORT` | `5055` | Flask server port |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `STATS_REFRESH_SECONDS` | `60` | Stats panel auto-refresh interval (seconds) |

## Camera/Label Filtering

The orchestrator filters events on a per-camera basis:

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

## API Endpoints

### GET /player

Built-in event viewer web page. Open in a browser or embed as an HA iframe card.

Features:
- Single-column responsive layout (scales to any device)
- HTML5 video player with snapshot poster
- GenAI title and full description display
- Event metadata (camera, label, timestamp)
- Camera filter dropdown
- Reviewed/Unreviewed/All/Stats filter (defaults to unreviewed)
- Stats dashboard when "Stats" is selected or when no events: event counts (today, week, month), reviewed/unreviewed totals, events per camera, storage by camera (clips, snapshots, descriptions), recent errors (last 10), last cleanup time, link to most recent notification, system info (uptime, MQTT, retention, etc.)
- "View most recent notification" link loads `/player?filter=all` so the most recent event is shown first (ignores reviewed/unreviewed filter)
- Stats layout: centered and symmetrical
- "Mark Reviewed" per-event and "Mark All Reviewed" bulk action
- Prev/Next event navigation
- Download and delete buttons
- Auto-refresh every 30 seconds (pauses during video playback)
- Stats view: configurable auto-refresh (default 60s via `stats_refresh_seconds`) plus manual Refresh button
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
      "hosted_clip": "/files/doorbell/1234567890_eventid/clip.mp4",
      "hosted_snapshot": "/files/doorbell/1234567890_eventid/snapshot.jpg"
    }
  ]
}
```

### GET /events/{camera}

List events for a specific camera.

```bash
curl http://localhost:5055/events/doorbell
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

Get statistics for the player dashboard (events, storage, errors, system). Used by the Stats view in the player.

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

Errors are limited to the last 10 (see container logs for full history). Storage is shown in MB or GB (GB when over 1 GB per camera). The `most_recent.url` links to `/player?filter=all` so the most recent event is shown first when the page loads.

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
| `summarized` | First meaningful line from review summary (truncated to 200 chars) |

Every notification includes an `image_url` (Frigate API snapshot with `snapshot=true`, or local snapshot once downloaded). The `video_url` field is included once the clip is downloaded.

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

### Required Helpers

Create at Settings > Devices & Services > Helpers.

**1. Notification Target Phone** (Text)
- **Entity ID**: `input_text.notification_target_phone`
- **Value**: Your phone's service name (e.g., `mobile_app_sm_s928u`)

**2. Frigate Buffer URL** (Text)
- **Entity ID**: `input_text.frigate_buffer_url`
- **Value**: Base URL of the buffer (e.g., `http://YOUR_BUFFER_IP:5055`)

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

```yaml
services:
  frigate-buffer:
    build: .
    container_name: frigate-buffer
    restart: unless-stopped
    ports:
      - "5055:5055"
    volumes:
      - /mnt/user/appdata/frigate_buffer:/app/storage
      - /etc/localtime:/etc/localtime:ro
    environment:
      - MQTT_BROKER=YOUR_MQTT_BROKER_IP
      - FRIGATE_URL=http://YOUR_FRIGATE_IP:5000
      - BUFFER_IP=YOUR_BUFFER_IP
      - LOG_LEVEL=DEBUG  # Optional: override config.yaml
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5055/status"]
      interval: 60s
      timeout: 10s
      retries: 3
```

Place your `config.yaml` inside the storage volume (e.g., `/mnt/user/appdata/frigate_buffer/config.yaml`) — the app will find it automatically. No separate file bind mount needed.

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
- Filtering decisions (camera/label allow/deny)
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

| Time | MQTT Topic | Phase | Action |
|------|------------|-------|--------|
| T+0s | `frigate/events` (type=new) | NEW | Create folder, send instant notification (Frigate snapshot fallback), then fetch local snapshot after delay |
| T+5s | `frigate/{camera}/tracked_object_update` (type=description) | DESCRIBED | Update with AI description |
| T+30s | `frigate/events` (type=end) | - | Download snapshot & clip (with retry), transcode |
| T+35s | *(background processing)* | - | Write summary, send `clip_ready` notification |
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

Events are organized by camera:

```
/app/storage/
├── doorbell/
│   ├── 1234567890_event-id-1/
│   │   ├── clip.mp4            # H.264 transcoded video
│   │   ├── snapshot.jpg        # Event snapshot
│   │   ├── summary.txt         # Event metadata (human-readable)
│   │   ├── metadata.json       # Structured metadata (threat_level, etc.)
│   │   ├── review_summary.md   # Frigate review summary (markdown)
│   │   └── .viewed             # Review marker (created when marked as reviewed)
│   └── 1234567891_event-id-2/
│       └── ...
├── front_yard/
│   └── 1234567892_event-id-3/
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
   ```
3. Verify camera names match exactly (case-sensitive)
4. Check that the camera is listed in the `cameras` config

### Clips Not Downloading

The app retries clip downloads up to 3 times with 5-second delays when Frigate returns HTTP 400 (clip not yet processed). If clips still fail:
- Check logs for retry attempts: `Clip not ready for ... (HTTP 400), retrying`
- Verify `FRIGATE_URL` is correct and accessible
- Verify Frigate API is responding: `curl http://your-frigate-ip:5000/api/events`

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
