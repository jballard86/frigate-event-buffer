# Frigate Event Buffer

A state-aware orchestrator that listens to Frigate NVR events via MQTT, tracks them through their lifecycle, sends Ring-style sequential notifications to Home Assistant, and maintains a rolling 3-day evidence locker.

## Features

- **MQTT Event Tracking**: Subscribes to Frigate's MQTT topics to detect and track events in real-time
- **Three-Phase Lifecycle**: Tracks events through NEW → DESCRIBED → FINALIZED phases
- **Ring-Style Notifications**: Sends progressive updates to Home Assistant as event details emerge
- **Camera/Label Filtering**: Only process events from specific cameras or with specific labels
- **Multi-Camera Support**: Handles events from multiple cameras simultaneously without state collision
- **Auto-Transcoding**: Downloads clips from Frigate and transcodes to H.264 for broad compatibility
- **FFmpeg Safety**: 60-second timeout with graceful termination prevents zombie processes
- **Rolling Retention**: Automatically cleans up events older than the retention period (default: 3 days)
- **REST API**: Serves events, clips, and snapshots to your Home Assistant dashboard
- **Debug Logging**: Configurable log levels for troubleshooting

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│              State-Aware Orchestrator                    │
├─────────────────────────────────────────────────────────┤
│  MQTT Client          EventStateManager    Flask API    │
│  ├─ frigate/events    ├─ active_events     ├─ /events   │
│  ├─ frigate/+/        ├─ phase tracking    ├─ /delete   │
│  │   tracked_object   │   (NEW→DESCRIBED   ├─ /files    │
│  └─ frigate/reviews   │    →FINALIZED)     └─ /status   │
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
  -v /mnt/user/appdata/frigate_buffer/config.yaml:/app/config.yaml:ro \
  frigate-buffer
```

### 3. Verify It's Running

```bash
curl http://localhost:5055/status
```

## Configuration

Configuration is loaded from three sources (in order of priority):

1. **Environment variables** (highest priority)
2. **config.yaml** file
3. **Default values** (lowest priority)

### config.yaml

Copy the example config and customize it:

```bash
cp config.yaml.example config.yaml
# Edit config.yaml with your values
```

The config file structure:

```yaml
# Camera filtering - only process events from these cameras
# Leave empty or omit to allow all cameras
cameras:
  allowed:
    - "Doorbell"
    - "Front_Yard"
    - "Carport"

# Label filtering - only process events with these labels
# Leave empty or omit to allow all labels
labels:
  allowed:
    - "person"
    - "package"
    - "car"
    - "dog"
    - "cat"

# Application settings
settings:
  retention_days: 3
  cleanup_interval_hours: 1
  ffmpeg_timeout_seconds: 60
  log_level: "DEBUG"  # DEBUG, INFO, WARNING, ERROR

# Network configuration (REQUIRED - no defaults)
network:
  mqtt_broker: "YOUR_MQTT_BROKER_IP"
  mqtt_port: 1883
  frigate_url: "http://YOUR_FRIGATE_IP:5000"
  ha_ip: "YOUR_HOME_ASSISTANT_IP"
  flask_port: 5055
  storage_path: "/app/storage"
```

### Environment Variables

Environment variables override config.yaml values:

| Variable | Default | Description |
|----------|---------|-------------|
| `MQTT_BROKER` | *(required)* | MQTT broker IP address |
| `MQTT_PORT` | `1883` | MQTT broker port |
| `HA_IP` | *(required)* | Home Assistant IP (used in notification URLs) |
| `FRIGATE_URL` | *(required)* | Frigate API base URL |
| `STORAGE_PATH` | `/app/storage` | Storage directory inside container |
| `RETENTION_DAYS` | `3` | Days to retain event folders |
| `FLASK_PORT` | `5055` | Flask server port |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |

## Camera/Label Filtering

The orchestrator can filter events based on camera name and detected label:

- **Empty lists = allow all**: If `allowed` is empty or omitted, all cameras/labels are processed
- **Non-empty lists = whitelist**: Only cameras/labels in the list are processed
- **Filtered events**: Events that don't match are silently ignored (visible in DEBUG logs)

Example: Only process person and package events from the doorbell:

```yaml
cameras:
  allowed:
    - "Doorbell"

labels:
  allowed:
    - "person"
    - "package"
```

## API Endpoints

### GET /events

List all stored events with summaries.

```bash
curl http://localhost:5055/events
```

### GET /files/{path}

Serve stored files (clips, snapshots). Clips are already transcoded to H.264.

```bash
curl http://localhost:5055/files/1234567890_eventid/clip.mp4 --output clip.mp4
```

### POST /delete/{folder}

Manually delete an event folder.

```bash
curl -X POST http://localhost:5055/delete/1234567890_eventid
```

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
    "by_phase": {"NEW": 1, "DESCRIBED": 1, "FINALIZED": 0},
    "by_camera": {"Doorbell": 2}
  },
  "config": {
    "mqtt_broker": "YOUR_LOCAL_IP",
    "frigate_url": "http://YOUR_LOCAL_IP:5000",
    "retention_days": 3,
    "allowed_cameras": ["Doorbell", "Front_Yard"],
    "allowed_labels": ["person", "package"],
    "log_level": "DEBUG",
    "ffmpeg_timeout": 60
  }
}
```

## MQTT Notifications

The orchestrator publishes notifications to `frigate/custom/notifications`:

```json
{
  "event_id": "1234567890.123-abcdef",
  "status": "new|described|finalized|clip_ready",
  "phase": "NEW|DESCRIBED|FINALIZED",
  "camera": "Doorbell",
  "label": "person",
  "title": "Person at Front Door",
  "message": "A person in blue jacket approaching the door",
  "image_url": "http://YOUR_HA_IP:5055/files/1234567890_eventid/snapshot.jpg",
  "video_url": "http://YOUR_HA_IP:5055/files/1234567890_eventid/clip.mp4",
  "tag": "frigate_1234567890.123-abcdef",
  "timestamp": 1234567890.123
}
```

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

```yaml
automation:
  - alias: "Frigate Event Notification"
    trigger:
      - platform: mqtt
        topic: frigate/custom/notifications
    action:
      - service: notify.mobile_app_your_phone
        data:
          title: "{{ trigger.payload_json.title }}"
          message: "{{ trigger.payload_json.message }}"
          data:
            image: "{{ trigger.payload_json.image_url }}"
            video: "{{ trigger.payload_json.video_url }}"
            tag: "{{ trigger.payload_json.tag }}"
            notification_icon: "mdi:cctv"
```

## Docker Compose

```yaml
version: '3.8'

services:
  frigate-buffer:
    build: .
    container_name: frigate-buffer
    restart: unless-stopped
    ports:
      - "5055:5055"
    volumes:
      - /mnt/user/appdata/frigate_buffer:/app/storage
      - /mnt/user/appdata/frigate_buffer/config.yaml:/app/config.yaml:ro
    environment:
      - LOG_LEVEL=DEBUG  # Optional: override config.yaml
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5055/status"]
      interval: 60s
      timeout: 10s
      retries: 3
```

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
| T+0s | `frigate/events` (type=new) | NEW | Create folder, send initial notification |
| T+5s | `frigate/{camera}/tracked_object_update` | DESCRIBED | Update with AI description |
| T+30s | `frigate/events` (type=end) | - | Download snapshot & clip, transcode |
| T+45s | `frigate/reviews` | FINALIZED | Write summary, send final notification |

## Storage Structure

```
/app/storage/
├── 1234567890_event-id-1/
│   ├── clip.mp4          # H.264 transcoded video
│   ├── snapshot.jpg      # Event snapshot
│   └── summary.txt       # Event metadata
├── 1234567891_event-id-2/
│   └── ...
```

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
   Filtered out event from camera 'BackYard' (allowed: ['Doorbell'])
   ```
3. Verify camera/label names match exactly (case-sensitive)

### Clips Not Downloading

Verify:
- `FRIGATE_URL` is correct and accessible
- Frigate API is responding: `curl http://your-frigate-ip:5000/api/events`

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
