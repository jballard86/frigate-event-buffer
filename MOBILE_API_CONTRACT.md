# Mobile API Contract

**Source of truth for the native Android client** (Jetpack Compose + Retrofit) that talks to the Frigate Event Buffer Flask backend. Use this document to define Retrofit interfaces, data classes, and media URL construction.

## Base URL

- All paths in this document are **relative** to the server base URL.
- **Full URL** = `{baseUrl}{path}` (e.g. `http://192.168.21.50:5000/cameras`).
- The Android app should configure `baseUrl` (scheme + host + port); when on Tailscale, use the Tailscale hostname or IP and the port the Flask app listens on (e.g. `5000`).
- No global API prefix: event/list endpoints live at `/cameras`, `/events`; daily review at `/api/daily-review/*`; test pipeline at `/api/test-multi-cam/*`.

---

## 1. Events

### 1.1 List cameras

**Endpoint path:** `/cameras`  
**HTTP method:** `GET`  
**Query parameters:** None  
**Path variables:** None  
**Request body:** None  

**Response (200):**

```json
{
  "cameras": ["events", "front_door", "garage"],
  "default": "events"
}
```

| Field       | Type     | Nullable | Description |
|------------|----------|----------|-------------|
| `cameras`  | string[] | No       | Sorted list of camera names; includes `"events"` when consolidated events exist. |
| `default`  | string   | Yes      | First camera in the list; null if empty. |

---

### 1.2 List all events

**Endpoint path:** `/events`  
**HTTP method:** `GET`  
**Query parameters:**

| Name     | Required | Description |
|----------|----------|-------------|
| `filter` | No       | Default `unreviewed`. One of: `unreviewed`, `reviewed`, `all`, `saved`, `test_events`. |

**Path variables:** None  
**Request body:** None  

**Response (200):**

- `filter=saved`: `{ "cameras": string[], "total_count": number, "events": Event[] }`
- `filter=test_events`: `{ "cameras": ["events"], "total_count": number, "events": Event[] }`
- Otherwise: `{ "cameras": string[], "total_count": number, "events": Event[] }`

**Event object** (see **Section 1.11** for full shape): each item in `events` has `event_id`, `camera`, `subdir`, `timestamp`, `title`, `description`, `has_clip`, `hosted_clip`, `hosted_snapshot`, `hosted_clips`, `viewed`, `consolidated`, `ongoing`, `genai_entries`, etc.

**Error (500):** `{ "error": "string" }`

---

### 1.3 List events by camera

**Endpoint path:** `/events/<camera>`  
**HTTP method:** `GET`  
**Query parameters:**

| Name     | Required | Description |
|----------|----------|-------------|
| `filter` | No       | Same as 1.2: `unreviewed`, `reviewed`, `all`, `saved`, `test_events`. |

**Path variables:**

| Name     | Description |
|----------|-------------|
| `camera` | Camera name (e.g. `front_door`) or `events` for consolidated events. For `filter=saved` or `filter=test_events`, still use a camera name; server returns filtered list. |

**Request body:** None  

**Response (200):**

- With `filter`: `{ "camera": string, "events": Event[], "cameras": string[], "total_count": number }` (for saved/test_events).
- Otherwise: `{ "camera": string, "events": Event[] }`.

**Error (500):** `{ "error": "string" }`

---

### 1.4 Keep event (move to saved)

**Endpoint path:** `/keep/<path:event_path>`  
**HTTP method:** `POST`  
**Query parameters:** None  
**Path variables:**

| Name         | Description |
|--------------|-------------|
| `event_path` | Path under storage, e.g. `front_door/1739123456_abc123` or `events/1739123456_abc123`. Must **not** start with `saved/`. |

**Request body:** None (empty POST).

**Response (200):**

```json
{
  "status": "success",
  "message": "Moved to saved/events/1739123456_abc123"
}
```

**Errors:**

- 400: `{ "status": "error", "message": "Invalid path" }` or `"Event is already saved"` or `"Invalid destination path"`.
- 404: `{ "status": "error", "message": "Event not found or invalid path" }`.
- 409: `{ "status": "error", "message": "Saved event already exists" }`.
- 500: `{ "status": "error", "message": "string" }`.

---

### 1.5 Delete event

**Endpoint path:** `/delete/<path:subdir>`  
**HTTP method:** `POST`  
**Query parameters:** None  
**Path variables:**

| Name     | Description |
|----------|-------------|
| `subdir` | Path to folder to delete, e.g. `front_door/1739123456_abc123` or `saved/camera/subdir`. Multi-segment paths supported. |

**Request body:** None  

**Response (200):**

```json
{
  "status": "success",
  "message": "Deleted folder: front_door/1739123456_abc123"
}
```

**Errors:**

- 400: `{ "status": "error", "message": "Invalid folder or path" }`.
- 404: `{ "status": "error", "message": "Folder not found" }`.
- 500: `{ "status": "error", "message": "string" }`.

---

### 1.6 Mark event viewed

**Endpoint path:** `/viewed/<path:event_path>`  
**HTTP method:** `POST`  
**Query parameters:** None  
**Path variables:** `event_path` — same as 1.4 (e.g. `camera/subdir` or `events/ce_id`).  
**Request body:** None  

**Response (200):** `{ "status": "success" }`

**Errors:** 400 invalid path, 404 event not found, 500: `{ "status": "error", "message": "string" }`.

---

### 1.7 Unmark event viewed

**Endpoint path:** `/viewed/<path:event_path>`  
**HTTP method:** `DELETE`  
**Query parameters:** None  
**Path variables:** Same `event_path` as 1.6.  
**Request body:** None  

**Response (200):** `{ "status": "success" }` (idempotent; no error if `.viewed` already absent).  
**Errors:** 400 invalid path.

---

### 1.8 Mark all events viewed

**Endpoint path:** `/viewed/all`  
**HTTP method:** `POST`  
**Query parameters:** None  
**Path variables:** None  
**Request body:** None  

**Response (200):**

```json
{
  "status": "success",
  "marked": 5
}
```

| Field    | Type  | Description |
|----------|-------|-------------|
| `marked` | int   | Number of event folders that were marked viewed. |

**Error (500):** `{ "status": "error", "message": "string" }`.

---

### 1.9 Event timeline (HTML — web only)

**Endpoint path:** `/events/<path:event_path>/timeline`  
**HTTP method:** `GET`  
**Query parameters:** None  
**Path variables:** `event_path` — e.g. `camera/subdir` or `events/ce_id`.  

**Response (200):** HTML page (Flask template). **Not for mobile.** Use **1.10** for JSON.

**Errors:** 400 invalid path, 404 event not found.

---

### 1.10 Event timeline download (JSON)

**Endpoint path:** `/events/<path:event_path>/timeline/download`  
**HTTP method:** `GET`  
**Query parameters:** None  
**Path variables:** `event_path` — same as 1.9.  
**Request body:** None  

**Response (200):** JSON with `Content-Disposition: attachment; filename="notification_timeline.json"`.

```json
{
  "event_id": "1739123456_abc123",
  "entries": [
    {
      "ts": "2025-02-26T10:15:00.123456",
      "source": "frigate_mqtt",
      "direction": "in",
      "label": "Event start",
      "data": { "payload": { "type": "new", "after": { "camera": "front_door" } } }
    },
    {
      "ts": "2025-02-26T10:15:01.000000",
      "source": "frigate_api",
      "direction": "out",
      "label": "Clip export request",
      "data": {}
    }
  ]
}
```

| Field       | Type     | Nullable | Description |
|------------|----------|----------|-------------|
| `event_id` | string   | Yes      | Event or CE id. |
| `entries`  | object[] | No       | Timeline entries; each has `ts` (string), `source`, `direction`, `label`, `data` (object). |

**Errors:** 400 invalid path, 404 event not found, 500 error reading timeline.

---

### 1.11 Event object (full shape)

Used in `/events`, `/events/<camera>`, and anywhere the API returns a list or single event. Build Retrofit/Kotlin data classes from this.

| Field                 | Type                    | Nullable | Description |
|-----------------------|-------------------------|----------|-------------|
| `event_id`            | string                  | No       | Event or CE id. |
| `camera`              | string                  | No       | Camera name; `"events"` for consolidated. |
| `subdir`              | string                  | No       | Folder name (e.g. `1739123456_abc123`). |
| `timestamp`           | string                  | No       | Unix timestamp as string (e.g. `"1739123456"`). |
| `summary`             | string                  | No       | Raw summary text. |
| `title`               | string                  | Yes      | GenAI or parsed title. |
| `description`         | string                  | Yes      | GenAI or parsed description. |
| `scene`               | string                  | Yes      | Longer narrative. |
| `label`               | string                  | No       | e.g. `"person"`, `"car"`. |
| `severity`            | string                  | Yes      | Severity label. |
| `threat_level`        | int                     | No       | 0=normal, 1=suspicious, 2=critical. |
| `review_summary`      | string                  | Yes      | Markdown review summary. |
| `has_clip`            | boolean                 | No       | True if at least one clip or summary video exists. |
| `has_snapshot`        | boolean                 | No       | True if at least one snapshot exists. |
| `viewed`              | boolean                 | No       | True if event marked reviewed. |
| `hosted_clip`        | string                  | Yes      | Path to primary clip/summary (e.g. `/files/events/ce_id/summary.mp4`). |
| `hosted_snapshot`    | string                  | Yes      | Path to primary snapshot (e.g. `/files/events/ce_id/camera/snapshot.jpg`). |
| `hosted_clips`       | array of `{camera, url}` | No       | All clips; `url` is path. May include `"Summary video"`. |
| `cameras`             | string[]                | Optional | Present for consolidated events. |
| `cameras_with_zones`  | array of `{camera, zones: string[]}` | No | Cameras and zones from timeline. |
| `consolidated`       | boolean                 | Optional | True for CE. |
| `ongoing`             | boolean                 | No       | True if not yet finalized (no clip and recent). |
| `genai_entries`       | object[]                | No       | Deduplicated GenAI entries: `title`, `scene`, `shortSummary`, `time`, `potential_threat_level`. |
| `end_timestamp`       | number                  | Optional | Unix timestamp (float) of event end. |
| `saved`               | boolean                 | Optional | True when event is under `saved/`. |

**Media URLs:** Use `hosted_clip`, `hosted_snapshot`, and `hosted_clips[].url` as **paths**; full URL = `{baseUrl}{path}` (see **Section 2**).

---

## 2. Media / Proxy

### 2.1 Serve stored file

**Endpoint path:** `/files/<path:filename>`  
**HTTP method:** `GET`  
**Query parameters:** None  
**Path variables:**

| Name       | Description |
|------------|-------------|
| `filename` | Path under storage, e.g. `front_door/1739123456_abc/clip.mp4`, `events/ce_id/camera/snapshot.jpg`, `saved/events/ce_id/camera/snapshot.jpg`. |

**Request body:** None  

**Response (200):** Binary body (e.g. video/mp4, image/jpeg). `Content-Type` and filename from Flask `send_from_directory`.  
**Error (404):** "File not found" (path outside storage or missing file).

---

### 2.2 Frigate event snapshot (proxy)

**Endpoint path:** `/api/events/<event_id>/snapshot.jpg`  
**HTTP method:** `GET`  
**Query parameters:** None  
**Path variables:**

| Name       | Description |
|------------|-------------|
| `event_id` | **Frigate** event id (from Frigate/MQTT), not the buffer's `event_path` or `subdir`. |

**Request body:** None  

**Response (200):** Image bytes (JPEG stream from Frigate).  
**Errors:** 503 "Frigate URL not configured", 502 "Snapshot unavailable" (plain text).

Use this for **Frigate's** snapshot of an event. For the buffer's stored snapshot use `/files/...` (e.g. `hosted_snapshot` from event JSON).

---

### 2.3 Frigate camera latest frame (live)

**Endpoint path:** `/api/cameras/<camera_name>/latest.jpg`  
**HTTP method:** `GET`  
**Query parameters:** None  
**Path variables:**

| Name          | Description |
|---------------|-------------|
| `camera_name` | Camera name (alphanumeric, underscore, hyphen). Must be in server's allowed cameras if configured. |

**Request body:** None  

**Response (200):** Image bytes (JPEG live frame).  
**Errors:** 400 "Invalid camera name", 404 "Camera not configured", 503 "Frigate URL not configured", 502 "Live frame unavailable" (plain text).

---

### 2.4 Media URLs — how to build full URLs (Android)

- **Stored clips and snapshots**  
  Event JSON returns **paths** such as:
  - `hosted_clip`: `/files/events/1739123456_abc/summary.mp4` or `/files/front_door/1739123456_abc/clip.mp4`
  - `hosted_snapshot`: `/files/events/1739123456_abc/front_door/snapshot.jpg`
  - `hosted_clips`: `[ { "camera": "front_door", "url": "/files/events/ce_id/front_door/clip.mp4" }, ... ]`

  **Full URL** = `{baseUrl}{path}` (e.g. `http://192.168.21.50:5000/files/events/1739123456_abc/front_door/snapshot.jpg`). No query parameters. Use for:
  - Playing `.mp4` clips (ExoPlayer, etc.)
  - Loading `.jpg` snapshots for event thumbnails or detail.

- **Frigate proxy (live / Frigate snapshot)**  
  - **Live frame:** `GET {baseUrl}/api/cameras/{camera_name}/latest.jpg` — use for live camera view.
  - **Frigate snapshot:** `GET {baseUrl}/api/events/{event_id}/snapshot.jpg` — use only when you have Frigate's `event_id` and want Frigate's image; for buffer-stored snapshots use `hosted_snapshot` + `/files/...` as above.

---

## 3. Stats

### 3.1 Stats dashboard

**Endpoint path:** `/stats`  
**HTTP method:** `GET`  
**Query parameters:** None  
**Path variables:** None  
**Request body:** None  

**Response (200):**

```json
{
  "events": {
    "today": 12,
    "this_week": 84,
    "this_month": 312,
    "total_reviewed": 200,
    "total_unreviewed": 112,
    "by_camera": { "front_door": 45, "garage": 30, "events": 237 }
  },
  "storage": {
    "total_display": { "value": 2.5, "unit": "GB" },
    "by_camera": { "front_door": { "value": 0.8, "unit": "GB" }, "garage": { "value": 0.5, "unit": "GB" } },
    "breakdown": {
      "clips": { "value": 1500, "unit": "MB" },
      "snapshots": { "value": 200, "unit": "MB" },
      "descriptions": { "value": 1, "unit": "MB" }
    }
  },
  "errors": [],
  "last_cleanup": {
    "at": "2025-02-26 10:00:00",
    "deleted": 3
  },
  "most_recent": {
    "event_id": "abc123",
    "camera": "front_door",
    "url": "/player?filter=all",
    "timestamp": 1739123456.0
  },
  "system": {
    "uptime_seconds": 86400,
    "mqtt_connected": true,
    "active_events": 2,
    "retention_days": 3,
    "cleanup_interval_hours": 1,
    "storage_path": "/data/storage",
    "stats_refresh_seconds": 60
  },
  "ha_helpers": {
    "gemini_month_cost": 1.25,
    "gemini_month_tokens": 50000
  }
}
```

| Field                | Type   | Nullable | Description |
|----------------------|--------|----------|-------------|
| `events.today`       | int    | No       | Events in last 24h. |
| `events.this_week`  | int    | No       | Events in last 7 days. |
| `events.this_month`  | int    | No       | Events in last 30 days. |
| `events.total_reviewed`   | int | No       | |
| `events.total_unreviewed` | int | No       | |
| `events.by_camera`   | object | No       | Camera name → count. |
| `storage.total_display` | `{ value: number, unit: "KB"\|"MB"\|"GB" }` | No | |
| `storage.by_camera`  | object | No       | Camera → size object. |
| `storage.breakdown`  | object | No       | clips, snapshots, descriptions size. |
| `errors`             | array  | No       | Recent error strings. |
| `last_cleanup`       | object | Yes      | `at` (string), `deleted` (int). |
| `most_recent`        | object | Yes      | `event_id`, `camera`, `url`, `timestamp`. |
| `system`             | object | No       | uptime, mqtt_connected, active_events, retention_days, etc. |
| `ha_helpers`         | object | Yes      | Present if HA configured: `gemini_month_cost`, `gemini_month_tokens`. |

---

### 3.2 Status

**Endpoint path:** `/status`  
**HTTP method:** `GET`  
**Query parameters:** None  
**Path variables:** None  
**Request body:** None  

**Response (200):**

```json
{
  "online": true,
  "mqtt_connected": true,
  "uptime_seconds": 86400.5,
  "uptime": "1 day, 0:00:00",
  "started_at": "2025-02-25 10:00:00",
  "active_events": { "total_active": 5, "by_phase": { "NEW": 1, "DESCRIBED": 0, "FINALIZED": 2, "SUMMARIZED": 2 }, "by_camera": { "front_door": 2, "events": 3 } },
  "metrics": {
    "notification_queue_size": 0,
    "active_threads": 12,
    "active_consolidated_events": [
      { "id": "ce_1739123456_abc", "state": "active", "cameras": ["front_door"], "start_time": 1739123456.0 }
    ],
    "recent_errors": []
  },
  "config": {
    "retention_days": 3,
    "log_level": "INFO",
    "ffmpeg_timeout": 60,
    "summary_padding_before": 15,
    "summary_padding_after": 15
  }
}
```

| Field           | Type   | Nullable | Description |
|-----------------|--------|----------|-------------|
| `online`        | boolean| No       | Always true when endpoint responds. |
| `mqtt_connected`| boolean| No       | MQTT broker connected. |
| `uptime_seconds`| number | No       | Seconds since start. |
| `uptime`        | string | No       | Human-readable uptime. |
| `started_at`     | string | No       | Server start time. |
| `active_events`  | object | No       | `total_active` (int), `by_phase` (object), `by_camera` (object). |
| `metrics`       | object | No       | queue_size, active_threads, active_consolidated_events, recent_errors. |
| `config`        | object | No       | retention_days, log_level, ffmpeg_timeout, summary_padding_*. |

---

## 4. Daily Review

### 4.1 List report dates

**Endpoint path:** `/api/daily-review/dates`  
**HTTP method:** `GET`  
**Query parameters:** None  
**Path variables:** None  
**Request body:** None  

**Response (200):**

```json
{
  "dates": ["2025-02-26", "2025-02-25", "2025-02-24"]
}
```

| Field  | Type     | Description |
|--------|----------|-------------|
| `dates`| string[] | Sorted YYYY-MM-DD, newest first. |

---

### 4.2 Get report for today (current)

**Endpoint path:** `/api/daily-review/current`  
**HTTP method:** `GET`  
**Query parameters:** None  
**Path variables:** None  
**Request body:** None  

**Response (200):**

```json
{
  "summary": "# Daily report 2025-02-26\n\n..."
}
```

**Error (404):** `{ "error": "No report for today yet" }`

---

### 4.3 Get report by date

**Endpoint path:** `/api/daily-review/<date_str>`  
**HTTP method:** `GET`  
**Query parameters:** None  
**Path variables:**

| Name       | Description |
|------------|-------------|
| `date_str` | `YYYY-MM-DD`. |

**Request body:** None  

**Response (200):** `{ "summary": "string" }` (markdown content).  
**Errors:** 400 `{ "error": "Invalid date format" }`, 404 `{ "error": "Report not found for this date" }`.

---

### 4.4 Generate report

**Endpoint path:** `/api/daily-review/generate`  
**HTTP method:** `POST`  
**Query parameters:**

| Name   | Required | Description |
|--------|----------|-------------|
| `date` | No       | `YYYY-MM-DD`; defaults to today if omitted. |

**Path variables:** None  
**Request body:** Optional JSON: `{ "date": "YYYY-MM-DD" }`. If both query and body provide `date`, server may use query or body (document both for compatibility).

**Response (200):** `{ "success": true, "date": "2025-02-26" }`  
**Errors:** 400 invalid date format, 503 `{ "error": "Daily reporter disabled (AI not enabled)" }` or `{ "error": "Report generation failed" }`.

---

## 5. Test Pipeline

Used by the Test multi-cam page; mobile may use for "run test" or "send prompt to AI" flows.

### 5.1 Prepare test folder

**Endpoint path:** `/api/test-multi-cam/prepare`  
**HTTP method:** `GET`  
**Query parameters:**

| Name    | Required | Description |
|---------|----------|-------------|
| `subdir`| Yes      | Source event path, e.g. `events/1739123456_abc` or `saved/events/1739123456_abc`. |

**Path variables:** None  
**Request body:** None  

**Response (200):** `{ "test_run_id": "test1" }` (or test2, test3, …).  
**Errors:** 400 `{ "error": "Missing subdir" }` or `{ "error": "Invalid or missing event folder" }`, 404 invalid/missing folder.

---

### 5.2 Event data (timeline + files)

**Endpoint path:** `/api/test-multi-cam/event-data`  
**HTTP method:** `GET`  
**Query parameters:**

| Name       | Required | Description |
|------------|----------|-------------|
| `test_run`| Yes      | Test folder name: `test1`, `test2`, … (pattern `test\d+`). |

**Path variables:** None  
**Request body:** None  

**Response (200):**

```json
{
  "timeline": { "event_id": "test1", "entries": [ ] },
  "event_files": [
    { "path": "metadata.json", "url": "/files/events/test1/metadata.json" },
    { "path": "front_door/clip.mp4", "url": "/files/events/test1/front_door/clip.mp4" }
  ]
}
```

| Field         | Type     | Description |
|---------------|----------|-------------|
| `timeline`    | object   | Same shape as 1.10. |
| `event_files` | object[] | Each: `path` (relative path), `url` (path for GET /files/...). |

**Errors:** 400 `{ "error": "Invalid test_run" }`, 404 `{ "error": "Test run folder not found" }`, 500.

---

### 5.3 AI payload (prompt + image URLs)

**Endpoint path:** `/api/test-multi-cam/ai-payload`  
**HTTP method:** `GET`  
**Query parameters:** `test_run` (required, same as 5.2).  
**Path variables:** None  
**Request body:** None  

**Response (200):**

```json
{
  "prompt": "System prompt text...",
  "image_urls": [
    "/files/events/test1/ai_frame_analysis/frames/frame_001.jpg",
    "/files/events/test1/ai_frame_analysis/frames/frame_002.jpg"
  ]
}
```

| Field        | Type     | Description |
|--------------|----------|-------------|
| `prompt`     | string   | Content of system_prompt.txt. |
| `image_urls` | string[] | Paths to frame images (prepend baseUrl for full URL). |

**Errors:** 400 invalid test_run, 404 test folder or system_prompt.txt not found, 500.

---

### 5.4 Stream (SSE)

**Endpoint path:** `/api/test-multi-cam/stream`  
**HTTP method:** `GET`  
**Query parameters:**

| Name       | Required | Description |
|------------|----------|-------------|
| `test_run` | No*      | If set, run pipeline from existing test folder (post-copy). |
| `subdir`   | No*      | If set (and no test_run), run full pipeline from source folder. One of test_run or subdir required. |

**Path variables:** None  
**Request body:** None  

**Response (200):** `text/event-stream`. Each event is a line: `data: {JSON}\n\n`.  
**Event types:** `log` — `{ "type": "log", "message": "string" }`; `done` — `{ "type": "done" }`; `error` — `{ "type": "error", "message": "string" }`.  
Client should stop reading after `done` or `error`.

**Errors:** 400 missing subdir/test_run or invalid folder.

---

### 5.5 Video request (SSE)

**Endpoint path:** `/api/test-multi-cam/video-request`  
**HTTP method:** `GET`  
**Query parameters:** `test_run` (required, same as 5.2).  
**Path variables:** None  
**Request body:** None  

**Response (200):** `text/event-stream`. Same format as 5.4: `data: {"type":"log","message":"..."}\n\n`, then `data: {"type":"done"}\n\n` or `data: {"type":"error","message":"..."}\n\n`.  
Server deletes existing clips in test folder, then requests new exports from Frigate and streams log messages.

**Errors:** 400 invalid test_run, 404 test folder not found, 503 `{ "error": "FRIGATE_URL not configured" }`.

---

### 5.6 Send prompt to AI

**Endpoint path:** `/api/test-multi-cam/send`  
**HTTP method:** `POST`  
**Query parameters:** `test_run` (required).  
**Path variables:** None  
**Request body:** None  

**Response (200):** Proxy analysis result (e.g. Gemini). Example:

```json
{
  "title": "Person walking in driveway",
  "shortSummary": "Brief summary.",
  "description": "Longer description.",
  "scene": "Narrative scene.",
  "potential_threat_level": 0
}
```

Fields may vary by proxy; common: `title`, `shortSummary`, `description`, `scene`, `potential_threat_level` (int).  
**Errors:** 400 invalid test_run, 404 test folder or system_prompt.txt or no frame images, 500 loading frames, 502 `{ "error": "Proxy request failed" }`.

---

## 6. Mobile registration

### 6.1 Register FCM device token

**Endpoint path:** `/api/mobile/register`  
**HTTP method:** `POST`  
**Query parameters:** None  
**Path variables:** None  

**Request body (JSON):**

```json
{
  "token": "fcm_device_token_here"
}
```

| Field   | Type   | Required | Description |
|---------|--------|----------|-------------|
| `token` | string | Yes      | FCM device token from the mobile app. |

**Response (200):**

```json
{
  "status": "success"
}
```

**Errors:**

- 400: Missing or empty `token`. Response body: `{ "error": "Missing or empty token" }`.

**Note:** The token is persisted to `mobile_preferences.json` under the server's storage path. No server restart is required; the backend uses this token for FCM push when mobile notifications are enabled.

---

*End of contract.*
