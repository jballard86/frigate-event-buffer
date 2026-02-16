# Multi-cam frame extractor: expected input and output payloads

This document describes the MQTT and Gemini proxy payloads as they would be sent and received by the multi-cam frame extractor service. Used for integration and proxy wiring.

---

## Prompt templates (placeholders)

Both **multi_cam_system_prompt.txt** and **report_prompt.txt** use `{placeholder}` labels. The scripts fill these from MQTT event data, config, and (for the report) previous day's recaps.

- **multi_cam_system_prompt.txt:** `{image_count}`, `{global_event_camera_list}`, `{first_image_number}`, `{last_image_number}`, `{current day and time}`, `{duration of the event}`, `{list of zones in global event, dont repeat zones}`, `{list of labels and sub_labels tracked in scene}` — filled in **multi_cam_frame_extracter.py** from `EventMetadataStore` (event_info: label, sub_label, entered_zones from frigate/events) and from the frame-extraction loop (image count, cameras, start/end time). The filled prompt is written to `multi_cam_frames/system_prompt_filled.txt` and used as the system message when sending to the proxy.
- **report_prompt.txt:** `{report_start_time}`, `{report_end_time}`, `{report_date_string}`, `{known_person_name}`, `{list_of_event_json_objects}` — filled in **daily_report_to_proxy.py** from the report date (yesterday), config `known_person_name`, and the list of event objects returned by `get_previous_day_recaps()` (each object: title, scene, confidence, threat_level, camera, time, context).

### Bridge: analysis_result.json

The Extractor writes the proxy response (or mock) as **analysis_result.json** inside each event's `multi_cam_frames/` folder. The Reporter scans storage for this file to build the previous day's event list. Schema:

- From proxy response: `title`, `scene`, `shortSummary`, `confidence`, `potential_threat_level`
- Added by Extractor: `event_id`, `camera`, `start_time`
- Optional: `context` (array of related event objects for same time window)

---

## 1. Inputs

### 1.1 MQTT – `frigate/events`

Used for event lifecycle (new / end) and overlap detection. Not the source of per-frame box/score; that comes from `tracked_object_update`.

**Example payload (type: end):**

```json
{
  "type": "end",
  "before": {
    "id": "1739123456.789-abcdef",
    "camera": "Doorbell",
    "label": "person",
    "start_time": 1739123456.0,
    "end_time": 1739123480.0,
    "has_clip": true,
    "has_snapshot": true
  },
  "after": {
    "id": "1739123456.789-abcdef",
    "camera": "Doorbell",
    "label": "person",
    "start_time": 1739123456.0,
    "end_time": 1739123480.0,
    "has_clip": true,
    "has_snapshot": true
  }
}
```

Relevant fields for this service: `type`, `after.id`, `after.camera`, `after.start_time`, `after.end_time`.

---

### 1.2 MQTT – `frigate/{camera}/tracked_object_update`

**Primary source of per-frame data** (frame_time, box, area, score). Topic gives camera; payload gives event id and object state.

**Example payload (type: update):**

```json
{
  "type": "update",
  "before": {
    "id": "1739123456.789-abcdef",
    "frame_time": 1739123460.5,
    "box": [0.12, 0.25, 0.45, 0.82],
    "area": 0.15,
    "score": 0.92
  },
  "after": {
    "id": "1739123456.789-abcdef",
    "frame_time": 1739123460.5,
    "box": [0.12, 0.26, 0.45, 0.82],
    "area": 0.15,
    "score": 0.91
  }
}
```

- **box**: Frigate 13 uses `[x_min, y_min, x_max, y_max]` (detect resolution). Older versions may use `[ymin, xmin, ymax, xmax]`. Normalize to `[ymin, xmin, ymax, xmax]` internally if needed for crop logic.
- **frame_time**: Unix timestamp of the frame.
- **area**: Relative or pixel area of the bounding box.
- **score**: Detection confidence (0–1).

For event end, `type` may be `"end"`; use `after` (or `before` when appropriate) to get the last frame.

---

### 1.3 Clip paths (from main app)

When integrated, clip locations follow the main app layout:

- **Single event:** `{STORAGE_PATH}/{camera}/{timestamp}_{event_id}/clip.mp4`  
  Example: `/app/storage/Doorbell/1739123456_1739123456.789-abcdef/clip.mp4`

- **Consolidated event:** `{STORAGE_PATH}/events/{ce_folder_name}/{camera}/clip.mp4`  
  Example: `/app/storage/events/1739123456_a1b2c3d4/Doorbell/clip.mp4`

Clips are produced by the Frigate Export API with `export_buffer_before` / `export_buffer_after`; the first frame of the file may be earlier than the first metadata timestamp.

---

## 2. Outputs

### 2.1 To Gemini proxy (OpenAI-compatible request)

- **Method:** POST  
- **URL:** From config (e.g. `GEMINI_PROXY_URL`), e.g. `http://REDACTED_LOCAL_IP:5050/v1/chat/completions` (exact path to be confirmed for your proxy).  
- **Headers:**  
  - `Content-Type: application/json`  
  - `Authorization: Bearer REDACTED_PROXY_KEY` (if proxy uses bearer token; confirm separately).  
- **Body:** OpenAI-format request: system prompt (loaded from `multi_cam_system_prompt.txt` for easy editing), and messages with content that can include image parts (base64-encoded frames). Format per Gemini docs for OpenAI-compatible requests; we send to the proxy, not directly to Google.

**Example request shape (conceptual):**

```json
{
  "model": "gemini-2.5-flash-lite",
  "messages": [
    {
      "role": "system",
      "content": "<contents of multi_cam_system_prompt.txt>"
    },
    {
      "role": "user",
      "content": [
        {
          "type": "image_url",
          "image_url": { "url": "data:image/jpeg;base64,<BASE64_DATA>" }
        },
        {
          "type": "text",
          "text": "Camera: Doorbell, time: 2025-02-15 14:30:00"
        }
      ]
    }
  ],
  "max_tokens": 1024,
  "temperature": 0.3
}
```

(Exact field names and structure depend on the proxy’s OpenAI-compatible API; adapt to match.)

---

### 2.2 From Gemini proxy (OpenAI-compatible response)

- **Content:** Summary or reply in the usual OpenAI shape. We only need `choices[0].message.content` (the summary text).
- **Token usage:** Handled by the proxy (rate limits, billing, HA dashboard); we do not use usage fields from the response.

**Example response shape (minimal):**

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "A person approaches the door at 14:30; the carport shows a vehicle at 14:30:05."
      },
      "finish_reason": "stop"
    }
  ]
}
```
