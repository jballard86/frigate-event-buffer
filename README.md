# Frigate Cinematic AI & Notification Buffer

**Target:** Frigate 0.17+ users who want advanced event descriptions, tiered notifications, multi-camera GenAI review, and a dynamic cinematic event player.  Send Notifications to the home Assistant Companian, Pushover, or an Android app (Frigate Event Viewer)

This project is a state-aware, GPU-accelerated companion app for **Frigate NVR** that acts as a smart buffer between your cameras and Home Assistant. It transforms raw, multi-camera security footage into cinematic highlight reels and delivers tiered, narrative-driven notifications to eliminate alert fatigue.

---

## 🌟 Key Features

- **Cinematic Summary Videos:** Instead of static, wide-angle clips, the built-in video engine acts like a live film editor. It uses YOLO to dynamically crop and smooth-pan to track a subject across the timeline. If a person walks from the doorbell to the carport, the video automatically cuts to the best camera angle, generating a single, seamless tracking shot of the entire event.
- **Multi-Camera AI Narratives:** While Frigate's built-in GenAI typically analyzes single-camera snapshots, this orchestrator buffers the whole event. It feeds Gemini a chronological timeline of motion-aware frames across **multiple** cameras, resulting in a complete, structured story (e.g., *"Subject walked up the driveway, checked the porch, and left via the street"*).
- **Tiered "Ring-Style" Notifications:** Events are managed through a state machine (**NEW** → **DESCRIBED** → **FINALIZED** → **SUMMARIZED**) that sends a single, continuously updating alert to Home Assistant.
  - **Instant Ping:** A near-instant, simple notification the second an event starts.
  - **AI Update (Silent):** Once the event ends, the orchestrator pulls the multi-cam timeline, runs the AI analysis, and silently updates the HA notification with the full narrative description.
  - **Video Payload (Silent):** 30–60 seconds later, once the GPU finishes stitching and panning the video, it is silently attached to the notification for immediate review.
- **Daily Narrative Reports:** Gemini automatically turns the day’s compiled AI event results into a human-readable, narrative Markdown digest of activity around your property.
- **Built-in Event Viewer:** Includes a self-hosted web server that feeds a custom, interactive Event Viewer card directly into Home Assistant via iframe, displaying the narrative and the tracked video side-by-side.
- **Rolling Evidence Locker:** Automatically manages its own storage (default 3 days) without touching your core Frigate recordings.

---

## Quick start

- **Docker (recommended):** See for clone, build, config, and run. No compose required; `docker build` + `docker run` with GPU and env vars.
- **Layout:** Repo root must have `Dockerfile` and `src/`. If you only see `frigate_buffer/` at root, run `git checkout main && git pull`.

---

## Config you need

### 1. Environment (`.env` or `docker run -e`)

Copy from `examples/.env.example` and set at least:


|                  |                                                      |
| ---------------- | ---------------------------------------------------- |
| **Variable**     | **Description**                                      |
| `FRIGATE_URL`    | Frigate NVR URL, e.g. `http://192.168.1.100:5000`    |
| `MQTT_BROKER`    | MQTT broker IP or hostname                           |
| `MQTT_PORT`      | Usually `1883` (or `8883` for TLS)                   |
| `GEMINI_API_KEY` | From if using AI                                     |
| `HA_TOKEN`       | Home Assistant long-lived access token               |
| `HA_URL`         | Home Assistant URL, e.g. `http://192.168.1.100:8123` |
| `STORAGE_PATH`   | Where clips/analysis are stored (e.g. `./storage`)   |


Optional: `MQTT_USER`, `MQTT_PASSWORD`, `GEMINI_PROXY_URL` (custom proxy). The full list of supported variables is in `examples/.env.example`.

### 2. App config (`config.yaml`)

- **cameras**: List cameras and labels to process; optional `event_filters` (tracked zones, exceptions).
- **settings**: `retention_days`, `log_level`, `gemini_frames_per_hour_cap`, etc.
- **network**: `mqtt_broker`, `mqtt_port`, `frigate_url`, `buffer_ip`, `flask_port`, `storage_path`. Env vars override these.
- **Optional**: `gemini` / `gemini_proxy`, `multi_cam`, `ha` (for stats-page API usage).

Paths in `docker run` must match where you put `config.yaml` and storage (see docs/INSTALL.md).

### 3. Docker Compose (optional)

If you use Compose, copy from `docker-compose.example.yaml` (if present) to `docker-compose.yaml`, then set the same env vars and volume mounts as in docs/INSTALL.md's `docker run` example. The app does not require Compose; plain `docker run` is supported.

---

## Requirements

- **Docker** (and for GPU: NVIDIA GPU, driver, , e.g. `docker run --gpus all` with `NVIDIA_DRIVER_CAPABILITIES=compute,video,utility`).
- **Frigate** 0.17+ and **MQTT**; **Home Assistant** for notifications. Optional: **Gemini API** (or proxy) for AI analysis.

---

## Time display

All times shown in the app (event viewer, daily reports, stats dashboard, API responses, logs) use **12-hour format with AM/PM** (e.g. `02:30:45 PM`). This is consistent across the web UI and generated reports.

---

**You are solely responsible for your own API usage and costs.** This project integrates with third-party services (including but not limited to Google Gemini). Misconfiguration, bugs, or unexpected behavior (e.g. retries, high request volume, or logic errors) can lead to increased API calls and charges. **The author and contributors are not responsible for any API bills or other costs you incur.** Use at your own risk; monitor usage and set appropriate limits (e.g. `gemini_frames_per_hour_cap`, billing alerts) in your environment.

---

## Docs

- **docs/INSTALL.md** — Full install: clone, config, build, run, update, troubleshooting.
- **MAP.md** — Primary entry point; project layout and architecture (for
  contributors/agents). Branch docs under docs/maps/ give focused context.
- **examples/home-assistant/** — Sample HA notification automation and dashboard YAML.

**Development:** Lint and format with Ruff: `pip install -e ".[dev]"` then `ruff check src tests` and `ruff format src tests`. If `ruff` is not on your PATH, use `python -m ruff check src tests` and `python -m ruff format src tests`.