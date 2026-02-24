# Frigate Event Buffer

State-aware orchestrator for **Frigate NVR**: buffers events, sends Ring-style sequential notifications to Home Assistant, optional AI analysis (Gemini), and a built-in event viewer. Tracks events through NEW → DESCRIBED → FINALIZED → SUMMARIZED and keeps a configurable rolling evidence locker (default 3 days).

**Target:** Frigate 0.17+ users who want event Descriptions, HA notifications, optional GenAI review, and an event player.

---

## Quick start

- **Docker (recommended):** See **[INSTALL.md](INSTALL.md)** for clone, build, config, and run. No compose required; `docker build` + `docker run` with GPU and env vars.
- **Layout:** Repo root must have `Dockerfile` and `src/`. If you only see `frigate_buffer/` at root, run `git checkout main && git pull`.

---

## Config you need

### 1. Environment (`.env` or `docker run -e`)

Copy from `.env.example` and set at least:


| Variable         | Description                                                       |
| ---------------- | ----------------------------------------------------------------- |
| `FRIGATE_URL`    | Frigate NVR URL, e.g. `http://192.168.1.100:5000`                 |
| `MQTT_BROKER`    | MQTT broker IP or hostname                                        |
| `MQTT_PORT`      | Usually `1883` (or `8883` for TLS)                                |
| `GEMINI_API_KEY` | From [Google AI Studio](https://aistudio.google.com/) if using AI |
| `HA_TOKEN`       | Home Assistant long-lived access token                            |
| `HA_URL`         | Home Assistant URL, e.g. `http://192.168.1.100:8123`              |
| `STORAGE_ROOT`   | Where clips/analysis are stored (e.g. `./storage`)                |


Optional: `MQTT_USER`, `MQTT_PASSWORD`, `GEMINI_PROXY_URL` (custom proxy). See `.env.example` for the full list.

### 2. App config (`config.yaml`)

```bash
cp config.example.yaml config.yaml
# Edit config.yaml: cameras, retention, network, optional Gemini/multi_cam
```

- `**cameras`:** List cameras and labels to process; optional `event_filters` (tracked zones, exceptions).
- `**settings`:** `retention_days`, `log_level`, `gemini_frames_per_hour_cap`, etc.
- `**network`:** `mqtt_broker`, `mqtt_port`, `frigate_url`, `buffer_ip`, `flask_port`, `storage_path`. Env vars override these.
- **Optional:** `gemini` / `gemini_proxy`, `multi_cam`, `ha` (for stats-page API usage).

Paths in `docker run` must match where you put `config.yaml` and storage (see INSTALL.md).

### 3. Docker Compose (optional)

If you use Compose, copy from `docker-compose.example.yaml` (if present) to `docker-compose.yaml`, then set the same env vars and volume mounts as in INSTALL.md’s `docker run` example. The app does not require Compose; plain `docker run` is supported.

---

## Requirements

- **Docker** (and for GPU: NVIDIA GPU, driver, [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html), e.g. `docker run --gpus all` with `NVIDIA_DRIVER_CAPABILITIES=compute,video,utility`).
- **Frigate** 0.17+ and **MQTT**; **Home Assistant** for notifications. Optional: **Gemini API** (or proxy) for AI analysis.

---

## Disclaimer / API usage warning

**You are solely responsible for your own API usage and costs.** This project integrates with third-party services (including but not limited to Google Gemini). Misconfiguration, bugs, or unexpected behavior (e.g. retries, high request volume, or logic errors) can lead to increased API calls and charges. **The author and contributors are not responsible for any API bills or other costs you incur.** Use at your own risk; monitor usage and set appropriate limits (e.g. `gemini_frames_per_hour_cap`, billing alerts) in your environment.

---

## Docs

- **[INSTALL.md](INSTALL.md)** — Full install: clone, config, build, run, update, troubleshooting.
- **MAP.md** — Project layout and architecture (for contributors/agents).
- **examples/home-assistant/** — Sample HA notification automation and dashboard YAML.

