# MAP.md — Primary Entry Point for AI Agents

Read this first; then open the Context Branch that matches your task to minimize
token use and maximize surgical accuracy.

---

## 1. Project Overview & Core Logic

**Frigate Event Buffer:** state-aware orchestrator for Frigate 0.17+ NVR.

- **Ingestion:** MQTT events, four-phase lifecycle (NEW → DESCRIBED → FINALIZED →
  SUMMARIZED); zone/exception filtering via SmartZoneFilter; consolidated events.
- **Outbound:** Ring-style HA notifications (clear_tag); rolling evidence locker
  (default 3 days); clip export/transcode-free storage; export watchdog (DELETE
  completed exports from Frigate).
- **AI (optional):** Gemini proxy: motion-aware frame extraction, smart crop,
  `analysis_result.json`; daily report from aggregated results.
- **Web:** Flask app: `/player`, `/stats-page`, `/daily-review`, REST API for
  events, clips, snapshots—embeddable in Home Assistant.

**Tech stack:** Python 3.12+; setuptools src layout (`src/frigate_buffer/`);
Flask + Gunicorn (single worker); paho-mqtt; schedule; YAML + env (Voluptuous).
State: EventStateManager, ConsolidatedEventManager; filesystem for events, clips,
timelines, daily reports. Video/AI: **PyNvVideoCodec** (gpu_decoder) only for
decode; FFmpeg h264_nvenc for compilation encode; Ultralytics YOLO; Gemini proxy
(HTTP). Testing: pytest, `pythonpath = ["src"]`.

**Video & AI pipeline (mandatory):** Zero-copy GPU only. Decode: PyNvVideoCodec
(gpu_decoder); `create_decoder(clip_path, gpu_id)`; frames → BCHW CUDA tensors.
No CPU decode fallback; no ffmpegcv. Single-threaded decode: app-wide
**GPU_LOCK** (video.py) serializes create_decoder and get_frames. Detection
sidecars: batched get_frames(4) → float32/255 → GPU resize → YOLO → bbox
scale-back; crop/resize via **crop_utils** (BCHW tensors only). NumPy/OpenCV only
at boundaries (e.g. tensor → JPEG, FFmpeg rawvideo stdin). Output:
torchvision.io.encode_jpeg; compilation streams HWC to FFmpeg h264_nvenc stdin.

**Explicit prohibitions (do not reintroduce):**

- **ffmpegcv** — Forbidden for decode/capture.
- **CPU-decoding fallbacks** — Forbidden (e.g. OpenCV VideoCapture, FFmpeg
  subprocess for decode). FFmpeg only: GIF generation, ffprobe metadata.
- **Production frame processing on NumPy in core path** — Forbidden; use
  `crop_utils` (BCHW). Legacy NumPy crop helpers removed; production uses
  crop_utils only.

---

## 2. Architectural Pattern

**Orchestrator-centric:** StateAwareOrchestrator owns MQTT routing, event/CE
handling, scheduling; delegates to managers (state, file, consolidation, zone
filter), services (lifecycle, download, notifications, timeline, AI, daily
reporter, export watchdog), Flask app. Logic in `src/frigate_buffer/`; entry:
`run_server.py` (Gunicorn -w 1 --threads 4); `main.py` → bootstrap(); wsgi.py →
application. Web under `src/frigate_buffer/web/`; create_app(orchestrator);
no business logic in server. EventQueryService reads from disk; routes call it;
no API-fetch in templates. Refs: gpu_pipeline_audit_report.md,
performance_final_verification.md.

---

## 3. Directory Structure (The Map)

Per-file purpose and dependencies live in Context Branches (see below).
Excludes: node_modules, .git, __pycache__, .pytest_cache, build artifacts.

```
frigate-event-buffer/
├── config.yaml
├── docker-compose.yaml
├── docker-compose.example.yaml
├── Dockerfile
├── pyproject.toml
├── requirements.txt
├── run_server.py
├── RULE.md
├── MAP.md
├── README.md
├── USER_GUIDE.md
├── docs/
│   ├── INSTALL.md
│   ├── MOBILE_API_CONTRACT.md
│   ├── SESSION.md
│   └── maps/ (INGESTION, PROCESSING, WEB, LIFECYCLE, NOTIFICATIONS, AI, TESTING)
├── MULTI_CAM_PLAN.md
├── DIAGNOSTIC_SIDECAR_TIMELINE_COMPILATION.md
├── gpu_pipeline_audit_report.md
├── performance_final_verification.md
├── .cursor/rules/
├── scripts/ (bench_post_download_pre_api, verify_gemini_proxy, README, scripts_readme)
├── src/frigate_buffer/
│   ├── main.py, wsgi.py, config.py, models.py, logging_utils.py, constants.py
│   ├── version.txt, orchestrator.py
│   ├── event_test/ (__init__.py, event_test_orchestrator.py)
│   ├── managers/ (file, state, consolidation, zone_filter)
│   ├── services/
│   │   ├── ai_analyzer, gemini_proxy_client, gpu_decoder, multi_clip_extractor
│   │   ├── timeline_ema, compilation_math, video, lifecycle, download
│   │   ├── notifications/ (base, dispatcher, providers/ha_mqtt, pushover)
│   │   ├── query, daily_reporter, frigate_export_watchdog, ha_storage_stats
│   │   ├── timeline, video_compilation, mqtt_handler, mqtt_client, crop_utils
│   │   ├── quick_title_service
│   │   └── report_prompt.txt, ai_analyzer_system_prompt.txt, quick_title_prompt.txt
│   └── web/
│       ├── frigate_proxy, path_helpers, report_helpers, server
│       ├── routes/ (api, daily_review, pages, proxy_routes, test_routes)
│       ├── templates/ (base, player, test_run, timeline, stats, daily_review, test.md)
│       └── static/ (purify.min.js, marked.min.js)
├── tests/ (conftest.py, test_*.py)
└── examples/
    ├── .env.example
    ├── config.example.yaml
    └── home-assistant/
```

---

## 4. Context Branches

For surgical edits, load only the branch that matches your task.

| Branch | Path | Scope |
|--------|------|--------|
| INGESTION | docs/maps/INGESTION.md | MQTT, Zone Filtering, State Management |
| PROCESSING | docs/maps/PROCESSING.md | GPU Pipeline, Video Decoding, Crop Utils |
| WEB | docs/maps/WEB.md | Flask Server, Routes, Templates, Proxy, Helpers |
| LIFECYCLE | docs/maps/LIFECYCLE.md | Lifecycle, Download, File, Query, Timeline |
| NOTIFICATIONS | docs/maps/NOTIFICATIONS.md | Dispatcher, Providers (HA, Pushover) |
| AI | docs/maps/AI.md | Analyzer, Gemini Proxy, Quick Title, Daily Reporter, Prompts |
| TESTING | docs/maps/TESTING.md | Test Suite (~10k lines), event_test (TEST button) |

---

## 5. Coding Standards & Conventions

**Ruff:** Lint and format (`ruff check src tests`, `ruff format src tests`). Line
length 88 chars (E501). Type hints on public APIs; Python 3.10+ union syntax
(str | None). Config: extend CONFIG_SCHEMA in config.py; invalid config exits 1.
Constants in constants.py; no magic numbers. Docstrings explain why (Google or
NumPy). Early returns; tests: Setup → Execute → Verify. **Token density:**
documentation targets ~12 tokens/line for agent efficiency.

**File placement:** New UI/route → web/routes/, web/templates/, web/static/. New
service/logic → services/ or managers/; register in orchestrator. New API
route → web/routes/; delegate to EventQueryService/FileManager. New config key
→ CONFIG_SCHEMA then config.get. New notification provider →
notifications/providers/; ADDING_PROVIDERS.md. Standalone script → scripts/.
Tests → tests/test_<name>.py. **Map maintenance:** When adding/removing files
or changing core flows, update MAP.md and the affected branch under docs/maps/.

---

*End of MAP.md*
