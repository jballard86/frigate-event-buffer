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
timelines, daily reports. Config: **`GPU_VENDOR`**, **`GPU_DEVICE_INDEX`** (legacy
**`CUDA_DEVICE_INDEX`**); YAML under **`multi_cam`** (`gpu_vendor`, `gpu_device_index`);
see **docs/INSTALL.md**. Video/AI: **PyNvVideoCodec** under **services/gpu_backends/nvidia/** (decode);
**GPU_VENDOR=intel** uses **native/intel_decode** + **services/gpu_backends/intel/** (QSV only);
**GPU_VENDOR=amd** uses **native/amd_decode** (**frigate_amd_decode**) + **services/gpu_backends/amd/** (VAAPI + optional HIP DRM→ROCm zero-copy BCHW, else CPU transfer; gpu-03);
public **gpu_decoder** shim re-exports NVIDIA; **get_gpu_backend(config)** (cached by
vendor) returns **GpuBackend**; **orchestrator** injects it into **VideoService**
and **QuickTitleService** (shared instance);
FFmpeg h264_nvenc argv from **nvidia/ffmpeg_encode**;
Ultralytics YOLO; Gemini proxy
(HTTP). Testing: pytest, `pythonpath = ["src"]`.

**Video & AI pipeline (mandatory):** Zero-copy GPU only. Decode: PyNvVideoCodec
(gpu_decoder shim); ``GpuBackend.create_decoder(clip_path, gpu_id)`` via
``get_gpu_backend(config)`` (VideoService injected; multi_clip_extractor and
video_compilation resolve per call); frames → BCHW CUDA tensors.
No CPU decode fallback; no ffmpegcv. Single-threaded decode: app-wide
**GPU_LOCK** (**services/gpu_backends/lock.py**; re-used from video) serializes
create_decoder and get_frames. Detection
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

**Intel note:** `GPU_VENDOR=intel` uses **QSV-only** decode via `frigate_intel_decode` (no CPU libavcodec fallback). True
DRM PRIME → XPU zero-copy is an experimental path gated by `INTEL_DECODE_XPU_ZEROCOPY` builds and
driver support for mapping hw frames to DRM PRIME.
When enabled, the native session attempts DRM PRIME mapping per frame and uses Level Zero DMA-BUF
import + an on-device NV12→RGB kernel to return `torch::kXPU` BCHW uint8 without CPU staging; env
CPU staging is not used for Intel HW frames when built with `INTEL_DECODE_XPU_ZEROCOPY` (fail closed).

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
├── config.example.yaml
├── .env.example
├── docker-compose.yaml
├── docker-compose.intel.example.yml (GPU_VENDOR=intel + DRI; see INSTALL)
├── docker-compose.rocm.example.yml (GPU_VENDOR=amd + /dev/kfd + DRI; see INSTALL)
├── docker-compose.example.yaml
├── Dockerfile
├── Dockerfile.intel (multi-stage: build frigate_intel_decode; no compiler in runtime)
├── Dockerfile.rocm (multi-stage: frigate_amd_decode; rocm/pytorch base; gpu-03)
├── pyproject.toml
├── requirements.txt
├── requirements-intel.txt (no PyNvVideoCodec; used by Dockerfile.intel)
├── requirements-rocm.txt (no PyNvVideoCodec; ROCm torch index; gpu-03 AMD)
├── run_server.py
├── RULE.md
├── MAP.md
├── README.md
├── USER_GUIDE.md
├── docs/
│   ├── FOLDER_STRUCTURE.md
│   ├── INSTALL.md
│   ├── MOBILE_API_CONTRACT.md
│   ├── SESSION.md
│   ├── Multi_GPU_Support_Integration_Plan/ (gpu-00 primary, gpu-01..04, README)
│   └── maps/ (INGESTION, PROCESSING, WEB, LIFECYCLE, NOTIFICATIONS, AI, TESTING)
├── DIAGNOSTIC_SIDECAR_TIMELINE_COMPILATION.md
├── gpu_pipeline_audit_report.md
├── performance_final_verification.md
├── .cursor/rules/
├── .github/workflows/ci.yml (Ruff, pytest, docker build Dockerfile.intel + smoke --strict)
├── .github/workflows/intel_arc_smoke.yml (manual; self-hosted intel-arc + DRI)
├── .github/workflows/rocm_docker_build.yml (manual; Dockerfile.rocm; large base image)
├── .github/workflows/amd_rocm_smoke.yml (manual; self-hosted amd-rocm + kfd/DRI)
├── scripts/ (build_*_decode.sh, docker_entrypoint_*.sh, smoke_* paths, run_intel_arc_docker_smoke.sh, run_amd_rocm_docker_smoke.sh)
├── native/intel_decode/ (gpu-02: QSV-only FFmpeg + libtorch; GPU_VENDOR=intel)
├── native/amd_decode/ (gpu-03: VAAPI/SW FFmpeg + libtorch; GPU_VENDOR=amd)
├── src/frigate_buffer/
│   ├── main.py, wsgi.py, config.py, models.py, logging_utils.py, constants.py
│   ├── version.txt, orchestrator.py
│   ├── event_test/
│   ├── managers/ (file, state, consolidation, zone_filter, preferences, snooze)
│   ├── services/ (see docs/maps/; gpu_backends/ nvidia, intel, amd decode+ffmpeg)
│   └── web/ (server, routes, templates, static; path_helpers, report_helpers, frigate_proxy)
├── tests/ (test_*ffmpeg.py, test_amd_decode_spike, test_amd_decoder mock, test_smoke_amd_rocm_path.py)
└── examples/

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

For per-file purpose and dependencies, see the relevant branch in **docs/maps/**.

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

## 6. AI Agent Directives (Rules & Conventions)

### Zero-copy GPU pipeline (mandatory)

- **Do not** add or use **ffmpegcv** for video decode or capture.
- **Do not** add **CPU-decoding fallbacks** (e.g. OpenCV VideoCapture or FFmpeg subprocess for decode) for the main pipeline; PyNvVideoCodec (gpu_decoder) is the only decode path. FFmpeg is allowed only for GIF generation and ffprobe metadata.
- **Production frame crops/resize** must use **`crop_utils`** with **BCHW tensors**. Do not add new NumPy/OpenCV-based crop or resize logic in the core frame path; use the existing tensor helpers in `crop_utils.py`. Legacy NumPy crop helpers were removed from ai_analyzer; use crop_utils for all production crops.

### File placement rules

| Need | Location |
|------|----------|
| New **UI component / page** | Add route in the appropriate blueprint under `src/frigate_buffer/web/routes/` (e.g. pages.py); template in `src/frigate_buffer/web/templates/`; static assets in `src/frigate_buffer/web/static/`. |
| New **user-defined test** (Test page) | See **`src/frigate_buffer/web/templates/test.md`** for sidebar button placement (nested under Test, above Reset Test), display options (Log, new horizontal bar, or both), and patterns. |
| New **business logic / service** | `src/frigate_buffer/services/` (or `managers/` if it is state/aggregation). Register and call from `orchestrator.py` (or from an existing service) as appropriate. |
| New **utility function** (generic, no I/O) | `src/frigate_buffer/services/` (e.g. `crop_utils.py`) or a new module under `services/` if it fits a clear domain. |
| New **API route** (REST) | Add in the appropriate blueprint under `src/frigate_buffer/web/routes/` (e.g. api.py or daily_review.py); use EventQueryService or FileManager for data; never put business logic in route handlers beyond delegation. |
| New **config key** | Add to **CONFIG_SCHEMA** in `src/frigate_buffer/config.py` first; then add flat key in config merge; use in code via `config.get('KEY', default)`. |
| New **notification provider** | Add provider under `src/frigate_buffer/services/notifications/providers/`; implement `BaseNotificationProvider`; register in config and `orchestrator._create_notifier()`. **See `services/notifications/ADDING_PROVIDERS.md`** for the provider contract, flow diagram, and mock examples. |
| New **standalone script** | `scripts/` at repo root. |
| **Tests** | `tests/test_<module_or_feature>.py`; mirror structure under `src/frigate_buffer/` where it helps. |

### Naming conventions

| Element | Convention | Example |
|---------|-------------|---------|
| **Python modules** | `snake_case.py` | `ai_analyzer.py`, `multi_clip_extractor.py` |
| **Classes** | PascalCase | `StateAwareOrchestrator`, `EventLifecycleService` |
| **Functions / methods** | snake_case | `_on_mqtt_message`, `resolve_clip_in_folder` |
| **Constants / config keys** | UPPER_SNAKE_CASE | `STORAGE_PATH`, `MAX_EVENT_LENGTH_SECONDS` |
| **Templates** | lowercase with underscores | `player.html`, `daily_review.html` |
| **Test files** | `test_<name>.py` | `test_ai_analyzer.py`, `test_lifecycle_service.py` |

### Coding standards

- **Ruff (lint & format):** The project uses **Ruff** for linting and formatting (`pyproject.toml`). When **creating or editing** any file in `src/` or `tests/`, follow Ruff’s rules: run `ruff check src tests` and `ruff format src tests` (or `python -m ruff check/format` if ruff is not on PATH). Line length is 88 characters (E501); fix or wrap long lines. Files listed in `[tool.ruff.lint.per-file-ignores]` are temporarily exempt from E501 until fixed—remove a file from that list once it complies. Do not introduce new lint/format violations.
- **Type hints:** Use type hints on all public function signatures and important internal APIs. Prefer Python 3.10+ syntax (e.g. `str | None`).
- **Config:** Always extend **CONFIG_SCHEMA** in `config.py` for new options; invalid config must exit with code 1.
- **State / side effects:** Core state lives in EventStateManager and ConsolidatedEventManager; services and FileManager are stateless or hold minimal caches (e.g. EventQueryService TTL cache).
- **Error handling:** Validate paths with FileManager (realpath/commonpath); log and handle Frigate/HTTP errors; do not crash the orchestrator on individual event failures.
- **Comments / docstrings:** Docstrings should explain *why* for non-obvious logic; use a consistent style (e.g. Google or NumPy). Prefer early returns over deep nesting.
- **Tests:** All new or changed logic in `src/` must have corresponding tests in `tests/`; keep tests simple (Setup → Execute → Verify).
- **Map maintenance:** When you add/remove files, change core flows, or rename important components, **update MAP.md and the affected branch under docs/maps/** so the map ecosystem stays the single source of truth.
- **Constants:** Hard-coded values go in constants.py; no magic numbers in code.
- **Time display:** All user-facing timestamps (UI, reports, summaries, stats dashboard, API responses, logs) use **12-hour format with AM/PM**. Python: use `DISPLAY_DATETIME_FORMAT` and `DISPLAY_TIME_ONLY_FORMAT` from `constants.py`. JavaScript: use `toLocaleTimeString(undefined, { hour12: true, ... })`. Internal/ISO formats (e.g. timeline entry `ts`) remain ISO 8601 where parsing is required.

---

*End of MAP.md*
