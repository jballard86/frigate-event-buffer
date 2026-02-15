# Frigate Event Buffer - Deep Dive Audit Report

**Date**: 2025-02-14
**Author**: Principal Software Architect & Security Researcher

## Executive Summary

This report identifies several new issues in security, performance, code hygiene, and error handling. While no critical command injection or XSS vulnerabilities were found, there are significant performance bottlenecks related to blocking I/O in the main event loop and potential exposure of sensitive information.

---

## 1. Security

### Medium: Potential XSS via Missing DOMPurify Fallback
**File**: `frigate_buffer/templates/player.html` (Lines ~526-538, ~304)

**Description**:
The frontend logic relies on `DOMPurify` (loaded from `/static/purify.min.js`) to sanitize markdown content (which may contain HTML) from `genai_entries` and `review_summary`. If the static file fails to load or is blocked, the code falls back to `marked.parse(b.content)` without sanitization. While `escapeHtml` is used for other fields, `marked` preserves raw HTML by default. If an attacker can inject malicious markdown/HTML into Frigate event metadata (e.g., via MQTT or API), and `DOMPurify` is missing, XSS is possible.

**Remediation**:
- Implement backend-side sanitization (e.g., using `bleach` in Python) before sending content to the template.
- OR, ensure `DOMPurify` is bundled reliably and modify the frontend logic to disable HTML rendering entirely if `DOMPurify` is undefined.

### Low: Sensitive Data Exposure in Logs
**File**: `frigate_buffer/orchestrator.py` (Line ~340)

**Description**:
The application logs the configured `FRIGATE_URL` at startup. If this URL contains credentials (e.g., `http://user:pass@host:5000`), they will be written to the logs in plain text.

**Remediation**:
- Parse the URL and strip credentials before logging.
- Use a helper function to redact sensitive information from log messages.

---

## 2. Performance

### High: Blocking I/O in Storage Stats Calculation
**File**: `frigate_buffer/managers/file.py` (Line ~438 in `compute_storage_stats`)
**Called from**: `frigate_buffer/web/server.py` (Line ~336)

**Description**:
The `/stats` endpoint calls `compute_storage_stats`, which iterates through **all** files in the storage directory and calls `os.path.getsize` on them. This is a synchronous, blocking operation. On a system with many events or slow storage (e.g., network mount), this will block the main thread and make the UI unresponsive.

**Remediation**:
- Move `compute_storage_stats` to a background thread (e.g., scheduled task running every minute).
- Store the result in a thread-safe variable (e.g., in `EventStateManager` or `Orchestrator`) and serve the cached value immediately.

---

## 3. Code Hygiene

### Medium: "God Class" Violation (FileManager)
**File**: `frigate_buffer/managers/file.py` (~490 lines)

**Description**:
The `FileManager` class has accumulated too many responsibilities. It handles:
- Path management and sanitization
- Downloading clips and snapshots (network I/O)
- Transcoding video (CPU intensive)
- Writing metadata files
- Cleaning up old events (lifecycle)
- Computing storage statistics (reporting)

**Remediation**:
- Split `FileManager` into smaller, focused classes:
  - `StorageManager`: Handles paths, file I/O, cleanup, and stats.
  - `DownloadManager`: Handles fetching content from Frigate.
  - `TranscodeService` (already exists but `FileManager` does too much coordination): Keep `VideoService` for ffmpeg calls.

### Low: Duplicate File Reads in Query Service
**File**: `frigate_buffer/services/query.py`

**Description**:
As noted in Performance, `EventQueryService` reads the same file (`notification_timeline.json`) multiple times in different methods for the same event request.

**Remediation**:
- Refactor `_extract_...` methods to accept the parsed JSON data as an argument, or create a `NotificationTimelineParser` class that parses the file once.

---

## 4. Error Handling

### Medium: Pokemon Exception Handling
**File**: `frigate_buffer/orchestrator.py` (Line ~97 in `_fetch_ha_state`)

**Description**:
The method `_fetch_ha_state` uses a broad `except Exception:` block that catches everything and passes (`pass`). This swallows potential errors like `KeyboardInterrupt` (though unlikely here) or meaningful configuration errors that should be logged.

```python
        except Exception:
            pass
```

**Remediation**:
- Catch specific exceptions (`requests.RequestException`, `ValueError`, `KeyError`).
- Log the error at `DEBUG` or `WARNING` level instead of silencing it completely.

### Medium: Broad Exception Catch in Video Service
**File**: `frigate_buffer/services/video.py` (Line ~68)

**Description**:
`generate_gif_from_clip` catches `Exception` generically. While it logs a warning, it might mask unexpected issues.

**Remediation**:
- Catch `subprocess.SubprocessError` and `OSError` specifically.

---

## Conclusion
The most critical issues to address are the performance bottlenecks in `EventQueryService` and `FileManager`, which will significantly degrade user experience as the number of events grows. Security risks are present but require specific conditions to exploit. Refactoring `FileManager` and improving error handling will improve long-term maintainability.
