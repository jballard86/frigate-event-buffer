# Refactoring Prompts

## 1. Refactor StateAwareOrchestrator to Extract Event Lifecycle Management

**Profession:** Senior Software Architect

**Description:**
The `StateAwareOrchestrator` class in `frigate_buffer/orchestrator.py` has become a "God Object," handling MQTT routing, event state updates, consolidated event lifecycle (closing, notifications), and scheduling. This makes it hard to test and maintain. Your task is to extract the consolidated event lifecycle logic (specifically `_on_consolidated_event_close`, `_process_event_end`, and related notification logic) into a new service class, e.g., `EventLifecycleService`. The orchestrator should delegate to this service rather than containing the logic directly. Ensure that the new service has clear dependencies (FileManager, NotificationPublisher, etc.) injected via its constructor.

**Code Snippet:**
`frigate_buffer/orchestrator.py`
```python
    def _on_consolidated_event_close(self, ce_id: str):
        """Called when CE close timer fires. Export clips, fetch summary, send notifications."""
        with self.consolidated_manager._lock:
            ce = self.consolidated_manager._events.get(ce_id)
        if not ce or not ce.closed:
            return

        export_before = self.config.get('EXPORT_BUFFER_BEFORE', 5)
        # ... (100+ lines of logic) ...
        self.consolidated_manager.remove(ce_id)
        logger.info(f"Consolidated event {ce_id} closed and cleaned up")
```

**Tool:** Yes, create a test tool that mocks the dependencies and verifies the lifecycle service correctly handles event closure and notifications.

---

## 2. Extract Video Processing Logic from FileManager

**Profession:** Video Engineering Expert

**Description:**
The `FileManager` in `frigate_buffer/managers/file.py` handles both file I/O operations and video processing (transcoding, GIF generation). This violates the Single Responsibility Principle. Extract all FFmpeg-related logic into a dedicated `VideoService` or `Transcoder` class. This new class should handle `_transcode_clip_to_h264`, `generate_gif_from_clip`, and process management (timeout handling, SIGTERM/SIGKILL). The `FileManager` should delegate video processing tasks to this new service.

**Code Snippet:**
`frigate_buffer/managers/file.py`
```python
    def _transcode_clip_to_h264(self, event_id: str, temp_path: str, final_path: str) -> bool:
        """Transcode clip_original.mp4 to H.264 clip.mp4. Removes temp on success."""
        process = None
        try:
            command = [
                "ffmpeg", "-y",
                "-i", temp_path,
                "-c:v", "libx264",
                # ...
                final_path,
            ]
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate(timeout=self.ffmpeg_timeout)
            if process.returncode == 0:
                os.remove(temp_path)
                logger.info(f"Transcoded clip for {event_id}")
                return True
            # ...
```

**Tool:** Yes, create a tool that mocks `subprocess.Popen` to verify `VideoService` correctly executes FFmpeg commands and handles timeouts/errors.

---

## 3. Extract Business Logic from Web Server to Service Layer

**Profession:** Backend Software Engineer

**Description:**
The Flask application in `frigate_buffer/web/server.py` contains significant business logic inside route handlers, particularly for querying and processing event lists (`_get_events_from_consolidated`, `_get_events_for_camera`, `_extract_genai_entries`, etc.). This logic involves direct filesystem traversal and file parsing. Refactor this logic into a dedicated `QueryService` or `EventQueryManager`. The web handlers should delegate to this service, which should return structured data models or DTOs rather than raw file system constructs.

**Code Snippet:**
`frigate_buffer/web/server.py`
```python
    def _get_events_from_consolidated() -> list:
        """Get consolidated events from events/{ce_id}/{camera}/ structure."""
        events_dir = os.path.join(storage_path, "events")
        events_list = []
        if not os.path.isdir(events_dir):
            return events_list

        for ce_id in sorted(os.listdir(events_dir), reverse=True):
            # ... (parsing logic) ...
            parsed = _parse_summary(summary_text)
            # ...
```

**Tool:** Yes, create a tool that mocks the filesystem structure and verifies that `QueryService` correctly returns parsed event data.

---

## 4. Normalize Event Models for Notification Consistency

**Profession:** Software Design Specialist

**Description:**
The application uses multiple classes (`EventState`, `ConsolidatedEvent`, and an ad-hoc `NotifyTarget` class created dynamically in `orchestrator.py`) to represent events for notifications. This relies heavily on duck typing and `getattr` calls in `NotificationPublisher`, leading to fragile code and potential runtime errors if attributes are missing or misspelled. Define a clear `NotificationEvent` protocol or interface that all event-like objects must implement. Update `EventState`, `ConsolidatedEvent`, and any dynamic objects to conform to this interface.

**Code Snippet:**
`frigate_buffer/services/notifier.py`
```python
    def _build_message(self, event: EventState, status: str) -> str:
        """Build notification message combining status context with best available description."""
        # Fragile attribute access
        best_desc = getattr(event, 'genai_description', None) or getattr(event, 'ai_description', None)
        # ...
```

**Tool:** Yes, create a tool that attempts to send notifications with various event objects (including incomplete ones) to verify that the normalized interface prevents errors.

---

# Bug Fix Prompts

## 5. Fix Race Condition in Consolidated Event Closing

**Profession:** Concurrency Expert

**Description:**
The `StateAwareOrchestrator` in `frigate_buffer/orchestrator.py` retrieves a consolidated event (`ce`) from `ConsolidatedEventManager` using a lock, but then releases the lock and processes the event (`_on_consolidated_event_close`). During this processing, the event state (`ce.closed`, metadata) could be modified by other threads (e.g., `_process_event_end`, `_handle_review`). Refactor the `ConsolidatedEventManager` and `Orchestrator` to ensure thread-safe access to the consolidated event lifecycle. This might involve extending the lock scope, using atomic operations, or redesigning the state transition to prevent modification after closure initiation.

**Code Snippet:**
`frigate_buffer/orchestrator.py`
```python
    def _on_consolidated_event_close(self, ce_id: str):
        """Called when CE close timer fires. Export clips, fetch summary, send notifications."""
        with self.consolidated_manager._lock:
            ce = self.consolidated_manager._events.get(ce_id)
        if not ce or not ce.closed:
            return

        # Lock is released here, but 'ce' is still being used and potentially modified by other threads
```

**Tool:** Yes, create a tool that simulates concurrent access to a consolidated event during closing to verify that no race conditions occur.

---

## 6. Fix Resource Leaks in File Downloads

**Profession:** Software Reliability Engineer

**Description:**
The `FileManager.download_and_transcode_clip` method in `frigate_buffer/managers/file.py` opens HTTP response streams and file handles without ensuring they are always closed, particularly in error cases. The `requests.get(..., stream=True)` response must be closed explicitly or used within a context manager. Additionally, the `open(temp_path, 'wb')` block handles file writing, but if an exception occurs during download (e.g. network interruption), the partial file might remain or the handle might not be flushed. Refactor this method to use context managers (`with`) for all resources and ensure robust cleanup in `finally` blocks.

**Code Snippet:**
`frigate_buffer/managers/file.py`
```python
    def download_and_transcode_clip(self, event_id: str, folder_path: str) -> bool:
        # ...
        try:
            # ...
            response = requests.get(url, timeout=120, stream=True)
            response.raise_for_status()

            # Response is not closed if subsequent code fails or returns early

            bytes_downloaded = 0
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    bytes_downloaded += len(chunk)

            # ...
```

**Tool:** Yes, create a tool that mocks network failures during download and verifies that resources (file handles, network connections) are properly released.

---

# Security Prompts

## 7. Harden File Path Security

**Profession:** Security Engineer

**Description:**
The `FileManager` in `frigate_buffer/managers/file.py` constructs file paths based on inputs (camera names, event IDs) that could potentially contain path traversal characters (e.g. `../`). While `sanitize_camera_name` mitigates some risk, `create_event_folder`, `create_consolidated_event_folder`, and `ensure_consolidated_camera_folder` lack explicit checks against path traversal. Implement rigorous path validation to ensure that all constructed paths are strictly within the designated `storage_path`. Verify that `os.path.realpath` resolves to a subdirectory of `self.storage_path`.

**Code Snippet:**
`frigate_buffer/managers/file.py`
```python
    def create_event_folder(self, event_id: str, camera: str, timestamp: float) -> str:
        """Create folder for event: {camera}/{timestamp}_{event_id} (legacy)"""
        sanitized_camera = self.sanitize_camera_name(camera)
        folder_name = f"{int(timestamp)}_{event_id}"
        camera_path = os.path.join(self.storage_path, sanitized_camera)
        folder_path = os.path.join(camera_path, folder_name)

        # Missing explicit path traversal check here

        os.makedirs(folder_path, exist_ok=True)
        logger.info(f"Created folder: {sanitized_camera}/{folder_name}")
        return folder_path
```

**Tool:** Yes, create a tool that attempts to create folders with path traversal characters (e.g., `../etc/passwd`) to verify that the validation prevents access outside the storage directory.

---

# Performance Prompts

## 8. Optimize Event Listing Performance with Caching

**Profession:** Performance Engineer

**Description:**
The `_get_events_for_camera` and `_get_events_from_consolidated` functions in `frigate_buffer/web/server.py` iterate through the file system on every request to `/events`. As the number of events grows, this will become a significant bottleneck, causing slow page loads and high CPU usage. Implement an in-memory caching layer for event lists, invalidated on file system changes (e.g., new event, deletion) or on a timer. Alternatively, maintain an index of events in memory (updated by `EventStateManager` and `FileManager`) to avoid filesystem scans for read operations.

**Code Snippet:**
`frigate_buffer/web/server.py`
```python
    def _get_events_from_consolidated() -> list:
        """Get consolidated events from events/{ce_id}/{camera}/ structure."""
        events_dir = os.path.join(storage_path, "events")
        # ...
        for ce_id in sorted(os.listdir(events_dir), reverse=True):
            # ... (expensive file reads: summary.txt, metadata.json, etc.) ...
```

**Tool:** Yes, create a tool that measures the response time of the `/events` endpoint with a large number of simulated events (e.g., 1000 folders) to verify the performance improvement.

---

# Tooling Prompts

## 9. Implement Configuration Validation

**Profession:** DevOps Engineer

**Description:**
The application currently loads `config.yaml` without validating its structure or required fields (e.g., `cameras`, `network`). This can lead to runtime errors or misconfigurations (e.g., missing keys, invalid types). Implement a configuration schema validation layer using a library like `cerberus` or `voluptuous` (or custom logic). Validate the loaded configuration against the schema on startup, failing fast with informative error messages if the config is invalid.

**Code Snippet:**
`frigate_buffer/config.py`
```python
    def load_config(self) -> dict:
        """Load configuration from YAML and env vars."""
        # ... (loading logic) ...

        # Missing validation step

        return config_dict
```

**Tool:** Yes, create a tool that tests the configuration validation by providing valid and invalid configuration files.

---

## 10. Enhance Diagnostic Tools and Status Endpoint

**Profession:** Site Reliability Engineer

**Description:**
The current `/status` endpoint in `frigate_buffer/web/server.py` provides basic uptime and active event counts but lacks detailed internal state visibility. This makes it difficult to diagnose deadlocks, thread leaks, or queue backups. Enhance the `/status` endpoint or add a new `/debug` endpoint to expose critical operational metrics:
- Number of pending notifications in the queue.
- Thread count and list of active threads.
- Consolidated Event Manager lock contention (if measurable) or active timer count.
- File system usage by camera (approximate).
- Recent error buffer contents (already in `/stats`, but duplicated here for completeness).

**Code Snippet:**
`frigate_buffer/web/server.py`
```python
    @app.route('/status')
    def status():
        """Return orchestrator status for monitoring."""
        # ... (existing code) ...

        # Missing internal diagnostics:
        # - Queue size
        # - Thread info
        # - Timer count

        return jsonify({
            # ...
        })
```

**Tool:** Yes, create a tool that queries the enhanced status endpoint and asserts that all new metrics are present and within expected ranges.
