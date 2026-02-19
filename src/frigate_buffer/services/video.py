import json
import os
import shutil
import subprocess
import threading
import logging
import time
from typing import Any

import ffmpegcv

logger = logging.getLogger('frigate-buffer')

# Truncation length for NVENC probe stderr in failure logs (general case).
STDERR_TRUNCATE = 1600
# Truncation length for ffmpeg -encoders output in NVENC-failure warning (stderr snippet).
FFMPEG_ENCODERS_SNIPPET_LEN = 500
# Extended stderr length when returncode is 234 (or other probe failures) for buffer/surface/AVERROR diagnosis.
STDERR_TRUNCATE_234 = 4000

# Module-level cache set by run_nvenc_preflight_probe() on main thread; workers skip subprocess when set.
_nvenc_preflight_result: bool | None = None
# Probe resolution used by _nvenc_probe_cmd(); set by run_nvenc_preflight_probe(config) or defaults.
_nvenc_probe_width: int = 1280
_nvenc_probe_height: int = 720

# Common errno (Linux) for mapping exit codes that may be -errno & 0xFF.
_ERRNO_NAMES: dict[int, str] = {
    1: "EPERM (Operation not permitted)",
    16: "EBUSY (Device or resource busy)",
    22: "EINVAL (Invalid argument)",
    12: "ENOMEM (Out of memory)",
}


def decode_nvenc_returncode(returncode: int) -> list[str]:
    """
    Map a process returncode (e.g. 234) to possible signal/errno/AVERROR interpretations.

    Helps diagnose NVENC probe failures: 234 is often a masked negative (e.g. 256-22 for
    SIGPIPE or -22 for EINVAL/SIGPIPE). Also checks 128+signal (SIGILL=4, SIGSEGV=11) and
    common errno when exit code is (-errno) & 0xFF.
    """
    interpretations: list[str] = []
    if returncode == 0:
        return interpretations
    # 128 + signal_number (common shell convention when process is killed by signal)
    if 128 <= returncode <= 255:
        sig = returncode - 128
        if sig == 4:
            interpretations.append("128+4: process killed by SIGILL (Illegal instruction)")
        elif sig == 11:
            interpretations.append("128+11: process killed by SIGSEGV (Segmentation fault)")
        elif sig == 22:
            interpretations.append("128+22: process killed by SIGPIPE (Broken pipe)")
    # 256 - signal (alternative encoding: 234 = 256-22 = SIGPIPE)
    if 0 <= returncode <= 255:
        sig_from_256 = 256 - returncode
        if 0 < sig_from_256 <= 31:
            interpretations.append(
                f"256-{sig_from_256}={returncode}: possible signal {sig_from_256} (e.g. SIGPIPE=22)"
            )
    # -errno & 0xFF (e.g. 234 = (-22) & 0xFF → EINVAL or signal 22)
    if returncode >= 256 - 22:  # negative errno in 8-bit
        neg = returncode - 256
        errno_name = _ERRNO_NAMES.get(-neg, None)
        if errno_name:
            interpretations.append(f"Possible -errno: {errno_name}")
    # 234-specific hints
    if returncode == 234:
        interpretations.append(
            "234: often NVENC/FFmpeg hardware-accel failure or AVERROR_EXTERNAL; check stderr for "
            "buffer/surface/CUDA/NVENC messages."
        )
    return interpretations


def _nvenc_probe_cmd() -> list[str]:
    """Return the exact ffmpeg command used for NVENC probe (shared by preflight and in-process probe).

    Uses module-level _nvenc_probe_width and _nvenc_probe_height (set by run_nvenc_preflight_probe
    from config, or 1280x720 default) so the probe matches transcode crop size and is safe for NVENC.
    """
    size = f"{_nvenc_probe_width}x{_nvenc_probe_height}"
    return [
        "ffmpeg", "-y", "-f", "lavfi", "-i", f"nullsrc=s={size}:d=0.1",
        "-c:v", "h264_nvenc", "-frames:v", "1", "-f", "null", "-",
    ]


def _log_nvenc_env_and_warn() -> None:
    """Log NVIDIA-related env vars (DEBUG) and warn if capabilities lack 'video'."""
    nv_visible = os.environ.get("NVIDIA_VISIBLE_DEVICES", "")
    nv_caps = os.environ.get("NVIDIA_DRIVER_CAPABILITIES", "")
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    logger.debug(
        "NVENC probe env: NVIDIA_VISIBLE_DEVICES=%r NVIDIA_DRIVER_CAPABILITIES=%r LD_LIBRARY_PATH=%s",
        nv_visible or "(unset)",
        nv_caps or "(unset)",
        (ld_path[:200] + "..." if len(ld_path) > 200 else ld_path) or "(unset)",
    )
    if nv_caps and "video" not in nv_caps.lower():
        logger.warning(
            "NVIDIA_DRIVER_CAPABILITIES does not include 'video'; NVENC may fail. See BUILD_NVENC.md."
        )


def _log_probe_failure_verbose(returncode: int, stderr: str, cmd: list[str]) -> None:
    """Log full command, extended stderr for 234, and decode_nvenc_returncode interpretations."""
    cmd_str = " ".join(cmd)
    logger.warning("NVENC probe command: %s", cmd_str)
    truncate = STDERR_TRUNCATE_234 if returncode == 234 else STDERR_TRUNCATE
    stderr_snippet = stderr[:truncate] + ("..." if len(stderr) > truncate else "")
    logger.warning("NVENC probe stderr (returncode=%s): %s", returncode, stderr_snippet)
    for interp in decode_nvenc_returncode(returncode):
        logger.info("NVENC returncode interpretation: %s", interp)


def run_nvenc_preflight_probe(config: dict | None = None) -> None:
    """
    Run NVENC probe on the main thread at startup and cache the result.

    Establishes CUDA/NVENC context on the main thread before any worker threads
    run, avoiding returncode 234 when the first GPU use would otherwise happen
    in a ThreadPoolExecutor worker. Workers then read the module-level cache
    and skip the subprocess entirely.

    If config is provided, NVENC_PROBE_WIDTH and NVENC_PROBE_HEIGHT are used
    for the probe resolution (default 1280x720); otherwise module defaults apply.
    """
    global _nvenc_preflight_result, _nvenc_probe_width, _nvenc_probe_height
    if _nvenc_preflight_result is not None:
        return
    if config is not None:
        _nvenc_probe_width = int(config.get("NVENC_PROBE_WIDTH", 1280))
        _nvenc_probe_height = int(config.get("NVENC_PROBE_HEIGHT", 720))
    _log_nvenc_env_and_warn()
    cmd = _nvenc_probe_cmd()
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=15,
        )
        stderr = (proc.stderr or b"").decode("utf-8", errors="replace")
        failed = (
            proc.returncode != 0
            or "No capable devices found" in stderr
            or "cannot load" in stderr.lower()
        )
        if failed:
            _nvenc_preflight_result = False
            _log_probe_failure_verbose(proc.returncode, stderr, cmd)
            logger.info(
                "NVENC preflight: unavailable; transcodes will use CPU (libx264). "
                "If startup showed NVENC success, see BUILD_NVENC.md and check verbose logs above."
            )
        else:
            _nvenc_preflight_result = True
            logger.info("NVENC preflight: success — GPU transcode enabled (main-thread init).")
    except Exception as e:
        _nvenc_preflight_result = False
        logger.warning("NVENC preflight: exception — %s; transcodes will use CPU (libx264).", e)


def log_gpu_status() -> None:
    """
    At startup: log GPU and NVENC diagnostic info to help debug transcode failures.

    Runs nvidia-smi, checks libnvidia-encode.so presence, and ffmpeg encoder list.
    When nvidia-smi works but NVENC still fails later, libnvidia-encode often indicates
    a driver/library mismatch (not mounted in container).
    """
    # nvidia-smi
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            proc = subprocess.run(
                [nvidia_smi, "--query-gpu=count", "--format=csv,noheader"],
                capture_output=True,
                timeout=5,
            )
            count_str = (proc.stdout or b"").decode("utf-8", errors="replace").strip()
            gpu_count = count_str.split("\n")[0] if count_str else "?"
            proc2 = subprocess.run(
                [nvidia_smi, "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True,
                timeout=5,
            )
            driver = (proc2.stdout or b"").decode("utf-8", errors="replace").strip().split("\n")[0] or "?"
            logger.info("GPU status: nvidia-smi OK, GPUs=%s, driver=%s", gpu_count, driver)
        except (subprocess.TimeoutExpired, OSError) as e:
            logger.warning("nvidia-smi found but failed: %s; container may not have GPU access.", e)
    else:
        logger.info("nvidia-smi not found or failed; container may not have GPU access.")

    # libnvidia-encode.so check (driver/library mismatch when GPU visible but NVENC fails)
    if nvidia_smi:
        found = False
        try:
            proc = subprocess.run(
                ["ldconfig", "-p"],
                capture_output=True,
                timeout=5,
            )
            out = (proc.stdout or b"").decode("utf-8", errors="replace")
            if "libnvidia-encode" in out.lower():
                found = True
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            pass
        if not found:
            try:
                proc = subprocess.run(
                    ["find", "/usr", "-name", "libnvidia-encode*"],
                    capture_output=True,
                    timeout=5,
                    stderr=subprocess.DEVNULL,
                )
                if proc.returncode == 0 and proc.stdout:
                    found = bool(proc.stdout.strip())
            except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
                pass
            if not found:
                logger.warning(
                    "libnvidia-encode.so not found in container; NVENC will fail despite GPU visibility. "
                    "Ensure NVIDIA_DRIVER_CAPABILITIES=compute,video,utility and NVIDIA Container Toolkit is configured."
                )

    # ffmpeg encoders (retry once after delay if first run misses NVENC — startup race on some hosts)
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        try:
            proc = subprocess.run(
                [ffmpeg_path, "-encoders"],
                capture_output=True,
                timeout=10,
            )
            stderr_raw = proc.stderr or b""
            stdout_raw = proc.stdout or b""
            # Search both streams; some builds print encoder list to stdout.
            encoders = (stderr_raw.decode("utf-8", errors="replace") + "\n" +
                        stdout_raw.decode("utf-8", errors="replace"))
            has_nvenc = "h264_nvenc" in encoders or "hevc_nvenc" in encoders
            if not has_nvenc:
                time.sleep(3)
                proc = subprocess.run(
                    [ffmpeg_path, "-encoders"],
                    capture_output=True,
                    timeout=10,
                )
                stderr_raw = proc.stderr or b""
                stdout_raw = proc.stdout or b""
                encoders = (stderr_raw.decode("utf-8", errors="replace") + "\n" +
                            stdout_raw.decode("utf-8", errors="replace"))
                has_nvenc = "h264_nvenc" in encoders or "hevc_nvenc" in encoders
            if has_nvenc:
                logger.info(
                    "FFmpeg NVENC: success — h264_nvenc/hevc_nvenc available; GPU transcode enabled."
                )
            else:
                logger.warning(
                    "FFmpeg NVENC: failure — FFmpeg does not report NVENC encoders; GPU transcode unavailable."
                )
                logger.warning(
                    "To fix: rebuild the image from this repo (docker build -t frigate-buffer:latest .) and "
                    "run the container with GPU access (--gpus all and NVIDIA_* env vars). See BUILD_NVENC.md."
                )
                # Truncated stderr often contains the real reason (e.g. "Cannot load libnvidia-encode.so.1").
                stderr_str = stderr_raw.decode("utf-8", errors="replace")
                if stderr_str:
                    snippet = (
                        stderr_str[-FFMPEG_ENCODERS_SNIPPET_LEN:]
                        if len(stderr_str) > FFMPEG_ENCODERS_SNIPPET_LEN
                        else stderr_str
                    )
                    logger.warning(
                        "FFmpeg -encoders stderr snippet (last %s chars): %s",
                        len(snippet),
                        snippet.strip() or "(empty)",
                    )
                # Full output at DEBUG for support without re-running.
                logger.debug(
                    "ffmpeg -encoders full output (stderr): %s",
                    stderr_str.strip() if stderr_str else "(none)",
                )
                logger.debug(
                    "ffmpeg -encoders full output (stdout): %s",
                    stdout_raw.decode("utf-8", errors="replace").strip() if stdout_raw else "(none)",
                )
        except (subprocess.TimeoutExpired, OSError) as e:
            logger.warning(
                "FFmpeg NVENC: could not check encoders — %s; assuming GPU transcode unavailable.", e
            )
    else:
        logger.warning("FFmpeg NVENC: ffmpeg not found in PATH; GPU transcode unavailable.")


def ensure_detection_model_ready(config: dict) -> bool:
    """
    At startup: check if the multi-cam detection model is downloaded; download if not; log result.
    Call after config is loaded. Returns True if model is ready, False if skipped or failed.
    """
    model_name = (config.get("DETECTION_MODEL") or "").strip()
    if not model_name:
        logger.info("Multi-cam detection model not configured (DETECTION_MODEL empty), skipping preload")
        return False
    try:
        existed = os.path.isfile(model_name)
        from ultralytics import YOLO
        model = YOLO(model_name)
        ckpt = getattr(model, "ckpt_path", None)
        path_str = f" at {ckpt}" if (ckpt and isinstance(ckpt, str) and os.path.isfile(ckpt)) else ""
        if existed or (ckpt and os.path.isfile(ckpt)):
            logger.info("Multi-cam detection model ready: %s (already downloaded)%s", model_name, path_str)
        else:
            logger.info("Multi-cam detection model ready: %s (downloaded)%s", model_name, path_str)
        return True
    except Exception as e:
        logger.warning("Multi-cam detection model preload failed for %s: %s", model_name, e)
        return False


# Lazy import so ultralytics is only loaded when writing detection sidecar
def _run_detection_on_frame(
    model: Any,
    frame: Any,
    device: str | None,
    names: dict[int, str],
) -> list[dict[str, Any]]:
    """Run YOLO on one frame; return list of {label, area} for sidecar."""
    detections: list[dict[str, Any]] = []
    try:
        results = model(frame, device=device, verbose=False)
        if not results:
            return detections
        for r in results:
            if r.boxes is None:
                continue
            xyxy = r.boxes.xyxy
            cls_ids = r.boxes.cls
            if xyxy is None:
                continue
            # Handle tensor/numpy: convert to list of floats
            try:
                xyxy = xyxy.cpu().numpy() if hasattr(xyxy, "cpu") else xyxy
            except Exception:
                pass
            try:
                cls_ids = cls_ids.cpu().numpy() if hasattr(cls_ids, "cpu") else cls_ids
            except Exception:
                pass
            n = min(len(xyxy), len(cls_ids)) if cls_ids is not None else len(xyxy)
            for i in range(n):
                if len(xyxy[i]) < 4:
                    continue
                x1, y1, x2, y2 = float(xyxy[i][0]), float(xyxy[i][1]), float(xyxy[i][2]), float(xyxy[i][3])
                area = int((x2 - x1) * (y2 - y1))
                cls_id = int(cls_ids[i]) if cls_ids is not None else 0
                label = names.get(cls_id, f"class_{cls_id}")
                detections.append({"label": label, "area": area})
    except Exception as e:
        logger.debug("Detection on frame failed: %s", e)
    return detections


class VideoService:
    """Handles video processing tasks like transcoding and GIF generation."""

    DEFAULT_FFMPEG_TIMEOUT = 60

    # Serializes NVENC probe so only one thread runs it (avoids concurrent probes and returncode 234).
    _nvenc_probe_lock = threading.Lock()

    def __init__(self, ffmpeg_timeout: int = DEFAULT_FFMPEG_TIMEOUT):
        self.ffmpeg_timeout = ffmpeg_timeout
        self._nvenc_available: bool | None = None  # None = not yet probed
        logger.debug(f"VideoService initialized with FFmpeg timeout: {ffmpeg_timeout}s")

    def _probe_nvenc(self) -> bool:
        """
        Run a minimal NVENC encode to detect if GPU encoding is available.

        If run_nvenc_preflight_probe() was called at startup (main thread), uses the
        cached result and does not run a subprocess. Otherwise serialized with a lock;
        logs once. Returns True if NVENC is usable.
        """
        if _nvenc_preflight_result is not None:
            self._nvenc_available = _nvenc_preflight_result
            return self._nvenc_available
        with self._nvenc_probe_lock:
            if self._nvenc_available is not None:
                return self._nvenc_available
            try:
                _log_nvenc_env_and_warn()
                cmd = _nvenc_probe_cmd()
                proc = subprocess.run(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    timeout=15,
                )
                stderr = (proc.stderr or b"").decode("utf-8", errors="replace")
                failed = (
                    proc.returncode != 0
                    or "No capable devices found" in stderr
                    or "cannot load" in stderr.lower()
                )
                if failed and proc.returncode != 0:
                    time.sleep(2.5)
                    proc = subprocess.run(
                        cmd,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                        timeout=15,
                    )
                    stderr = (proc.stderr or b"").decode("utf-8", errors="replace")
                    failed = (
                        proc.returncode != 0
                        or "No capable devices found" in stderr
                        or "cannot load" in stderr.lower()
                    )
                if failed:
                    self._nvenc_available = False
                    _log_probe_failure_verbose(proc.returncode, stderr, cmd)
                    logger.info(
                        "NVENC probe: unavailable; transcodes will use CPU (libx264). "
                        "See BUILD_NVENC.md; if 234 persists after preflight, check verbose logs above."
                    )
                    return False
                self._nvenc_available = True
                logger.info("NVENC probe: success — GPU transcode enabled for this run.")
                return True
            except Exception as e:
                self._nvenc_available = False
                logger.warning("NVENC probe: exception — %s; transcodes will use CPU (libx264).", e)
                return False

    def transcode_clip_to_h264(
        self,
        event_id: str,
        temp_path: str,
        final_path: str,
        detection_sidecar_path: str | None = None,
        detection_model: str | None = None,
        detection_device: str | None = None,
    ) -> tuple[bool, str]:
        """Transcode clip_original.mp4 to H.264 clip.mp4 (NVDEC decode, NVENC encode when GPU available).
        Removes temp on success. Falls back to libx264 if GPU path fails.
        Returns (success, backend_string) where backend_string is 'GPU' or 'CPU: <reason>' for SSE logging."""
        if not self._probe_nvenc():
            ok, _ = self._transcode_clip_libx264(
                event_id, temp_path, final_path,
                detection_sidecar_path=detection_sidecar_path,
                detection_model=detection_model,
                detection_device=detection_device,
                cpu_reason="NVENC unavailable",
            )
            return (ok, "CPU: NVENC unavailable")
        try:
            ok = self._transcode_clip_nvenc(
                event_id, temp_path, final_path,
                detection_sidecar_path=detection_sidecar_path,
                detection_model=detection_model,
                detection_device=detection_device,
            )
            if ok:
                return (True, "GPU")
        except Exception as e:
            logger.warning(
                "GPU transcode failed for event %s: %s: %s. "
                "Falling back to CPU (libx264).",
                event_id, type(e).__name__, e,
            )
        ok, _ = self._transcode_clip_libx264(
            event_id, temp_path, final_path,
            detection_sidecar_path=detection_sidecar_path,
            detection_model=detection_model,
            detection_device=detection_device,
            cpu_reason="GPU transcode failed, fallback",
        )
        return (ok, "CPU: GPU transcode failed, fallback")

    def _transcode_clip_nvenc(
        self,
        event_id: str,
        temp_path: str,
        final_path: str,
        detection_sidecar_path: str | None = None,
        detection_model: str | None = None,
        detection_device: str | None = None,
    ) -> bool:
        """Decode with VideoCaptureNV (NVDEC), encode with VideoWriterNV (h264_nvenc), mux audio via ffmpeg.
        Optionally run ultralytics on each frame and write detection sidecar (same decode pass).
        May raise BrokenPipeError if the encoder subprocess exits early (e.g. NVENC session limit or GPU busy)."""
        cap = ffmpegcv.VideoCaptureNV(temp_path)
        if not cap.isOpened():
            raise RuntimeError("VideoCaptureNV could not open input")
        try:
            fps = getattr(cap, "fps", None) or 30.0
            ret, first_frame = cap.read()
            if not ret or first_frame is None:
                raise RuntimeError("No frames in input")
            dirname = os.path.dirname(final_path)
            video_only_path = os.path.join(dirname, "clip_nvenc_tmp.mp4")

            yolo_model = None
            yolo_names: dict[int, str] = {}
            device = (detection_device or "").strip() or None
            if detection_sidecar_path and detection_model:
                try:
                    from ultralytics import YOLO
                    yolo_model = YOLO(detection_model)
                    yolo_names = yolo_model.names or {}
                except Exception as e:
                    logger.warning("Could not load YOLO for detection sidecar: %s", e)

            sidecar_entries: list[dict[str, Any]] = []
            frame_idx = 0

            try:
                vidout = ffmpegcv.VideoWriterNV(
                    video_only_path,
                    "h264_nvenc",
                    fps,
                    pix_fmt="bgr24",
                )
                try:
                    vidout.write(first_frame)
                    if yolo_model is not None:
                        det = _run_detection_on_frame(yolo_model, first_frame, device or "", yolo_names)
                        sidecar_entries.append({"timestamp_sec": frame_idx / fps, "detections": det})
                    frame_idx += 1
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        vidout.write(frame)
                        if yolo_model is not None:
                            det = _run_detection_on_frame(yolo_model, frame, device or "", yolo_names)
                            sidecar_entries.append({"timestamp_sec": frame_idx / fps, "detections": det})
                        frame_idx += 1
                finally:
                    vidout.release()
            except Exception:
                if os.path.exists(video_only_path):
                    try:
                        os.remove(video_only_path)
                    except OSError:
                        pass
                raise

            if detection_sidecar_path and sidecar_entries:
                try:
                    with open(detection_sidecar_path, "w", encoding="utf-8") as f:
                        json.dump(sidecar_entries, f, indent=None, separators=(",", ":"))
                    logger.debug("Wrote detection sidecar %s (%d frames)", detection_sidecar_path, len(sidecar_entries))
                except OSError as e:
                    logger.warning("Failed to write detection sidecar %s: %s", detection_sidecar_path, e)
        finally:
            cap.release()

        # Mux video + audio from original into final_path
        mux_cmd = [
            "ffmpeg", "-y",
            "-i", video_only_path,
            "-i", temp_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0?",
            "-movflags", "+faststart",
            final_path,
        ]
        try:
            proc = subprocess.run(
                mux_cmd,
                capture_output=True,
                timeout=self.ffmpeg_timeout,
            )
            if proc.returncode != 0:
                logger.warning(f"Mux failed for {event_id}: {proc.stderr.decode()[:500]}")
                if os.path.exists(video_only_path):
                    os.remove(video_only_path)
                return False
            if os.path.exists(video_only_path):
                os.remove(video_only_path)
            os.remove(temp_path)
            logger.info(f"Transcoded clip for {event_id} (NVENC)")
            return True
        except Exception as e:
            logger.warning(f"Mux failed for {event_id}: {e}")
            if os.path.exists(video_only_path):
                try:
                    os.remove(video_only_path)
                except OSError:
                    pass
            return False

    def _transcode_clip_libx264(
        self,
        event_id: str,
        temp_path: str,
        final_path: str,
        detection_sidecar_path: str | None = None,
        detection_model: str | None = None,
        detection_device: str | None = None,
        cpu_reason: str = "libx264",
    ) -> tuple[bool, str]:
        """Transcode using ffmpeg libx264 (CPU). Used when GPU path is unavailable.
        Returns (success, backend_string). cpu_reason is for SSE (e.g. 'NVENC unavailable')."""
        if detection_sidecar_path and detection_model:
            self._write_detection_sidecar_cpu(
                temp_path,
                detection_sidecar_path,
                detection_model,
                (detection_device or "").strip() or None,
            )
        process = None
        try:
            command = [
                "ffmpeg", "-y",
                "-i", temp_path,
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-c:a", "aac",
                "-movflags", "+faststart",
                final_path,
            ]
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate(timeout=self.ffmpeg_timeout)
            if process.returncode == 0:
                os.remove(temp_path)
                logger.info(f"Transcoded clip for {event_id}")
                return (True, f"CPU: {cpu_reason}")
            logger.error(f"FFmpeg error for {event_id}: {stderr.decode()[:500]}")
            if os.path.exists(temp_path):
                os.rename(temp_path, final_path)
            return (True, f"CPU: {cpu_reason}")
        except subprocess.TimeoutExpired:
            self._terminate_process_gracefully(process, event_id)
            if os.path.exists(temp_path):
                os.rename(temp_path, final_path)
            return (True, f"CPU: {cpu_reason}")
        except Exception as e:
            logger.exception(f"Transcode failed for {event_id}: {e}")
            self._terminate_process_gracefully(process, event_id)
            if os.path.exists(temp_path):
                os.rename(temp_path, final_path)
            return (True, f"CPU: {cpu_reason}")

    def _write_detection_sidecar_cpu(
        self,
        temp_path: str,
        detection_sidecar_path: str,
        detection_model: str,
        device: str | None,
    ) -> None:
        """Decode temp_path with CPU (ffmpegcv.VideoCapture), run YOLO per frame, write detection.json.
        Same sidecar format as _transcode_clip_nvenc so multi-clip extractor can use it."""
        cap = None
        try:
            cap = ffmpegcv.VideoCapture(temp_path)
            if not cap.isOpened():
                return
            fps = getattr(cap, "fps", None) or 30.0
            yolo_model = None
            yolo_names: dict[int, str] = {}
            try:
                from ultralytics import YOLO
                yolo_model = YOLO(detection_model)
                yolo_names = yolo_model.names or {}
            except Exception as e:
                logger.warning("Could not load YOLO for detection sidecar (CPU path): %s", e)
                return
            sidecar_entries: list[dict[str, Any]] = []
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                det = _run_detection_on_frame(yolo_model, frame, device or "", yolo_names)
                sidecar_entries.append({"timestamp_sec": frame_idx / fps, "detections": det})
                frame_idx += 1
            if sidecar_entries:
                try:
                    with open(detection_sidecar_path, "w", encoding="utf-8") as f:
                        json.dump(sidecar_entries, f, indent=None, separators=(",", ":"))
                    logger.debug("Wrote detection sidecar %s (%d frames)", detection_sidecar_path, len(sidecar_entries))
                except OSError as e:
                    logger.warning("Failed to write detection sidecar %s: %s", detection_sidecar_path, e)
        finally:
            if cap is not None:
                cap.release()

    def generate_gif_from_clip(self, clip_path: str, output_path: str,
                               fps: int = 5, duration_sec: float = 5.0) -> bool:
        """Generate animated GIF from video clip using FFmpeg. Returns True on success."""
        try:
            scale = "320:-1"
            cmd = [
                "ffmpeg", "-y", "-i", clip_path,
                "-vf", f"fps={fps},scale={scale}",
                "-t", str(duration_sec),
                output_path
            ]
            proc = subprocess.run(cmd, capture_output=True, timeout=self.ffmpeg_timeout)
            if proc.returncode == 0 and os.path.exists(output_path):
                logger.info(f"Generated GIF from {clip_path}")
                return True
        except FileNotFoundError:
            logger.warning("GIF generation failed: 'ffmpeg' executable not found")
        except subprocess.CalledProcessError as e:
            logger.warning(f"GIF generation failed (subprocess error): {e}")
        except Exception as e:
            logger.warning(f"GIF generation failed: {e}")
        return False

    def _terminate_process_gracefully(self, process, event_id: str, timeout: float = 5.0):
        """Gracefully terminate a process: SIGTERM first, then SIGKILL if needed."""
        if process is None or process.poll() is not None:
            return  # Already dead

        logger.debug(f"Sending SIGTERM to FFmpeg for {event_id}")
        try:
            process.terminate()  # SIGTERM - allows graceful shutdown
        except OSError:
            return  # Process already gone

        try:
            process.wait(timeout=timeout)  # Wait for graceful exit
            logger.debug(f"FFmpeg for {event_id} terminated gracefully")
        except subprocess.TimeoutExpired:
            logger.warning(f"FFmpeg for {event_id} didn't respond to SIGTERM, sending SIGKILL")
            try:
                process.kill()  # SIGKILL - force kill
                process.wait()  # Reap zombie
            except OSError:
                pass  # Process already gone
