import json
import os
import shutil
import subprocess
import logging
from typing import Any

import ffmpegcv

logger = logging.getLogger('frigate-buffer')

STDERR_TRUNCATE = 800


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
                    "Ensure NVIDIA_DRIVER_CAPABILITIES=all and NVIDIA Container Toolkit is configured."
                )

    # ffmpeg encoders
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        try:
            proc = subprocess.run(
                [ffmpeg_path, "-encoders"],
                capture_output=True,
                timeout=10,
            )
            encoders = (proc.stderr or proc.stdout or b"").decode("utf-8", errors="replace")
            has_nvenc = "h264_nvenc" in encoders or "hevc_nvenc" in encoders
            if has_nvenc:
                logger.info("FFmpeg reports NVENC encoders (h264_nvenc/hevc_nvenc) available")
            else:
                logger.warning(
                    "FFmpeg does not report NVENC encoders; GPU transcode will be unavailable. "
                    "BtbN GPL/LGPL builds do not include NVENC. Build FFmpeg with --enable-nvenc "
                    "or use an image/FFmpeg that does (see this repo's Dockerfile)."
                )
        except (subprocess.TimeoutExpired, OSError) as e:
            logger.warning("Could not check ffmpeg encoders: %s", e)
    else:
        logger.warning("ffmpeg not found in PATH")


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

    def __init__(self, ffmpeg_timeout: int = DEFAULT_FFMPEG_TIMEOUT):
        self.ffmpeg_timeout = ffmpeg_timeout
        self._nvenc_available: bool | None = None  # None = not yet probed
        logger.debug(f"VideoService initialized with FFmpeg timeout: {ffmpeg_timeout}s")

    def _probe_nvenc(self) -> bool:
        """
        Run a minimal NVENC encode to detect if GPU encoding is available.
        Logs exactly once (INFO) with the result. Returns True if NVENC is usable.
        """
        if self._nvenc_available is not None:
            return self._nvenc_available
        try:
            cmd = [
                "ffmpeg", "-y", "-f", "lavfi", "-i", "nullsrc=s=64x64:d=0.1",
                "-c:v", "h264_nvenc", "-frames:v", "1", "-f", "null", "-",
            ]
            proc = subprocess.run(
                cmd,
                capture_output=True,
                timeout=15,
            )
            stderr = (proc.stderr or b"").decode("utf-8", errors="replace")
            if proc.returncode != 0 or "No capable devices found" in stderr or "cannot load" in stderr.lower():
                self._nvenc_available = False
                stderr_snippet = stderr[:STDERR_TRUNCATE] + ("..." if len(stderr) > STDERR_TRUNCATE else "")
                logger.warning(
                    "NVENC probe failed (returncode=%s). stderr: %s",
                    proc.returncode,
                    stderr_snippet,
                )
                logger.info("NVENC probe: unavailable, using libx264 for transcodes")
                return False
            self._nvenc_available = True
            logger.info("NVENC probe: available, GPU transcode enabled")
            return True
        except Exception as e:
            self._nvenc_available = False
            logger.warning("NVENC probe exception: %s", e)
            logger.info("NVENC probe: unavailable, using libx264 for transcodes")
            return False

    def transcode_clip_to_h264(
        self,
        event_id: str,
        temp_path: str,
        final_path: str,
        detection_sidecar_path: str | None = None,
        detection_model: str | None = None,
        detection_device: str | None = None,
) -> bool:
        """Transcode clip_original.mp4 to H.264 clip.mp4 (NVDEC decode, NVENC encode when GPU available).
        Removes temp on success. Falls back to libx264 if GPU path fails.
        When detection_sidecar_path is set (multi-cam), runs ultralytics on each frame in the NVENC
        pass and writes detection.json; if we fall back to libx264 no sidecar is written."""
        if not self._probe_nvenc():
            return self._transcode_clip_libx264(event_id, temp_path, final_path)
        try:
            if self._transcode_clip_nvenc(
                event_id, temp_path, final_path,
                detection_sidecar_path=detection_sidecar_path,
                detection_model=detection_model,
                detection_device=detection_device,
            ):
                return True
        except Exception as e:
            logger.warning(
                "GPU transcode failed (%s). Reason: %s: %s. "
                "Ensure: (1) ffmpeg built with NVENC, (2) NVIDIA Container Toolkit running, "
                "(3) deploy.resources.reservations.devices includes GPU.",
                event_id, type(e).__name__, e,
            )
        return self._transcode_clip_libx264(event_id, temp_path, final_path)

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

    def _transcode_clip_libx264(self, event_id: str, temp_path: str, final_path: str) -> bool:
        """Transcode using ffmpeg libx264 (CPU). Used when GPU path is unavailable."""
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
                return True
            logger.error(f"FFmpeg error for {event_id}: {stderr.decode()[:500]}")
            if os.path.exists(temp_path):
                os.rename(temp_path, final_path)
            return True
        except subprocess.TimeoutExpired:
            self._terminate_process_gracefully(process, event_id)
            if os.path.exists(temp_path):
                os.rename(temp_path, final_path)
            return True
        except Exception as e:
            logger.exception(f"Transcode failed for {event_id}: {e}")
            self._terminate_process_gracefully(process, event_id)
            if os.path.exists(temp_path):
                os.rename(temp_path, final_path)
            return True

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
