"""
Video service: PyNvVideoCodec GPU decode, YOLO detection, sidecars, GIF via FFmpeg.

Decode path uses gpu_decoder (SimpleDecoder, use_device_memory=True, max_width=4096).
Frames are BCHW RGB from decoder, normalized to float32 [0,1] before YOLO. VRAM released
after each chunk (del + empty_cache). GPU_LOCK serializes decoder and get_frames.
"""

from __future__ import annotations

import json
import logging
import math
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from frigate_buffer.constants import (
    GIF_PREVIEW_WIDTH,
    NVDEC_INIT_FAILURE_PREFIX,
    is_tensor,
)
from frigate_buffer.services.gpu_decoder import create_decoder

logger = logging.getLogger("frigate-buffer")

# Serialize PyNvVideoCodec decoder access (create_decoder and get_frames).
GPU_LOCK = threading.Lock()

# In-memory cache for ffprobe results keyed by clip path (realpath when exists).
# Reduces redundant subprocess I/O when clip probed by sidecar, extraction, compilation.
# Bounded: clear when over limit so metadata for deleted clips does not stay in RAM.
_METADATA_CACHE_MAX_SIZE = 500
_METADATA_CACHE: dict[str, tuple[int, int, float, float] | None] = {}


def _flush_logger() -> None:
    """Flush all logging handlers so logs are not lost on C++ segfault/abort."""
    for h in logger.handlers:
        try:
            h.flush()
        except Exception:
            pass
    root = logging.getLogger()
    for h in root.handlers:
        try:
            h.flush()
        except Exception:
            pass


# Chunk size for batched decode to protect VRAM when running YOLO.
BATCH_SIZE = 4

# COCO class 0 = person; we restrict YOLO to this class only for sidecar.
_PERSON_CLASS_ID = 0

# Throttle YOLO inference logs to at most once per this many seconds.
_YOLO_INFERENCE_LOG_INTERVAL = 5.0
_last_yolo_inference_log_time: float = 0.0


def log_gpu_status() -> None:
    """
    At startup: log GPU diagnostic info for decode (NVDEC) troubleshooting.

    Runs nvidia-smi to confirm GPU visibility. PyNvVideoCodec uses NVDEC for decode.
    """
    nvidia_smi = __import__("shutil").which("nvidia-smi")
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
            driver = (proc2.stdout or b"").decode(
                "utf-8", errors="replace"
            ).strip().split("\n")[0] or "?"
            logger.info(
                "GPU status: nvidia-smi OK, GPUs=%s, driver=%s (NVDEC for decode)",
                gpu_count,
                driver,
            )
        except (subprocess.TimeoutExpired, OSError) as e:
            logger.warning(
                "nvidia-smi found but failed: %s; container may not have GPU access.", e
            )
    else:
        logger.info("nvidia-smi not found; NVDEC decode requires GPU.")


def get_detection_model_path(config: dict) -> str:
    """
    Return the absolute path where the detection model should be stored/loaded.

    Uses STORAGE_PATH so the model persists across container restarts (e.g. in Docker).
    Ultralytics will download to this path if the file does not exist.
    """
    storage = config.get("STORAGE_PATH", "/app/storage")
    model_name = (config.get("DETECTION_MODEL") or "").strip() or "yolov8n.pt"
    return os.path.join(storage, "yolo_models", os.path.basename(model_name))


def ensure_detection_model_ready(config: dict) -> bool:
    """
    At startup: check if multi-cam detection model is downloaded; download if not.
    Call after config is loaded. Returns True if ready, False if skipped or failed.
    """
    model_name = (config.get("DETECTION_MODEL") or "").strip()
    if not model_name:
        logger.info(
            "Multi-cam detection model not configured (DETECTION_MODEL empty), skip",
        )
        return False
    model_path = get_detection_model_path(config)
    try:
        existed = os.path.isfile(model_path)
        from ultralytics import YOLO

        model = YOLO(model_path)
        ckpt = getattr(model, "ckpt_path", None)
        path_str = (
            f" at {ckpt}"
            if (ckpt and isinstance(ckpt, str) and os.path.isfile(ckpt))
            else ""
        )
        if existed or (ckpt and os.path.isfile(ckpt)):
            logger.info(
                "Multi-cam detection model ready: %s (already downloaded)%s",
                model_name,
                path_str,
            )
        else:
            logger.info(
                "Multi-cam detection model ready: %s (downloaded)%s",
                model_name,
                path_str,
            )
        return True
    except Exception as e:
        logger.warning(
            "Multi-cam detection model preload failed for %s: %s", model_name, e
        )
        return False


def _get_native_resolution(clip_path: str) -> tuple[int, int] | None:
    """Return (width, height) of the video stream from the clip file using ffprobe."""
    meta = _get_video_metadata(clip_path)
    if meta is None:
        return None
    w, h, _, _ = meta
    return (w, h) if (w > 0 and h > 0) else None


def _get_video_metadata(clip_path: str) -> tuple[int, int, float, float] | None:
    """
    Single ffprobe call returning (width, height, fps, duration_sec).
    Returns None on failure or missing stream.
    Results are cached by path (realpath when the file exists) so repeated callers
    (sidecar, multi-clip extraction, compilation) do not re-run ffprobe.
    """
    cache_key = os.path.realpath(clip_path) if os.path.exists(clip_path) else clip_path
    if len(_METADATA_CACHE) > _METADATA_CACHE_MAX_SIZE:
        _METADATA_CACHE.clear()
    if cache_key in _METADATA_CACHE:
        return _METADATA_CACHE[cache_key]
    try:
        out = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,r_frame_rate",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                clip_path,
            ],
            capture_output=True,
            timeout=10,
            check=False,
        )
        if out.returncode != 0 or not out.stdout:
            return None
        data = json.loads(out.stdout.decode("utf-8", errors="replace"))
        streams = data.get("streams") or []
        fmt = data.get("format") or {}
        if not streams:
            return None
        s = streams[0]
        w = int(s.get("width") or 0)
        h = int(s.get("height") or 0)
        fps_str = (s.get("r_frame_rate") or "").strip()
        if "/" in fps_str:
            num, den = fps_str.split("/", 1)
            fps = float(num) / float(den) if float(den) else 30.0
        else:
            fps = float(fps_str) if fps_str else 30.0
        dur_str = fmt.get("duration")
        duration = float(dur_str) if dur_str else 0.0
        result = (w, h, fps, duration)
        _METADATA_CACHE[cache_key] = result
        return result
    except (
        FileNotFoundError,
        subprocess.TimeoutExpired,
        ValueError,
        OSError,
        json.JSONDecodeError,
    ) as e:
        logger.debug("ffprobe metadata failed for %s: %s", clip_path, e)
    _METADATA_CACHE[cache_key] = None
    return None


def _decoder_reader_ready(reader: Any) -> bool:
    """
    True if the reader/decoder is ready for get_frames.

    Supports DecoderContext (gpu_decoder) and readers with frame_count or _decoder.
    """
    if hasattr(reader, "frame_count"):
        return True
    return hasattr(reader, "_decoder")


def _decoder_frame_count(
    reader: Any,
    fallback_fps: float,
    fallback_duration_sec: float,
) -> int:
    """
    Get frame count from decoder/reader, or ffprobe-derived duration * fps.
    Supports DecoderContext (len(ctx), ctx.frame_count) and readers with shape/len.
    """
    if hasattr(reader, "frame_count"):
        return reader.frame_count
    try:
        return len(reader)
    except (AttributeError, TypeError, Exception):
        pass
    try:
        shape = getattr(reader, "shape", None)
        if shape is not None and len(shape) >= 1:
            return int(shape[0])
    except (AttributeError, TypeError, IndexError, Exception):
        pass
    return max(0, int(fallback_duration_sec * fallback_fps))


def _scale_detections_to_native(
    detections: list[dict[str, Any]],
    read_w: int,
    read_h: int,
    native_w: int,
    native_h: int,
) -> list[dict[str, Any]]:
    """Scale bbox, centerpoint, area from decoded frame coords to native resolution."""
    if read_w <= 0 or read_h <= 0 or native_w <= 0 or native_h <= 0:
        return detections
    scale_x = native_w / read_w
    scale_y = native_h / read_h
    area_scale = scale_x * scale_y
    out: list[dict[str, Any]] = []
    for d in detections:
        bbox = d.get("bbox")
        cp = d.get("centerpoint")
        area = d.get("area", 0)
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            x1, y1, x2, y2 = (
                float(bbox[0]),
                float(bbox[1]),
                float(bbox[2]),
                float(bbox[3]),
            )
            bbox = [
                x1 * scale_x,
                y1 * scale_y,
                x2 * scale_x,
                y2 * scale_y,
            ]
        else:
            bbox = d.get("bbox", [])
        if isinstance(cp, (list, tuple)) and len(cp) >= 2:
            cp = [float(cp[0]) * scale_x, float(cp[1]) * scale_y]
        else:
            cp = d.get("centerpoint", [])
        out.append(
            {
                "label": d.get("label", "person"),
                "bbox": bbox,
                "centerpoint": cp,
                "area": int(float(area) * area_scale) if area else 0,
            }
        )
    return out


def _run_detection_on_frame(
    model: Any,
    frame: Any,
    device: str | None,
    imgsz: int = 640,
) -> list[dict[str, Any]]:
    """
    Run YOLO on one frame (person class); return {label, bbox, centerpoint, area}.
    frame: numpy HWC BGR or torch.Tensor BCHW; if uint8, normalized to float32 [0,1].
    """
    import torch

    detections: list[dict[str, Any]] = []
    try:
        if is_tensor(frame):
            t = frame
            if t.dim() == 3:
                t = t.unsqueeze(0)
            if t.dtype == torch.uint8:
                t = t.float()
                t.div_(255.0)
            elif t.dtype != torch.float32 and t.dtype != torch.float64:
                t = t.float()
        else:
            import numpy as np

            arr = np.asarray(frame, dtype=np.uint8)
            if arr.ndim == 2:
                arr = np.expand_dims(arr, axis=-1)
            if arr.ndim == 3:
                arr = np.expand_dims(arr, axis=0)
            if arr.shape[-1] == 3:
                arr = arr[:, :, :, [2, 1, 0]]
            t = torch.from_numpy(arr.copy()).float() / 255.0
            if t.dim() == 4 and t.shape[-1] == 3:
                t = t.permute(0, 3, 1, 2)
            if torch.cuda.is_available():
                t = t.cuda()
        results = model(
            t, device=device, verbose=False, classes=[_PERSON_CLASS_ID], imgsz=imgsz
        )
        if not results:
            return detections
        for r in results:
            if r.boxes is None:
                continue
            xyxy = r.boxes.xyxy
            if xyxy is None:
                continue
            try:
                logger.debug("_run_detection: converting xyxy tensor to CPU/numpy")
                _flush_logger()
                xyxy = xyxy.cpu().numpy() if hasattr(xyxy, "cpu") else xyxy
                logger.debug("_run_detection: xyxy converted to numpy")
                _flush_logger()
            except Exception:
                pass
            for i in range(len(xyxy)):
                if len(xyxy[i]) < 4:
                    continue
                x1, y1, x2, y2 = (
                    float(xyxy[i][0]),
                    float(xyxy[i][1]),
                    float(xyxy[i][2]),
                    float(xyxy[i][3]),
                )
                area = int((x2 - x1) * (y2 - y1))
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                detections.append(
                    {
                        "label": "person",
                        "bbox": [x1, y1, x2, y2],
                        "centerpoint": [cx, cy],
                        "area": area,
                    }
                )
    except Exception as e:
        logger.debug("Detection on frame failed: %s", e)
    return detections


def _run_detection_on_batch(
    model: Any,
    batch: Any,
    device: str | None,
    imgsz: int = 640,
) -> list[list[dict[str, Any]]]:
    """
    Run YOLO on batch (BCHW float32 [0,1]); return list of detection lists per frame.
    Frames resized on GPU to imgsz (aspect ratio, 32-multiple); bboxes scaled.
    """
    global _last_yolo_inference_log_time
    import torch
    import torch.nn.functional as F

    batch_size = int(batch.shape[0])
    read_h, read_w = int(batch.shape[2]), int(batch.shape[3])
    out: list[list[dict[str, Any]]] = []
    try:
        if batch.dtype == torch.uint8:
            batch = batch.float()
            batch.div_(255.0)
        target_imgsz = max(320, int(imgsz))
        scale = target_imgsz / max(read_h, read_w) if max(read_h, read_w) > 0 else 1.0
        target_w = max(32, math.ceil((read_w * scale) / 32) * 32)
        target_h = max(32, math.ceil((read_h * scale) / 32) * 32)
        batch_resized = F.interpolate(
            batch, size=(target_h, target_w), mode="bilinear", align_corners=False
        )
        now = time.monotonic()
        should_log = (
            now - _last_yolo_inference_log_time
        ) >= _YOLO_INFERENCE_LOG_INTERVAL
        if should_log:
            _last_yolo_inference_log_time = now
            logger.debug(
                "_run_detection_on_batch: calling model.predict (YOLO inference)"
            )
            _flush_logger()
        results = model(
            batch_resized,
            device=device,
            verbose=False,
            classes=[_PERSON_CLASS_ID],
            imgsz=imgsz,
        )
        if should_log:
            logger.debug("_run_detection_on_batch: model.predict returned")
            _flush_logger()
        scale_x = read_w / target_w if target_w > 0 else 1.0
        scale_y = read_h / target_h if target_h > 0 else 1.0
        if not results:
            return [[] for _ in range(batch_size)]
        for r in results:
            detections: list[dict[str, Any]] = []
            if r.boxes is not None and r.boxes.xyxy is not None:
                xyxy = r.boxes.xyxy
                try:
                    logger.debug(
                        "_run_detection_on_batch: converting xyxy tensor to CPU/numpy"
                    )
                    _flush_logger()
                    xyxy = xyxy.cpu().numpy() if hasattr(xyxy, "cpu") else xyxy
                    logger.debug("_run_detection_on_batch: xyxy converted to numpy")
                    _flush_logger()
                except Exception:
                    pass
                for i in range(len(xyxy)):
                    if len(xyxy[i]) < 4:
                        continue
                    x1 = float(xyxy[i][0]) * scale_x
                    y1 = float(xyxy[i][1]) * scale_y
                    x2 = float(xyxy[i][2]) * scale_x
                    y2 = float(xyxy[i][3]) * scale_y
                    area = int((x2 - x1) * (y2 - y1))
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    detections.append(
                        {
                            "label": "person",
                            "bbox": [x1, y1, x2, y2],
                            "centerpoint": [cx, cy],
                            "area": area,
                        }
                    )
            out.append(detections)
        while len(out) < batch_size:
            out.append([])
    except Exception as e:
        logger.debug("Detection on batch failed: %s", e)
        out = [[] for _ in range(batch_size)]
    return out


class VideoService:
    """Video decode (NVDEC), YOLO detection, sidecars, GIF via FFmpeg subprocess."""

    DEFAULT_FFMPEG_TIMEOUT = 60

    def __init__(self, ffmpeg_timeout: int = DEFAULT_FFMPEG_TIMEOUT):
        self.ffmpeg_timeout = ffmpeg_timeout
        self._sidecar_app_lock: threading.Lock | None = None
        logger.debug(
            "VideoService initialized with FFmpeg timeout=%ss (GIF only)",
            ffmpeg_timeout,
        )

    def set_sidecar_app_lock(self, lock: threading.Lock) -> None:
        """Set app-level lock so one sidecar batch (TEST or lifecycle) runs at once."""
        self._sidecar_app_lock = lock

    def run_detection_on_image(
        self, image_bgr: Any, config: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """
        Run YOLO person detection on a single image (e.g. from latest.jpg).
        Accepts numpy HWC BGR or torch.Tensor BCHW; normalizes to float32 [0,1].
        Uses app-level sidecar lock. Returns list of {label, bbox, centerpoint, area}.
        """
        lock = self._sidecar_app_lock
        if lock is not None:
            lock.acquire()
        try:
            detection_model = (config.get("DETECTION_MODEL") or "").strip()
            detection_device = (config.get("DETECTION_DEVICE") or "").strip() or None
            detection_imgsz = max(320, int(config.get("DETECTION_IMGSZ", 640)))
            if not detection_model:
                return []
            try:
                from ultralytics import YOLO

                model_path = get_detection_model_path(config)
                model = YOLO(model_path)
            except Exception as e:
                logger.warning("Could not load YOLO for run_detection_on_image: %s", e)
                return []
            return _run_detection_on_frame(
                model, image_bgr, detection_device, imgsz=detection_imgsz
            )
        finally:
            if lock is not None:
                lock.release()

    def generate_detection_sidecar(
        self,
        clip_path: str,
        sidecar_path: str,
        config: dict,
        *,
        yolo_model: Any = None,
        yolo_lock: threading.Lock | None = None,
    ) -> bool:
        """
        Decode clip with PyNvVideoCodec, run YOLO every N frames, write detection.json.
        No CPU/ffmpegcv fallback; on failure logs context and returns False.
        Frames BCHW RGB, float32 [0,1]; after each chunk del batch and empty_cache().
        """
        import torch

        detection_model = (config.get("DETECTION_MODEL") or "").strip()
        detection_device = (config.get("DETECTION_DEVICE") or "").strip() or None
        detection_frame_interval = max(
            1, int(config.get("DETECTION_FRAME_INTERVAL", 5))
        )
        detection_imgsz = max(320, int(config.get("DETECTION_IMGSZ", 640)))
        cuda_device_index = int(config.get("CUDA_DEVICE_INDEX", 0))

        meta = _get_video_metadata(clip_path)
        if meta:
            native_w, native_h, fps_val, duration_sec = meta
        else:
            native_w, native_h, fps_val, duration_sec = 0, 0, 30.0, 0.0

        fps = fps_val if fps_val > 0 else 30.0

        try:
            with GPU_LOCK:
                with create_decoder(clip_path, gpu_id=cuda_device_index) as ctx:
                    frame_count = len(ctx)
                    if frame_count <= 0 and duration_sec > 0:
                        frame_count = max(1, int(duration_sec * fps))
                    if frame_count <= 0:
                        logger.warning(
                            "Could not determine frame count for %s", clip_path
                        )
                        return False
                    interval = detection_frame_interval
                    indices = list(range(0, frame_count, interval))
                    if not indices:
                        indices = [0]

                    model_to_use = yolo_model
                    if model_to_use is None and detection_model:
                        try:
                            from ultralytics import YOLO

                            model_path = get_detection_model_path(config)
                            model_to_use = YOLO(model_path)
                            logger.debug(
                                "YOLO loaded for detection sidecar: %s", clip_path
                            )
                        except Exception as e:
                            logger.warning(
                                "Could not load YOLO for detection sidecar: %s", e
                            )

                    sidecar_entries: list[dict[str, Any]] = []
                    read_h, read_w = 0, 0

                    for chunk_start in range(0, len(indices), BATCH_SIZE):
                        chunk_indices = indices[chunk_start : chunk_start + BATCH_SIZE]
                        try:
                            batch = ctx.get_frames(chunk_indices)
                        except Exception as e:
                            logger.error(
                                "%s get_frames failed. clip=%s indices=%s error=%s",
                                NVDEC_INIT_FAILURE_PREFIX,
                                clip_path,
                                chunk_indices,
                                e,
                                exc_info=True,
                            )
                            logger.warning(
                                "Decoder get_frames failed for %s: %s",
                                clip_path,
                                e,
                            )
                            if torch.cuda.is_available():
                                try:
                                    logger.debug(
                                        "VRAM summary: %s", torch.cuda.memory_summary()
                                    )
                                except Exception:
                                    pass
                            break

                        if read_h == 0 and batch is not None and batch.numel() > 0:
                            read_h = int(batch.shape[2])
                            read_w = int(batch.shape[3])

                        batch = batch.float()
                        batch.div_(255.0)

                        if model_to_use is not None:
                            if yolo_lock is not None:
                                with yolo_lock:
                                    det_lists = _run_detection_on_batch(
                                        model_to_use,
                                        batch,
                                        detection_device,
                                        imgsz=detection_imgsz,
                                    )
                            else:
                                det_lists = _run_detection_on_batch(
                                    model_to_use,
                                    batch,
                                    detection_device,
                                    imgsz=detection_imgsz,
                                )
                            for i, idx in enumerate(chunk_indices):
                                det = det_lists[i] if i < len(det_lists) else []
                                if (
                                    native_w > 0
                                    and native_h > 0
                                    and read_w > 0
                                    and read_h > 0
                                ):
                                    det = _scale_detections_to_native(
                                        det, read_w, read_h, native_w, native_h
                                    )
                                sidecar_entries.append(
                                    {
                                        "frame_number": idx,
                                        "timestamp_sec": idx / fps if fps > 0 else 0,
                                        "detections": det,
                                    }
                                )
                        else:
                            for idx in chunk_indices:
                                sidecar_entries.append(
                                    {
                                        "frame_number": idx,
                                        "timestamp_sec": idx / fps if fps > 0 else 0,
                                        "detections": [],
                                    }
                                )

                        del batch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()  # Reclaim VRAM after each batch

                    if native_w <= 0 and read_w > 0:
                        native_w, native_h = read_w, read_h
                payload = {
                    "native_width": native_w or read_w,
                    "native_height": native_h or read_h,
                    "entries": sidecar_entries,
                }
            # GPU_LOCK and decoder exited; write outside lock to avoid holding GPU idle.
            with open(sidecar_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=None, separators=(",", ":"))
            logger.debug(
                "Wrote detection sidecar %s (%d frames)",
                sidecar_path,
                len(sidecar_entries),
            )
            return True
        except Exception as e:
            logger.error(
                "%s (decoder open or decode failed). path=%s error=%s",
                NVDEC_INIT_FAILURE_PREFIX,
                clip_path,
                e,
            )
            logger.warning(
                "Failed to generate detection sidecar for %s: %s",
                clip_path,
                e,
                exc_info=True,
            )
            if torch.cuda.is_available():
                try:
                    logger.debug("VRAM summary: %s", torch.cuda.memory_summary())
                except Exception:
                    pass
            return False
        finally:
            try:
                del batch
            except NameError:
                pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def generate_detection_sidecars_for_cameras(
        self,
        tasks: list[tuple[str, str, str]],
        config: dict[str, Any],
    ) -> list[tuple[str, bool]]:
        """
        Generate detection sidecars for multiple cams in parallel; shared YOLO and lock.
        tasks: (camera_name, clip_path, sidecar_path). Returns (camera_name, ok).
        """
        if not tasks:
            return []
        lock = self._sidecar_app_lock
        if lock is not None:
            lock.acquire()
        try:
            yolo_model = None
            yolo_lock = threading.Lock()
            detection_model = (config.get("DETECTION_MODEL") or "").strip()
            if detection_model:
                try:
                    from ultralytics import YOLO

                    model_path = get_detection_model_path(config)
                    yolo_model = YOLO(model_path)
                except Exception as e:
                    logger.warning("Could not load shared YOLO for sidecar: %s", e)
            results: list[tuple[str, bool]] = []
            with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
                future_to_cam = {}
                for camera_name, clip_path, sidecar_path in tasks:
                    future = executor.submit(
                        self.generate_detection_sidecar,
                        clip_path,
                        sidecar_path,
                        config,
                        yolo_model=yolo_model,
                        yolo_lock=yolo_lock if yolo_model is not None else None,
                    )
                    future_to_cam[future] = camera_name
                for future in as_completed(future_to_cam):
                    camera_name = future_to_cam[future]
                    try:
                        ok = future.result()
                    except Exception:
                        logger.exception("Sidecar %s failed", camera_name)
                        ok = False
                    results.append((camera_name, ok))
            order = {cam: i for i, (cam, _, _) in enumerate(tasks)}
            results.sort(key=lambda r: order.get(r[0], 0))
            return results
        finally:
            if lock is not None:
                lock.release()

    def generate_gif_from_clip(
        self, clip_path: str, output_path: str, fps: int = 5, duration_sec: float = 5.0
    ) -> bool:
        """
        Generate animated GIF from video clip using FFmpeg with CUDA hardware
        acceleration. Uses NVDEC for decode and scale_cuda for resize to minimize
        CPU usage. The CPU is only used for the final GIF palette generation
        and encoding.
        """
        try:
            # Mandatory HW acceleration: no CPU fallback.
            # scale_cuda uses GIF_PREVIEW_WIDTH from constants.
            # palettegen/paletteuse are CPU-bound filters, so we hwdownload after
            # scaling.
            filter_str = (
                f"scale_cuda={GIF_PREVIEW_WIDTH}:-1,"
                f"hwdownload,format=nv12,format=rgb24,"
                f"fps={fps},split[a][b];"
                f"[a]palettegen=stats_mode=single[p];"
                f"[b][p]paletteuse=dither=bayer"
            )

            cmd = [
                "ffmpeg",
                "-y",
                "-hwaccel",
                "cuda",
                "-hwaccel_output_format",
                "cuda",
                "-i",
                clip_path,
                "-t",
                str(duration_sec),
                "-filter_complex",
                filter_str,
                output_path,
            ]
            proc = subprocess.run(cmd, capture_output=True, timeout=self.ffmpeg_timeout)
            if proc.returncode == 0 and os.path.exists(output_path):
                logger.info("Generated HW-accelerated GIF from %s", clip_path)
                return True
            else:
                stderr = (
                    proc.stderr.decode("utf-8", errors="replace") if proc.stderr else ""
                )
                logger.error(
                    "GIF generation failed (exit %d). cmd=%s stderr=%s",
                    proc.returncode,
                    " ".join(cmd),
                    stderr,
                )
        except FileNotFoundError:
            logger.warning("GIF generation failed: 'ffmpeg' executable not found")
        except subprocess.TimeoutExpired:
            logger.warning(
                "GIF generation failed: FFmpeg timed out after %ss", self.ffmpeg_timeout
            )
        except Exception as e:
            logger.warning("GIF generation failed: %s", e, exc_info=True)
        return False
