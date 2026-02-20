import json
import os
import subprocess
import logging
from typing import Any

import ffmpegcv

logger = logging.getLogger('frigate-buffer')


def log_gpu_status() -> None:
    """
    At startup: log GPU diagnostic info for decode (NVDEC) troubleshooting.

    Runs nvidia-smi to confirm GPU visibility. Hardware decode (NVDEC) is used
    by ffmpegcv for frame reading; encoding is not used.
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
            driver = (proc2.stdout or b"").decode("utf-8", errors="replace").strip().split("\n")[0] or "?"
            logger.info("GPU status: nvidia-smi OK, GPUs=%s, driver=%s (NVDEC used for decode)", gpu_count, driver)
        except (subprocess.TimeoutExpired, OSError) as e:
            logger.warning("nvidia-smi found but failed: %s; container may not have GPU access.", e)
    else:
        logger.info("nvidia-smi not found; NVDEC decode may fall back to CPU.")


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


# COCO class 0 = person; we restrict YOLO to this class only for sidecar.
_PERSON_CLASS_ID = 0


def _get_native_resolution(clip_path: str) -> tuple[int, int] | None:
    """
    Return (width, height) of the video stream from the clip file using ffprobe.
    Used so we can scale YOLO bbox from decoded (resized) frame coords back to native for the sidecar.
    """
    try:
        out = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=width,height", "-of", "csv=p=0",
                clip_path,
            ],
            capture_output=True,
            timeout=10,
            check=False,
        )
        if out.returncode != 0 or not out.stdout:
            return None
        line = out.stdout.decode("utf-8", errors="replace").strip().split("\n")[0]
        parts = line.split(",")
        if len(parts) >= 2:
            return int(parts[0].strip()), int(parts[1].strip())
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, OSError) as e:
        logger.debug("ffprobe native resolution failed for %s: %s", clip_path, e)
    return None


def _scale_detections_to_native(
    detections: list[dict[str, Any]],
    read_w: int,
    read_h: int,
    native_w: int,
    native_h: int,
) -> list[dict[str, Any]]:
    """Scale bbox, centerpoint, and area from decoded (read) frame coordinates to native resolution."""
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
            x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
            bbox = [
                x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y,
            ]
        else:
            bbox = d.get("bbox", [])
        if isinstance(cp, (list, tuple)) and len(cp) >= 2:
            cp = [float(cp[0]) * scale_x, float(cp[1]) * scale_y]
        else:
            cp = d.get("centerpoint", [])
        out.append({
            "label": d.get("label", "person"),
            "bbox": bbox,
            "centerpoint": cp,
            "area": int(float(area) * area_scale) if area else 0,
        })
    return out


def _run_detection_on_frame(
    model: Any,
    frame: Any,
    device: str | None,
    imgsz: int = 640,
) -> list[dict[str, Any]]:
    """Run YOLO on one frame (person class only); return list of {label, bbox, centerpoint, area} for sidecar."""
    detections: list[dict[str, Any]] = []
    try:
        # classes=[0]: COCO person; hardcoded for robustness across models
        results = model(
            frame, device=device, verbose=False, classes=[_PERSON_CLASS_ID], imgsz=imgsz
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
                xyxy = xyxy.cpu().numpy() if hasattr(xyxy, "cpu") else xyxy
            except Exception:
                pass
            for i in range(len(xyxy)):
                if len(xyxy[i]) < 4:
                    continue
                x1, y1, x2, y2 = float(xyxy[i][0]), float(xyxy[i][1]), float(xyxy[i][2]), float(xyxy[i][3])
                area = int((x2 - x1) * (y2 - y1))
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                detections.append({
                    "label": "person",
                    "bbox": [x1, y1, x2, y2],
                    "centerpoint": [cx, cy],
                    "area": area,
                })
    except Exception as e:
        logger.debug("Detection on frame failed: %s", e)
    return detections


class VideoService:
    """Handles video decode (frame reading) and GIF generation. No encoding; clips are used as-is."""

    DEFAULT_FFMPEG_TIMEOUT = 60

    def __init__(self, ffmpeg_timeout: int = DEFAULT_FFMPEG_TIMEOUT):
        self.ffmpeg_timeout = ffmpeg_timeout
        logger.debug("VideoService initialized with FFmpeg timeout: %ss", ffmpeg_timeout)

    def generate_detection_sidecar(
        self,
        clip_path: str,
        sidecar_path: str,
        config: dict,
    ) -> bool:
        """
        Decode-only pass: read clip with ffmpegcv (optionally resized for GPU efficiency), run YOLO every N frames, write detection.json.

        Uses NVDEC (VideoCaptureNV) when available, else CPU (VideoCapture). May resize during decode to crop target size (CROP_WIDTH x CROP_HEIGHT); all bbox/centerpoint/area in the sidecar are scaled to **native** resolution so Phase 3 crop logic is correct. Writes native_width and native_height in the sidecar. Does not re-encode.
        Returns True if sidecar was written (even if empty); False on open/read failure.
        """
        detection_model = (config.get("DETECTION_MODEL") or "").strip()
        detection_device = (config.get("DETECTION_DEVICE") or "").strip() or None
        detection_frame_interval = max(1, int(config.get("DETECTION_FRAME_INTERVAL", 5)))
        detection_imgsz = max(320, int(config.get("DETECTION_IMGSZ", 640)))
        crop_w = max(1, int(config.get("CROP_WIDTH", 1280)))
        crop_h = max(1, int(config.get("CROP_HEIGHT", 720)))

        native_res = _get_native_resolution(clip_path)
        native_w = native_res[0] if native_res else 0
        native_h = native_res[1] if native_res else 0
        use_resize = native_w > 0 and native_h > 0 and (native_w != crop_w or native_h != crop_h)

        cap = None
        try:
            try:
                if use_resize:
                    cap = ffmpegcv.VideoCaptureNV(
                        clip_path, resize=(crop_w, crop_h), resize_keepratio=False
                    )
                else:
                    cap = ffmpegcv.VideoCaptureNV(clip_path)
            except Exception:
                try:
                    if use_resize:
                        cap = ffmpegcv.VideoCapture(
                            clip_path, resize=(crop_w, crop_h), resize_keepratio=False
                        )
                    else:
                        cap = ffmpegcv.VideoCapture(clip_path)
                except Exception:
                    cap = ffmpegcv.VideoCapture(clip_path)
            if not cap.isOpened():
                logger.warning("Could not open clip for sidecar: %s", clip_path)
                return False

            read_h, read_w = 0, 0
            fps = getattr(cap, "fps", None) or 30.0
            yolo_model = None
            if detection_model:
                try:
                    from ultralytics import YOLO
                    yolo_model = YOLO(detection_model)
                    logger.debug("YOLO loaded for detection sidecar: %s", clip_path)
                except Exception as e:
                    logger.warning("Could not load YOLO for detection sidecar: %s", e)

            sidecar_entries: list[dict[str, Any]] = []
            frame_idx = 0
            interval = detection_frame_interval

            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                if read_h == 0 and frame is not None:
                    read_h, read_w = frame.shape[:2]
                if yolo_model is not None and frame_idx % interval == 0:
                    det = _run_detection_on_frame(
                        yolo_model, frame, detection_device, imgsz=detection_imgsz
                    )
                    if use_resize and native_w > 0 and native_h > 0 and read_w > 0 and read_h > 0:
                        det = _scale_detections_to_native(det, read_w, read_h, native_w, native_h)
                    sidecar_entries.append({
                        "frame_number": frame_idx,
                        "timestamp_sec": frame_idx / fps,
                        "detections": det,
                    })
                frame_idx += 1

            if native_w <= 0 and read_w > 0:
                native_w, native_h = read_w, read_h
            payload: dict[str, Any] = {
                "native_width": native_w or read_w,
                "native_height": native_h or read_h,
                "entries": sidecar_entries,
            }
            with open(sidecar_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=None, separators=(",", ":"))
            logger.debug("Wrote detection sidecar %s (%d frames)", sidecar_path, len(sidecar_entries))
            return True
        except Exception as e:
            logger.warning("Failed to generate detection sidecar for %s: %s", clip_path, e)
            return False
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass

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
                logger.info("Generated GIF from %s", clip_path)
                return True
        except FileNotFoundError:
            logger.warning("GIF generation failed: 'ffmpeg' executable not found")
        except subprocess.CalledProcessError as e:
            logger.warning("GIF generation failed (subprocess error): %s", e)
        except Exception as e:
            logger.warning("GIF generation failed: %s", e)
        return False
