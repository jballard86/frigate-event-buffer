"""
Video compilation service.
Handles segment-level video processing using hardware acceleration.
Generates single, stitched, cropped compilation videos.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from collections import Counter, defaultdict
from typing import Any

from frigate_buffer.constants import (
    COMPILATION_DEFAULT_NATIVE_HEIGHT,
    COMPILATION_DEFAULT_NATIVE_WIDTH,
    NVDEC_INIT_FAILURE_PREFIX,
)
from frigate_buffer.services import crop_utils, timeline_ema
from frigate_buffer.services.compilation_math import (
    _nearest_entry_at_t,
    calculate_crop_at_time,
    calculate_segment_crop,
    smooth_crop_centers_ema,
    smooth_zoom_ema,
)  # type: ignore[reportMissingModuleSource]
from frigate_buffer.services.gpu_decoder import create_decoder
from frigate_buffer.services.video import GPU_LOCK, _get_video_metadata

logger = logging.getLogger("frigate-buffer")

# Re-export for tests and callers that import from video_compilation
convert_timeline_to_segments = timeline_ema.convert_timeline_to_segments
assignments_to_slices = timeline_ema.assignments_to_slices
_trim_slices_to_action_window = timeline_ema._trim_slices_to_action_window

# Output frame rate for compilation (smooth panning samples at this rate).
COMPILATION_OUTPUT_FPS = 20

# Chunk size for batched decode to protect VRAM (same pattern as video.py).
# Decoder returns up to this many frames per get_frames call; after each chunk we
# del batch and empty_cache.
BATCH_SIZE = 4


def _compilation_ffmpeg_cmd_and_log_path(
    tmp_output_path: str, target_w: int, target_h: int
) -> tuple[list[str], str]:
    """
    Build FFmpeg h264_nvenc command and log path for compilation encode.
    Shared by _encode_frames_via_ffmpeg and _run_pynv_compilation (streaming path).
    """
    log_file_path = os.path.join(
        os.path.dirname(tmp_output_path), "ffmpeg_compile.log"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{target_w}x{target_h}",
        "-r",
        "20",
        "-thread_queue_size",
        "512",
        "-i",
        "pipe:0",
        "-c:v",
        "h264_nvenc",
        "-preset",
        "p1",
        "-tune",
        "hq",
        "-rc",
        "vbr",
        "-cq",
        "24",
        "-an",
        "-pix_fmt",
        "yuv420p",
        tmp_output_path,
    ]
    return cmd, log_file_path


def _open_compilation_ffmpeg_process(
    tmp_output_path: str, target_w: int, target_h: int
) -> tuple[subprocess.Popen[bytes], str, Any, list[str]]:
    """
    Open FFmpeg subprocess for compilation encode; stderr to log file.
    On FileNotFoundError closes log and raises.
    Returns (proc, log_file_path, log_file, cmd). Caller must call
    _close_compilation_ffmpeg_and_check when done (or on exception).
    """
    cmd, log_file_path = _compilation_ffmpeg_cmd_and_log_path(
        tmp_output_path, target_w, target_h
    )
    log_file = open(log_file_path, "w", encoding="utf-8")
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=log_file,
        )
    except FileNotFoundError:
        log_file.close()
        logger.error(
            "Compilation encode failed: ffmpeg not found. "
            "Compilation requires GPU encoding (h264_nvenc). No CPU fallback. "
            "Ensure FFmpeg is installed and on PATH with NVENC support.",
        )
        raise RuntimeError(
            "ffmpeg not found; compilation encoding is GPU-only (h264_nvenc), "
            "no CPU fallback"
        ) from None
    return proc, log_file_path, log_file, cmd


def _close_compilation_ffmpeg_and_check(
    proc: subprocess.Popen[bytes],
    log_file: Any,
    log_file_path: str,
    cmd: list[str],
    *,
    check_returncode: bool = True,
) -> None:
    """
    Close stdin, wait for process, close log file. If check_returncode and
    returncode != 0, log and raise RuntimeError. Used by both _encode_frames_via_ffmpeg
    and _run_pynv_compilation. Set check_returncode=False when caller will raise its own
    error (e.g. BrokenPipeError).
    """
    if proc.stdin is not None:
        proc.stdin.close()
    proc.wait()
    log_file.close()
    if check_returncode and proc.returncode != 0:
        logger.error(
            "Compilation encode failed: FFmpeg exited with code %s. Command: %s. "
            "Check %s for full stderr.",
            proc.returncode,
            " ".join(cmd),
            log_file_path,
        )
        raise RuntimeError(
            f"FFmpeg h264_nvenc encode failed (exit {proc.returncode}); "
            f"check {log_file_path!r} for details"
        )


def _encode_frames_via_ffmpeg(
    frames: list,
    target_w: int,
    target_h: int,
    tmp_output_path: str,
) -> None:
    """
    Encode a list of RGB frames to MP4 using FFmpeg with h264_nvenc only (GPU).

    frames: list of numpy arrays (H, W, 3) uint8 RGB with H=target_h, W=target_w.
    FFmpeg stderr is routed to a physical log file in the event folder to avoid
    OS-level pipe deadlock. h264_nvenc cannot ingest rgb24; we request yuv420p output.
    No CPU fallback; on failure logs and raises, advising to check ffmpeg_compile.log.
    """
    if not frames:
        return
    proc, log_file_path, log_file, cmd = _open_compilation_ffmpeg_process(
        tmp_output_path, target_w, target_h
    )
    assert proc.stdin is not None
    try:
        for arr in frames:
            proc.stdin.write(arr.tobytes())
        _close_compilation_ffmpeg_and_check(proc, log_file, log_file_path, cmd)
    except BrokenPipeError:
        _close_compilation_ffmpeg_and_check(
            proc, log_file, log_file_path, cmd, check_returncode=False
        )
        logger.error(
            "FFmpeg closed stdin (broken pipe). Command: %s. Check %s for stderr.",
            " ".join(cmd),
            log_file_path,
        )
        raise RuntimeError(
            f"FFmpeg broke pipe during encode; check {log_file_path!r} for details"
        ) from None
    except Exception as e:
        _close_compilation_ffmpeg_and_check(
            proc, log_file, log_file_path, cmd, check_returncode=False
        )
        logger.error(
            "Compilation encode failed while writing frames: %s. Command: %s. "
            "Check %s for stderr.",
            e,
            " ".join(cmd),
            log_file_path,
        )
        raise


def _resolve_clip_path(
    ce_dir: str, camera: str, resolve_clip_in_folder: object
) -> str:
    """Resolve clip path for camera under ce_dir; raise FileNotFoundError if missing."""
    cam_dir = os.path.join(ce_dir, camera)
    clip_name = (
        resolve_clip_in_folder(cam_dir)
        if callable(resolve_clip_in_folder)
        else None
    )
    if not clip_name:
        clip_name = f"{camera}.mp4"
    # Coerce to str for join() (resolve_clip_in_folder is object for mocks).
    clip_path = os.path.join(cam_dir, str(clip_name))
    if not os.path.isfile(clip_path):
        raise FileNotFoundError(f"Clip not found: {clip_path}")
    return clip_path


def _load_sidecars_for_cameras(
    ce_dir: str, cameras: list[str]
) -> tuple[dict[str, dict], dict[str, list[float]]]:
    """
    Load detection.json per camera; return (sidecar_cache, sidecar_timestamps).
    On missing file or parse error use fallback entries=[], native from constants
    so compilation still runs (center crop). Replicates failsafe from
    generate_compilation_video/compile_ce_video.
    """
    sidecar_cache: dict[str, dict] = {}
    sidecar_timestamps: dict[str, list[float]] = {}
    empty = {
        "entries": [],
        "native_width": COMPILATION_DEFAULT_NATIVE_WIDTH,
        "native_height": COMPILATION_DEFAULT_NATIVE_HEIGHT,
    }
    for cam in cameras:
        path = os.path.join(ce_dir, cam, "detection.json")
        if os.path.isfile(path):
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        entries = data.get("entries") or []
                        nw = (
                            int(data.get("native_width", 0) or 0)
                            or COMPILATION_DEFAULT_NATIVE_WIDTH
                        )
                        nh = (
                            int(data.get("native_height", 0) or 0)
                            or COMPILATION_DEFAULT_NATIVE_HEIGHT
                        )
                        sidecar_cache[cam] = {
                            "entries": entries,
                            "native_width": nw,
                            "native_height": nh,
                        }
                    else:
                        sidecar_cache[cam] = {
                            "entries": data or [],
                            "native_width": COMPILATION_DEFAULT_NATIVE_WIDTH,
                            "native_height": COMPILATION_DEFAULT_NATIVE_HEIGHT,
                        }
            except Exception as e:
                logger.error(
                    "Compilation fallback to center crop: error loading sidecar "
                    "for camera=%s path=%s error=%s; output will be static.",
                    cam,
                    path,
                    e,
                )
                sidecar_cache[cam] = dict(empty)
        else:
            logger.error(
                "Compilation fallback to center crop: sidecar missing for "
                "camera=%s path=%s; output will be static.",
                cam,
                path,
            )
            sidecar_cache[cam] = dict(empty)
        entries = sidecar_cache[cam].get("entries") or []
        sidecar_timestamps[cam] = sorted(
            float(e.get("timestamp_sec") or 0) for e in entries
        )
    return sidecar_cache, sidecar_timestamps


def _log_stutter_once(
    cam: str, clip_path: str, logged_set: set[str]
) -> None:
    """Log INFO once per camera for stutter/missing frames; add cam to logged_set."""
    if cam not in logged_set:
        logger.info(
            "Possible stutter or missing frames from %s, check original file "
            "for confirmation. path=%s",
            cam,
            clip_path,
        )
        logged_set.add(cam)


def _run_pynv_compilation(
    slices: list[dict],
    ce_dir: str,
    tmp_output_path: str,
    target_w: int,
    target_h: int,
    resolve_clip_in_folder: object,
    cuda_device_index: int = 0,
) -> None:
    """
    Decode each slice with PyNvVideoCodec (gpu_decoder); crop with smooth panning.
    Uses PTS-based frame selection (get_index_from_time_in_seconds) to reduce
    variable frame rate jitter. Frames streamed to FFmpeg stdin (no in-memory
    frame list) to avoid RAM spikes. Encode via FFmpeg h264_nvenc only (GPU);
    no CPU fallback. Output 20fps, no audio.
    """
    import torch

    if not slices:
        return

    proc, log_file_path, log_file, cmd = _open_compilation_ffmpeg_process(
        tmp_output_path, target_w, target_h
    )

    assert proc.stdin is not None
    logged_cameras: set[str] = set()
    logged_stutter_cameras: set[str] = set()  # One INFO per cam for stutter/missing.
    slices_per_cam = Counter(sl["camera"] for sl in slices)
    missing_crop_by_cam: dict[str, list[tuple[float, float]]] = defaultdict(list)
    try:
        for slice_idx, sl in enumerate(slices):
            cam = sl["camera"]
            clip_path = _resolve_clip_path(ce_dir, cam, resolve_clip_in_folder)
            t0 = sl["start_sec"]
            t1 = sl["end_sec"]
            duration = t1 - t0
            n_frames = max(1, round(duration * float(COMPILATION_OUTPUT_FPS)))
            if "crop_start" not in sl or "crop_end" not in sl:
                missing_crop_by_cam[cam].append((t0, t1))
            crop_start = sl.get("crop_start")
            crop_end = sl.get("crop_end")
            output_times = [
                t0 + i / float(COMPILATION_OUTPUT_FPS) for i in range(n_frames)
            ]

            # ffprobe outside GPU_LOCK so GPU not held during I/O (Finding 4.1).
            slice_meta = _get_video_metadata(clip_path)
            fallback_fps = (
                slice_meta[2] if slice_meta and slice_meta[2] > 0 else 30.0
            )
            fallback_duration = (
                slice_meta[3] if slice_meta else duration
            )

            batch_to_free: Any = None
            try:
                with GPU_LOCK:
                    with create_decoder(clip_path, gpu_id=cuda_device_index) as ctx:
                        frame_count = len(ctx)
                        if frame_count <= 0:
                            fps = fallback_fps
                            frame_count = max(1, int(fallback_duration * fps))
                        # PTS-based indices to reduce jitter (decoder time→index).
                        src_indices = [
                            min(
                                max(0, ctx.get_index_from_time_in_seconds(t)),
                                frame_count - 1,
                            )
                            for t in output_times
                        ]
                        if not src_indices:
                            continue
                        ih, iw = (
                            None,
                            None,
                        )  # Set from first chunk batch shape; crop params once.
                        # Type checker: start_cx/end_cx etc bound when ih is None.
                        start_cx = start_cy = end_cx = end_cy = 0.0
                        w_d = h_d = we_d = he_d = 1
                        for chunk_start in range(0, len(src_indices), BATCH_SIZE):
                            chunk_indices = src_indices[
                                chunk_start : chunk_start + BATCH_SIZE
                            ]
                            try:
                                batch = ctx.get_frames(chunk_indices)
                                batch_to_free = batch
                            except Exception as e:
                                logger.error(
                                    "%s (decoder get_frames failed). path=%s "
                                    "chunk_indices=%s error=%s",
                                    NVDEC_INIT_FAILURE_PREFIX,
                                    clip_path,
                                    chunk_indices,
                                    e,
                                    exc_info=True,
                                )
                                logger.warning(
                                    "Compilation: decoder get_frames failed for slice "
                                    "%s (%s): %s",
                                    slice_idx,
                                    clip_path,
                                    e,
                                )
                                if torch.cuda.is_available():
                                    try:
                                        logger.debug(
                                            "VRAM summary: %s",
                                            torch.cuda.memory_summary(),
                                        )
                                    except Exception:
                                        pass
                                break
                            if batch.shape[0] == 0:
                                logger.error(
                                    "Compilation: decoder returned 0 frames for chunk "
                                    "(slice %s). Skipping chunk. path=%s",
                                    slice_idx,
                                    clip_path,
                                )
                                del batch
                                batch_to_free = None
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                continue
                            # First chunk: set decoder dims and crop params once.
                            if ih is None:
                                _, _, ih, iw = batch.shape
                                sw = sl.get("native_width") or iw
                                sh = sl.get("native_height") or ih
                                scale_x = (iw / sw) if sw > 0 else 1.0
                                scale_y = (ih / sh) if sh > 0 else 1.0
                                if crop_start and crop_end:
                                    xs, ys, w, h = crop_start
                                    xe, ye, we, he = crop_end
                                    w_d = max(1, min(iw, int(w * scale_x)))
                                    h_d = max(1, min(ih, int(h * scale_y)))
                                    we_d = max(1, min(iw, int(we * scale_x)))
                                    he_d = max(1, min(ih, int(he * scale_y)))
                                    xs_d = xs * scale_x
                                    ys_d = ys * scale_y
                                    xe_d = xe * scale_x
                                    ye_d = ye * scale_y
                                    start_cx = xs_d + w_d / 2.0
                                    start_cy = ys_d + h_d / 2.0
                                    end_cx = xe_d + we_d / 2.0
                                    end_cy = ye_d + he_d / 2.0
                                else:
                                    start_cx = iw / 2.0
                                    start_cy = ih / 2.0
                                    end_cx = iw / 2.0
                                    end_cy = ih / 2.0
                                # Static-frame: same decoder index for all frames.
                                if len(src_indices) > 1 and len(set(src_indices)) == 1:
                                    logger.debug(
                                        "Compilation static frame: same frame index "
                                        "%s for all %s frames camera=%s "
                                        "slice [%.2f, %.2f]; check decoder/time.",
                                        src_indices[0],
                                        n_frames,
                                        cam,
                                        t0,
                                        t1,
                                    )
                                    if cam not in logged_stutter_cameras:
                                        _log_stutter_once(
                                            cam, clip_path, logged_stutter_cameras
                                        )
                                if cam not in logged_cameras:
                                    logger.debug(
                                        "Compilation camera=%s: frame %sx%s, "
                                        "crop center (%.0f,%.0f)->(%.0f,%.0f), "
                                        "target %sx%s, n_slices=%s, n_frames=%s",
                                        cam,
                                        iw,
                                        ih,
                                        start_cx,
                                        start_cy,
                                        end_cx,
                                        end_cy,
                                        target_w,
                                        target_h,
                                        slices_per_cam.get(cam, 0),
                                        n_frames,
                                    )
                                    logged_cameras.add(cam)
                            # Fewer frames than requested — repeat last frame.
                            if batch.shape[0] < len(chunk_indices):
                                logger.debug(
                                    "Compilation: decoder returned fewer frames than "
                                    "requested for chunk camera=%s slice [%.2f, %.2f] "
                                    "(%s of %s). Repeating last frame of chunk.",
                                    cam,
                                    t0,
                                    t1,
                                    batch.shape[0],
                                    len(chunk_indices),
                                )
                                if cam not in logged_stutter_cameras:
                                    _log_stutter_once(
                                        cam, clip_path, logged_stutter_cameras
                                    )
                            for j in range(len(chunk_indices)):
                                safe_j = min(j, batch.shape[0] - 1)
                                frame = batch[safe_j : safe_j + 1]
                                i = chunk_start + j
                                t = output_times[i]
                                t_progress = (
                                    (t - t0) / duration if duration > 1e-6 else 0.0
                                )
                                current_cx = start_cx + t_progress * (end_cx - start_cx)
                                current_cy = start_cy + t_progress * (end_cy - start_cy)
                                if crop_start and crop_end:
                                    current_w_d = max(
                                        2, int(w_d + t_progress * (we_d - w_d)) & ~1
                                    )
                                    current_h_d = max(
                                        2, int(h_d + t_progress * (he_d - h_d)) & ~1
                                    )
                                    current_w_d = min(
                                        iw if iw is not None else target_w,
                                        max(1, current_w_d),
                                    )
                                    current_h_d = min(
                                        ih if ih is not None else target_h,
                                        max(1, current_h_d),
                                    )
                                else:
                                    current_w_d = min(
                                        iw if iw is not None else target_w, target_w
                                    )
                                    current_h_d = min(
                                        ih if ih is not None else target_h, target_h
                                    )
                                current_cx_int = int(current_cx)
                                current_cy_int = int(current_cy)
                                cropped = crop_utils.crop_around_center_to_size(
                                    frame,
                                    current_cx_int,
                                    current_cy_int,
                                    current_w_d,
                                    current_h_d,
                                    target_w,
                                    target_h,
                                )
                                arr = cropped[0].permute(1, 2, 0).cpu().numpy()
                                proc.stdin.write(arr.tobytes())
                                proc.stdin.flush()
                            del batch
                            batch_to_free = None
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
            except Exception as e:
                logger.error(
                    "%s (decoder failed for slice %s). path=%s error=%s",
                    NVDEC_INIT_FAILURE_PREFIX,
                    slice_idx,
                    clip_path,
                    e,
                    exc_info=True,
                )
                logger.warning("Skipping slice %s (%s): %s", slice_idx, clip_path, e)
            finally:
                if batch_to_free is not None:
                    del batch_to_free
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        for cam, pairs in missing_crop_by_cam.items():
            n = len(pairs)
            first, last = pairs[0], pairs[-1]
            logger.error(
                "Compilation: slice missing crop_start/crop_end for camera=%s in %s "
                "slices (e.g. [%.2f, %.2f]–[%.2f, %.2f]); using fallback crop.",
                cam,
                n,
                first[0],
                first[1],
                last[0],
                last[1],
            )
    except BrokenPipeError:
        logger.error(
            "FFmpeg closed stdin (broken pipe). Command: %s. Check %s for stderr.",
            " ".join(cmd),
            log_file_path,
        )
        raise RuntimeError(
            f"FFmpeg broke pipe during encode; check {log_file_path!r} for details"
        ) from None
    except Exception as e:
        logger.error(
            "Compilation encode failed while writing frames: %s. Command: %s. "
            "Check %s for stderr.",
            e,
            " ".join(cmd),
            log_file_path,
        )
        raise
    finally:
        _close_compilation_ffmpeg_and_check(
            proc, log_file, log_file_path, cmd, check_returncode=False
        )
    if proc.returncode != 0:
        logger.error(
            "Compilation encode failed: FFmpeg exited with code %s. Command: %s. "
            "Check %s for full stderr.",
            proc.returncode,
            " ".join(cmd),
            log_file_path,
        )
        raise RuntimeError(
            f"FFmpeg h264_nvenc encode failed (exit {proc.returncode}); "
            f"check {log_file_path!r} for details"
        )


def generate_compilation_video(
    slices: list[dict],
    ce_dir: str,
    output_path: str,
    target_w: int = 1440,
    target_h: int = 1080,
    crop_smooth_alpha: float = 0.0,
    config: dict | None = None,
    sidecars: dict[str, dict] | None = None,
) -> None:
    """
    Concatenates slices into a final 20fps cropped video. Decode and crop via
    PyNvVideoCodec (gpu_decoder) and PyTorch; encode via FFmpeg h264_nvenc only
    (GPU; no CPU fallback). Smooth panning uses t/duration interpolation.
    Optional EMA smoothing of crop centers. No audio. When sidecars is provided
    (e.g. from compile_ce_video), detection.json is not read from disk.
    """
    logger.info(f"Starting compilation of {len(slices)} slices to {output_path}")

    from frigate_buffer.services.query import resolve_clip_in_folder

    # Load sidecar once per camera: use provided dict or read from disk
    sidecar_cache: dict[str, dict] = {}
    sidecar_timestamps: dict[str, list[float]] = {}
    if sidecars is not None:
        sidecar_cache = sidecars
        for sl in slices:
            cam = sl["camera"]
            if cam in sidecar_timestamps:
                continue
            entries = sidecar_cache.get(cam, {}).get("entries") or []
            sidecar_timestamps[cam] = sorted(
                float(e.get("timestamp_sec") or 0) for e in entries
            )
    else:
        cameras = list(dict.fromkeys(sl["camera"] for sl in slices))
        sidecar_cache, sidecar_timestamps = _load_sidecars_for_cameras(
            ce_dir, cameras
        )

    no_entries_by_cam: dict[str, list[tuple[float, float]]] = defaultdict(list)
    no_detections_by_cam: dict[str, list[tuple[float, float]]] = defaultdict(list)

    for i, sl in enumerate(slices):
        cam = sl["camera"]
        sidecar_data = sidecar_cache.get(cam) or {}
        sw = int(sidecar_data.get("native_width") or COMPILATION_DEFAULT_NATIVE_WIDTH)
        sh = int(sidecar_data.get("native_height") or COMPILATION_DEFAULT_NATIVE_HEIGHT)
        ts_sorted = sidecar_timestamps.get(cam)
        t0 = sl["start_sec"]
        t1 = sl["end_sec"]
        entries = sidecar_data.get("entries") or []
        # crop_start/crop_end are in native (sw, sh) space; stored on slice for
        # decoder-space scaling in _run_pynv_compilation.
        if not entries:
            no_entries_by_cam[cam].append((t0, t1))
        else:
            entry0 = _nearest_entry_at_t(entries, t0, ts_sorted)
            entry1 = _nearest_entry_at_t(entries, t1, ts_sorted)
            dets0 = (entry0 or {}).get("detections") or []
            dets1 = (entry1 or {}).get("detections") or []
            if not dets0 or not dets1:
                no_detections_by_cam[cam].append((t0, t1))
        tracking_target_frame_percent = (
            int(config.get("TRACKING_TARGET_FRAME_PERCENT", 40)) if config else 40
        )
        sl["crop_start"] = calculate_crop_at_time(
            sidecar_data,
            t0,
            sw,
            sh,
            target_w,
            target_h,
            timestamps_sorted=ts_sorted,
            tracking_target_frame_percent=tracking_target_frame_percent,
        )
        # Last slice of a camera run: hold crop (no pan to switch-time position)
        # to avoid panning away from the person at the cut.
        is_last_of_run = (i + 1 < len(slices)) and (slices[i + 1]["camera"] != cam)
        if is_last_of_run:
            sl["crop_end"] = sl["crop_start"]
        else:
            sl["crop_end"] = calculate_crop_at_time(
                sidecar_data,
                t1,
                sw,
                sh,
                target_w,
                target_h,
                timestamps_sorted=ts_sorted,
                tracking_target_frame_percent=tracking_target_frame_percent,
            )
        sl["native_width"] = sw
        sl["native_height"] = sh

    for cam, pairs in no_entries_by_cam.items():
        n = len(pairs)
        first, last = pairs[0], pairs[-1]
        logger.error(
            "Compilation: no sidecar entries for camera=%s in %s slices "
            "(e.g. [%.2f, %.2f]–[%.2f, %.2f]); using fallback crop.",
            cam,
            n,
            first[0],
            first[1],
            last[0],
            last[1],
        )
    for cam, pairs in no_detections_by_cam.items():
        n = len(pairs)
        logger.error(
            "Compilation: no detections at slice start/end for camera=%s in %s "
            "slices; using fallback crop (center or nearby detection within 5s).",
            cam,
            n,
        )

    zoom_smooth_alpha = (
        float(config.get("COMPILATION_ZOOM_SMOOTH_EMA_ALPHA", 0.25)) if config else 0.25
    )
    if zoom_smooth_alpha > 0:
        smooth_zoom_ema(slices, zoom_smooth_alpha, target_w, target_h)

    if crop_smooth_alpha > 0:
        smooth_crop_centers_ema(slices, crop_smooth_alpha)

    cuda_device_index = int(config.get("CUDA_DEVICE_INDEX", 0)) if config else 0
    temp_path = output_path.replace(".mp4", "_temp.mp4")
    if not temp_path.endswith(".mp4"):
        temp_path = temp_path + ".mp4"
    try:
        _run_pynv_compilation(
            slices=slices,
            ce_dir=ce_dir,
            tmp_output_path=temp_path,
            target_w=target_w,
            target_h=target_h,
            resolve_clip_in_folder=resolve_clip_in_folder,
            cuda_device_index=cuda_device_index,
        )
        if os.path.isfile(temp_path) and os.path.getsize(temp_path) > 0:
            os.rename(temp_path, output_path)
            logger.info(f"Successfully compiled summary video to {output_path}")
        else:
            raise FileNotFoundError(
                f"Compiler finished but tmp result file was empty or missing: "
                f"{temp_path}"
            )
    except Exception as e:
        log_path = os.path.join(ce_dir, "compilation_error.log")
        try:
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(str(e))
        except OSError:
            pass
        logger.error(f"Compilation failed: {e}")
        raise


def compile_ce_video(
    ce_dir: str,
    global_end: float,
    config: dict,
    primary_camera: str | None = None,
) -> str | None:
    """
    High-level orchestrator for video compilation. Scans the CE directory,
    parses detection sidecars, uses timeline_ema to establish segment assignments,
    and then generates the hardware-accelerated summary video. Returns the path
    to the compiled video if successful, None otherwise.
    """
    logger.info(f"Starting compilation process for {ce_dir}")

    # 1. Gather sidecars once (entries, native_*) for timeline and compilation
    from frigate_buffer.services.query import resolve_clip_in_folder

    cameras = []
    try:
        with os.scandir(ce_dir) as it:
            for entry in it:
                if entry.is_dir() and not entry.name.startswith("."):
                    cam = entry.name
                    clip = resolve_clip_in_folder(entry.path)
                    if clip:
                        cameras.append(cam)
    except OSError:
        pass

    if not cameras:
        logger.warning(f"No cameras found with clips in {ce_dir}")
        return None

    sidecar_cache, _ = _load_sidecars_for_cameras(ce_dir, cameras)

    if not sidecar_cache:
        logger.warning(
            f"No sidecars available in {ce_dir} for compilation."
        )
        return None

    # Use actual clip/sidecar duration when longer than requested;
    # summary covers full content (avoids truncation when export window shorter).
    actual_duration_sec = 0.0
    for cam in sidecar_cache:
        entries = sidecar_cache[cam].get("entries") or []
        if entries:
            last_ts = float(entries[-1].get("timestamp_sec") or 0)
            actual_duration_sec = max(actual_duration_sec, last_ts)
    if actual_duration_sec > global_end:
        logger.info(
            "Compilation: extending timeline from %.1fs to %.1fs (sidecar content "
            "longer than requested window).",
            global_end,
            actual_duration_sec,
        )
    global_end = max(global_end, actual_duration_sec)

    native_sizes = {
        cam: (sidecar_cache[cam]["native_width"], sidecar_cache[cam]["native_height"])
        for cam in sidecar_cache
    }

    # Helper to calculate area of person targets at t_sec
    def _person_area_at_time(cam_name: str, t_sec: float) -> float:
        entries = (sidecar_cache.get(cam_name) or {}).get("entries") or []
        if not entries:
            return 0.0
        # Find nearest
        nearest = min(
            entries,
            key=lambda e: abs((e.get("timestamp_sec") or 0) - t_sec),
        )
        area: float = 0.0
        for d in nearest.get("detections") or []:
            if (d.get("label") or "").lower() in ("person", "people", "pedestrian"):
                area += float(d.get("area") or 0)
        return area

    # 2. Compute timeline using same config as frame timeline (AI prompt)
    step_sec = float(config.get("MAX_MULTI_CAM_FRAMES_SEC", 2))
    if step_sec <= 0:
        step_sec = 2.0
    max_frames_min = int(config.get("MAX_MULTI_CAM_FRAMES_MIN", 45))
    multiplier = float(config.get("CAMERA_TIMELINE_ANALYSIS_MULTIPLIER", 2.0))

    dense_times = timeline_ema.build_dense_times(
        step_sec, max_frames_min, multiplier, global_end
    )

    assignments = timeline_ema.build_phase1_assignments(
        dense_times,
        cameras,
        _person_area_at_time,
        native_sizes,
        ema_alpha=float(config.get("CAMERA_TIMELINE_EMA_ALPHA", 0.4)),
        primary_bias_multiplier=float(
            config.get("CAMERA_TIMELINE_PRIMARY_BIAS_MULTIPLIER", 1.2)
        ),
        primary_camera=primary_camera,
        hysteresis_margin=float(config.get("CAMERA_SWITCH_HYSTERESIS_MARGIN", 1.15)),
        min_segment_frames=int(
            config.get("CAMERA_SWITCH_MIN_SEGMENT_FRAMES", 5)
        ),
    )

    if not assignments:
        logger.warning("No camera assignments could be generated for compilation.")
        return None

    logger.debug(f"Generated {len(assignments)} timeline points via EMA.")

    # 3. One slice per assignment; trim to action window (first/last detection ± roll).
    slices = assignments_to_slices(assignments, global_end)
    sidecars_entries = {cam: sidecar_cache[cam]["entries"] for cam in sidecar_cache}
    slices = _trim_slices_to_action_window(slices, sidecars_entries, global_end)
    if not slices:
        logger.warning(
            "No slices remain after trimming to action window; skipping compilation."
        )
        return None

    out_name = os.path.basename(os.path.abspath(ce_dir)) + "_summary.mp4"
    output_path = os.path.join(ce_dir, out_name)
    crop_smooth_alpha = float(
        config.get("COMPILATION_CROP_SMOOTH_EMA_ALPHA", 0.0)
    )

    try:
        generate_compilation_video(
            slices,
            ce_dir,
            output_path,
            target_w=int(config.get("SUMMARY_TARGET_WIDTH", 1440)),
            target_h=int(config.get("SUMMARY_TARGET_HEIGHT", 1080)),
            crop_smooth_alpha=crop_smooth_alpha,
            config=config,
            sidecars=sidecar_cache,
        )
        return output_path
    except Exception as e:
        logger.error(f"Failed to generate summary video: {e}")
        return None
