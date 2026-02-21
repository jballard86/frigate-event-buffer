"""
Video compilation service.
Handles segment-level video processing using hardware acceleration.
Generates single, stitched, cropped compilation videos.
"""

from __future__ import annotations

import os
import json
import logging
import subprocess

from frigate_buffer.services import timeline_ema

logger = logging.getLogger("frigate-buffer")

def convert_timeline_to_segments(timeline_points: list[tuple[float, str]], global_end: float) -> list[dict]:
    """
    Groups continuous time steps for the same camera into segments.
    global_end is required, but if for some reason it is 0 or None, it falls back
    to the last point + the step interval of the first two points.
    """
    segments = []
    if not timeline_points:
        return segments

    if not global_end:
        step = 1.0
        if len(timeline_points) > 1:
            step = timeline_points[1][0] - timeline_points[0][0]
        global_end = timeline_points[-1][0] + step

    current_cam = timeline_points[0][1]
    seg_start = timeline_points[0][0]

    for i in range(1, len(timeline_points)):
        t, cam = timeline_points[i]
        if cam != current_cam:
            segments.append({
                "camera": current_cam,
                "start_sec": seg_start,
                "end_sec": t
            })
            current_cam = cam
            seg_start = t

    segments.append({
        "camera": current_cam,
        "start_sec": seg_start,
        "end_sec": global_end
    })

    return segments

def calculate_segment_crop(
    segment: dict,
    sidecar_data: dict,
    source_width: int,
    source_height: int,
    target_w: int = 1440,
    target_h: int = 1080
) -> tuple[int, int, int, int]:
    """
    Finds the center of mass for a segment based on sidecar detections
    and returns clamped (x, y, w, h) crop variables.
    """
    start_sec = segment["start_sec"]
    end_sec = segment["end_sec"]

    # Prevent target from being larger than source; simulate letterbox or scale down
    if source_width < target_w or source_height < target_h:
        scale = min(source_width / target_w, source_height / target_h)
        target_w = int(target_w * scale)
        target_h = int(target_h * scale)

    weighted_cx_sum: float = 0.0
    weighted_cy_sum: float = 0.0
    total_area: float = 0.0

    for entry in sidecar_data.get("entries", []):
        t = entry.get("timestamp_sec", 0.0)
        # We consider inclusive start, exclusive end for frame gathering typically
        if start_sec <= t < end_sec:
            for det in entry.get("detections", []):
                area = float(det.get("area", 1.0))
                if area <= 0:
                    area = 1.0
                    
                cp = det.get("centerpoint")
                if isinstance(cp, (list, tuple)) and len(cp) >= 2:
                    weighted_cx_sum += float(cp[0]) * area
                    weighted_cy_sum += float(cp[1]) * area
                    total_area += area
                else:
                    box = det.get("box") or det.get("bbox")
                    if isinstance(box, (list, tuple)) and len(box) >= 4:
                        cx = (float(box[0]) + float(box[2])) / 2.0
                        cy = (float(box[1]) + float(box[3])) / 2.0
                        weighted_cx_sum += cx * area
                        weighted_cy_sum += cy * area
                        total_area += area

    if total_area > 0:
        # EMA simulation using area-weighted generic mass tracker
        avg_cx = weighted_cx_sum / total_area
        avg_cy = weighted_cy_sum / total_area
    else:
        # Fallback to center if no detections
        avg_cx = source_width / 2.0
        avg_cy = source_height / 2.0

    # Calculate top-left for the crop box
    x = int(avg_cx - target_w / 2.0)
    y = int(avg_cy - target_h / 2.0)

    # Clamp the coordinates to ensure the crop box doesn't exceed video boundaries
    x = max(0, min(source_width - target_w, x))
    y = max(0, min(source_height - target_h, y))

    return (x, y, target_w, target_h)

def generate_compilation_video(
    segments: list[dict], 
    ce_dir: str, 
    output_path: str,
    target_w: int = 1440,
    target_h: int = 1080
) -> None:
    """
    Concatenates segments into a final 20fps 4:3 cropped video without audio using `-hwaccel cuda`.
    Dynamically builds FFmpeg filtergraph.
    """
    logger.info(f"Starting compilation of {len(segments)} segments to {output_path}")

    from frigate_buffer.services.query import resolve_clip_in_folder

    inputs = []
    filter_complex = []
    
    tmp_output_path = output_path + ".tmp"
    
    # Calculate crop logic, load sidecars, gather inputs
    for i, seg in enumerate(segments):
        cam = seg["camera"]
        cam_dir = os.path.join(ce_dir, cam)
        clip_name = resolve_clip_in_folder(cam_dir)
        if not clip_name:
            clip_name = f"{cam}.mp4"
        clip_path = os.path.join(cam_dir, clip_name)
        sidecar_path = os.path.join(cam_dir, "detection.json")
        
        # Omit hardware acceleration prefix to prevent context-overflow memory issues.
        # We instead configure it globally on the wrapper command or format.
        inputs.extend(["-i", clip_path])
            
        sidecar_data = {}
        if os.path.isfile(sidecar_path):
            try:
                with open(sidecar_path, "r", encoding="utf-8") as f:
                    sidecar_data = json.load(f)
            except Exception as e:
                logger.error(f"Error loading sidecar for {cam}: {e}")
                
        sw = sidecar_data.get("native_width", 1920)
        sh = sidecar_data.get("native_height", 1080)
        
        x, y, w, h = calculate_segment_crop(seg, sidecar_data, sw, sh, target_w, target_h)
        seg["crop"] = (x, y, w, h)
        seg["input_idx"] = i
        
    logger.debug(f"Calculated segments for compilation: {segments}")
    
    concat_inputs = []
    for i, seg in enumerate(segments):
        idx = seg["input_idx"]
        start = seg["start_sec"]
        end = seg["end_sec"]
        x, y, w, h = seg["crop"]
        
        # filter chain: trim -> setpts -> fps -> crop
        fg = f"[{idx}:v]trim=start={start}:end={end},setpts=PTS-STARTPTS,fps=20,crop={w}:{h}:{x}:{y}[v{i}]"
        filter_complex.append(fg)
        concat_inputs.append(f"[v{i}]")
        
    concat_filter = "".join(concat_inputs) + f"concat=n={len(segments)}:v=1:a=0[outv]"
    filter_complex.append(concat_filter)
    
    fg_str = ";".join(filter_complex)
    logger.debug(f"Compilation FFmpeg filter_complex:\n{fg_str}")
    
    cmd = ["ffmpeg", "-y", "-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
    cmd.extend(inputs)
    cmd.extend([
        "-filter_complex", fg_str,
        "-map", "[outv]",
        "-an",
        "-c:v", "h264_nvenc",
        "-pix_fmt", "yuv420p",
        tmp_output_path
    ])
    
    logger.debug(f"Compilation raw FFmpeg command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if os.path.isfile(tmp_output_path) and os.path.getsize(tmp_output_path) > 0:
            os.rename(tmp_output_path, output_path)
            logger.info(f"Compilation finished successfully. Output saved to: {output_path}")
        else:
            raise FileNotFoundError(f"Compiler terminated effectively, but tmp result file was empty or missing: {tmp_output_path}")
    except subprocess.CalledProcessError as e:
        ffmpeg_log_path = os.path.join(ce_dir, "ffmpeg_compilation_error.log")
        with open(ffmpeg_log_path, "w", encoding="utf-8") as f:
            f.write(e.stderr or "")
        logger.error(f"Compilation FFmpeg failed! Check {ffmpeg_log_path} for raw output. Process error:\n{e.stderr}")
        raise
    except Exception as general_error:
        logger.error(f"Compilation failed unexpectedly: {general_error}")
        raise

def compile_ce_video(ce_dir: str, global_end: float, config: dict, primary_camera: str | None = None) -> str | None:
    """
    High-level orchestrator for video compilation.
    Scans the CE directory, parses detection sidecars, uses timeline_ema to establish segment assignments, 
    and then generates the hardware-accelerated summary video.
    Returns the path to the compiled video if successful, None otherwise.
    """
    logger.info(f"Starting compilation process for {ce_dir}")
    
    # 1. Gather sidecars and native sizes
    sidecars = {}
    native_sizes = {}
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
        
    for cam in cameras:
        path = os.path.join(ce_dir, cam, "detection.json")
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        sidecars[cam] = data.get("entries") or []
                        nw = int(data.get("native_width", 0) or 0)
                        nh = int(data.get("native_height", 0) or 0)
                        native_sizes[cam] = (nw, nh)
                    elif isinstance(data, list):
                        sidecars[cam] = data
                        native_sizes[cam] = (0, 0)
            except Exception as e:
                logger.error(f"Failed to load sidecar for {cam}: {e}")
                
    if not sidecars:
        logger.warning(f"No sidecars available in {ce_dir} for compilation.")
        return None
        
    # Helper to calculate area of person targets at t_sec
    def _person_area_at_time(cam_name: str, t_sec: float) -> float:
        entries = sidecars.get(cam_name) or []
        if not entries:
            return 0.0
        # Find nearest
        nearest = min(entries, key=lambda e: abs((e.get("timestamp_sec") or 0) - t_sec))
        area: float = 0.0
        for d in nearest.get("detections") or []:
            if (d.get("label") or "").lower() in ("person", "people", "pedestrian"):
                area += float(d.get("area") or 0)
        return area
        
    # 2. Compute timeline using EMA logic
    step_sec = float(config.get("MAX_FRAMES_SEC", 1.0))
    if step_sec <= 0:
        step_sec = 1.0
    max_frames_min = int(config.get("MULTI_CAM_MAX_FRAMES_MIN", 30))
    multiplier = float(config.get("CAMERA_TIMELINE_ANALYSIS_MULTIPLIER", 2.0))
    
    dense_times = timeline_ema.build_dense_times(step_sec, max_frames_min, multiplier, global_end)
    
    assignments = timeline_ema.build_phase1_assignments(
        dense_times,
        cameras,
        _person_area_at_time,
        native_sizes,
        ema_alpha=float(config.get("CAMERA_TIMELINE_EMA_ALPHA", 0.4)),
        primary_bias_multiplier=float(config.get("CAMERA_TIMELINE_PRIMARY_BIAS_MULTIPLIER", 1.2)),
        primary_camera=primary_camera,
        hysteresis_margin=float(config.get("CAMERA_SWITCH_HYSTERESIS_MARGIN", 1.15)),
        min_segment_frames=int(config.get("CAMERA_SWITCH_MIN_SEGMENT_FRAMES", 5)),
    )
    
    if not assignments:
        logger.warning("No camera assignments could be generated for compilation.")
        return None
        
    logger.debug(f"Generated {len(assignments)} timeline points via EMA.")
    
    # 3. Convert to segments and generate
    segments = convert_timeline_to_segments(assignments, global_end)
    out_name = os.path.basename(os.path.abspath(ce_dir)) + "_summary.mp4"
    output_path = os.path.join(ce_dir, out_name)
    
    try:
        generate_compilation_video(
            segments, 
            ce_dir, 
            output_path, 
            target_w=int(config.get("SUMMARY_TARGET_WIDTH", 1440)), 
            target_h=int(config.get("SUMMARY_TARGET_HEIGHT", 1080))
        )
        return output_path
    except Exception as e:
        logger.error(f"Failed to generate summary video: {e}")
        return None
