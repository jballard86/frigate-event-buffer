"""Voluptuous fragment for YAML ``multi_cam``."""

from __future__ import annotations

from voluptuous import Any, Optional

MULTI_CAM_SCHEMA = {
    Optional(
        "max_multi_cam_frames_min"
    ): int,  # Maximum frames to extract per clip (cap).
    Optional("max_multi_cam_frames_sec"): Any(
        int, float
    ),  # Target interval in seconds between frames (e.g. 1, 0.5).
    Optional("crop_width"): int,  # Width of output crop for stitched/multi-cam frames.
    Optional(
        "crop_height"
    ): int,  # Height of output crop for stitched/multi-cam frames.
    Optional(
        "multi_cam_system_prompt_file"
    ): str,  # Path to system prompt for multi-cam Gemini; empty = built-in.
    Optional("smart_crop_padding"): Any(
        int, float
    ),  # Padding fraction around motion-based crop (e.g. 0.15).
    Optional(
        "detection_model"
    ): str,  # Ultralytics model for detection sidecar (e.g. yolov8n.pt).
    Optional("detection_device"): str,  # Device for detection (e.g. cuda:0, cpu).
    Optional("detection_frame_interval"): int,  # Run YOLO every N frames; default 5.
    Optional(
        "detection_imgsz"
    ): int,  # YOLO inference size (higher = better small objects); default 640.
    # Timeline / EMA (Phase 1 grid + EMA + hysteresis + merge)
    Optional("camera_timeline_analysis_multiplier"): Any(
        int, float
    ),  # Denser analysis grid vs base step (2 or 3); default 2.
    Optional("camera_timeline_ema_alpha"): Any(
        int, float
    ),  # EMA smoothing factor 0–1; default 0.4.
    Optional("camera_timeline_primary_bias_multiplier"): Any(
        int, float
    ),  # Weight on primary camera area curve; default 1.2.
    Optional(
        "camera_switch_min_segment_frames"
    ): int,  # Min frames per segment; short runs merged; default 5.
    Optional("camera_switch_hysteresis_margin"): Any(
        int, float
    ),  # New camera must exceed current by this factor; default 1.15.
    Optional(
        "camera_timeline_final_yolo_drop_no_person"
    ): bool,  # If true drop frames with no person after Phase 2 YOLO.
    Optional(
        "decode_second_camera_cpu_only"
    ): bool,  # If true, use CPU decode for 2nd+ cams (contention workaround).
    Optional(
        "log_extraction_phase_timing"
    ): bool,  # Log elapsed time per extraction phase for debugging.
    Optional(
        "merge_frame_timeout_sec"
    ): int,  # Timeout (sec) when merge waits for camera frame; default 10.
    Optional(
        "tracking_target_frame_percent"
    ): int,  # Person area >= this % => full-frame resize; default 40.
    Optional(
        "person_area_debug"
    ): bool,  # Draw person area (px²) on frame bottom-right when true.
    Optional("compilation_zoom_smooth_ema_alpha"): Any(
        int, float
    ),  # EMA alpha for compilation zoom smoothing (0–1); default 0.25.
    # Intel QSV compilation encode tuning (gpu-02 Phase 7); optional.
    Optional("intel"): {
        Optional("qsv_encode_preset"): str,
        Optional("qsv_encode_global_quality"): int,
    },
    # GPU backend (multi-vendor prep; registry uses these in gpu-01+).
    Optional("gpu_vendor"): str,  # nvidia | intel | amd (see registry and native/*/).
    Optional("gpu_device_index"): int,  # Adapter index for decode/runtime (default 0).
    Optional(
        "cuda_device_index"
    ): int,  # Deprecated: use gpu_device_index; still accepted for legacy YAML.
}
