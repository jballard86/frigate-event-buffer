#!/usr/bin/env python3
"""Optional AMD ROCm smoke: ROCm PyTorch tensor + app AMD runtime + native decode probe.

Run on Linux with ROCm and a ROCm ``torch`` wheel (``torch.version.hip`` set) for full
checks. ``--strict-native`` alone verifies ``frigate_amd_decode`` import
(e.g. no ``/dev/kfd``).
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys


def _prepend_src_to_path() -> None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = os.path.join(root, "src")
    if os.path.isdir(src) and src not in sys.path:
        sys.path.insert(0, src)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Smoke: ROCm torch tensor + frigate_buffer amd runtime.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Exit 2 if ROCm GPU path is not usable (no CUDA or not a ROCm torch build)."
        ),
    )
    parser.add_argument(
        "--strict-native",
        action="store_true",
        help="Exit 2 if frigate_amd_decode cannot be imported.",
    )
    parser.add_argument(
        "--yolo",
        action="store_true",
        help="Import ultralytics and print its version (no model download).",
    )
    parser.add_argument(
        "clip",
        nargs="?",
        default=None,
        help="Optional video path; first frame via native when import succeeds.",
    )
    args = parser.parse_args()

    import torch

    print(f"torch {torch.__version__}")
    hip = getattr(torch.version, "hip", None)
    print(f"torch.version.hip={hip!r}")

    rocm_path_ok = bool(torch.cuda.is_available() and hip is not None)
    if not rocm_path_ok:
        if not torch.cuda.is_available():
            print(
                "skip: torch.cuda.is_available() is False (no ROCm/CUDA device visible)"
            )
        else:
            print(
                "skip: torch.version.hip is None "
                "(NVIDIA CUDA build, not a ROCm PyTorch wheel)"
            )
        if args.strict:
            return 2
    else:
        device = torch.device("cuda:0")
        tensor = torch.zeros((1, 3, 64, 64), dtype=torch.uint8, device=device)
        print(f"tensor ok shape={tuple(tensor.shape)} device={tensor.device}")

    if args.yolo:
        try:
            import ultralytics
        except ImportError as exc:
            print(f"ultralytics import failed: {exc}")
            return 2 if args.strict else 0
        ver = getattr(ultralytics, "__version__", "unknown")
        print(f"ultralytics {ver}")

    _prepend_src_to_path()
    try:
        from frigate_buffer.services.gpu_backends.amd.runtime import (
            default_detection_device,
            tensor_device_for_decode,
        )
    except ModuleNotFoundError as exc:
        print(f"frigate_buffer not importable ({exc}); skipping app runtime checks")
        if args.strict_native:
            try:
                importlib.import_module("frigate_amd_decode")
            except ImportError:
                return 2
        return 0

    cfg: dict = {"GPU_DEVICE_INDEX": 0}
    print(f"tensor_device_for_decode(0)={tensor_device_for_decode(0)!r}")
    print(f"default_detection_device(cfg)={default_detection_device(cfg)!r}")

    native_mod = None
    try:
        native_mod = importlib.import_module("frigate_amd_decode")
    except ImportError as exc:
        print(f"frigate_amd_decode: import failed: {exc}")
        if args.strict_native:
            return 2
    else:
        print("frigate_amd_decode: import ok")

    if args.clip:
        if native_mod is None:
            print(
                "error: clip path set but frigate_amd_decode is not available",
                file=sys.stderr,
            )
            return 2
        t = native_mod.decode_first_frame_bchw_rgb(args.clip)
        print(f"first_frame shape={tuple(t.shape)} dtype={t.dtype}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
