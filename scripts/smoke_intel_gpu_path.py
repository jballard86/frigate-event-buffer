#!/usr/bin/env python3
"""Optional Arc / Intel path smoke: torch, frigate_intel_decode, app runtime helpers.

Run from repo root after building native/intel_decode (see
native/intel_decode/README.md) or inside Dockerfile.intel. Use ``--strict`` in CI
when the extension must be present.

Why: catches libtorch ABI skew and missing LD_LIBRARY_PATH before full app startup.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys


def _prepend_src_to_path() -> None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = os.path.join(root, "src")
    if os.path.isdir(src) and src not in sys.path:
        sys.path.insert(0, src)


def _run_vainfo(*, strict_dri: bool) -> int:
    """Print ``vainfo --display drm`` output.

    Returns 3 if strict_dri and the probe failed.
    """
    vainfo_bin = shutil.which("vainfo")
    if not vainfo_bin:
        print("vainfo: not found on PATH (install vainfo in image or on host)")
        return 3 if strict_dri else 0
    try:
        proc = subprocess.run(
            [vainfo_bin, "--display", "drm"],
            capture_output=True,
            timeout=15,
            text=True,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"vainfo: failed to run: {exc}")
        return 3 if strict_dri else 0
    combined = (proc.stdout or "") + (proc.stderr or "")
    limit = 12000
    if len(combined) <= limit:
        print(combined)
    else:
        print(combined[:limit] + "\n... [truncated]")
    if proc.returncode != 0:
        print(f"vainfo: exit code {proc.returncode}")
        return 3 if strict_dri else 0
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Smoke: torch, frigate_intel_decode, intel runtime helpers.",
    )
    parser.add_argument(
        "clip",
        nargs="?",
        default=None,
        help=(
            "Optional H.264/HEVC clip; decodes first frame via native QSV "
            "(frigate_intel_decode)."
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit 2 if frigate_intel_decode cannot be imported.",
    )
    parser.add_argument(
        "--vainfo",
        action="store_true",
        help="Run vainfo --display drm first (VA-API / driver visibility).",
    )
    parser.add_argument(
        "--strict-dri",
        action="store_true",
        help="With --vainfo, exit 3 if vainfo is missing or non-zero (hardware check).",
    )
    args = parser.parse_args()

    if args.strict_dri and not args.vainfo:
        print("error: --strict-dri requires --vainfo", file=sys.stderr)
        return 2

    if args.vainfo:
        dri_rc = _run_vainfo(strict_dri=args.strict_dri)
        if dri_rc != 0:
            return dri_rc

    import torch

    print(f"torch {torch.__version__}")

    try:
        import frigate_intel_decode as native
    except ImportError as exc:
        msg = f"frigate_intel_decode import failed (build native/intel_decode): {exc}"
        print(msg)
        return 2 if args.strict else 0

    print(f"frigate_intel_decode {native.version()}")

    _prepend_src_to_path()
    try:
        from frigate_buffer.services.gpu_backends.intel.runtime import (
            default_detection_device,
            tensor_device_for_decode,
        )
    except ModuleNotFoundError as exc:
        print(f"frigate_buffer not importable ({exc}); skipping app runtime checks")
        return 0

    cfg: dict = {"GPU_DEVICE_INDEX": 0}
    print(f"tensor_device_for_decode(0)={tensor_device_for_decode(0)!r}")
    print(f"default_detection_device(cfg)={default_detection_device(cfg)!r}")

    if args.clip:
        t = native.decode_first_frame_bchw_rgb(args.clip)
        print(f"first_frame {tuple(t.shape)} {t.dtype}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
