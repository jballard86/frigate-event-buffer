"""Load and cache the active :class:`GpuBackend` from merged config."""

from __future__ import annotations

import threading

from frigate_buffer.services.gpu_backends.types import GpuBackend

_lock = threading.Lock()
_cached: GpuBackend | None = None
_cached_vendor: str | None = None


def _normalized_vendor(config: dict) -> str:
    raw = config.get("GPU_VENDOR") or "nvidia"
    if isinstance(raw, str):
        s = raw.strip().lower()
        return s if s else "nvidia"
    return "nvidia"


def get_gpu_backend(config: dict) -> GpuBackend:
    """Return the process-wide backend for ``GPU_VENDOR`` (cached after first call).

    ``nvidia``, ``intel``, and ``amd`` are implemented; ``intel`` requires
    ``frigate_intel_decode``; ``amd`` requires ``frigate_amd_decode`` when
    decoding (see ``native/`` trees and gpu-03 plan).
    Config is normally validated in :func:`frigate_buffer.config.load_config`;
    this function re-checks so tests and partial dicts fail consistently.
    """
    vendor = _normalized_vendor(config)
    global _cached, _cached_vendor
    if _cached is not None and _cached_vendor == vendor:
        return _cached
    with _lock:
        if _cached is not None and _cached_vendor == vendor:
            return _cached
        if vendor == "nvidia":
            from frigate_buffer.services.gpu_backends.nvidia import build_nvidia_backend

            _cached = build_nvidia_backend()
        elif vendor == "intel":
            from frigate_buffer.services.gpu_backends.intel import build_intel_backend

            _cached = build_intel_backend()
        elif vendor == "amd":
            from frigate_buffer.services.gpu_backends.amd import build_amd_backend

            _cached = build_amd_backend()
        else:
            raise ValueError(
                f"GPU_VENDOR={vendor!r} is not supported; use 'nvidia', 'intel', or "
                "'amd' (see docs/Multi_GPU_Support_Integration_Plan/)."
            )
        _cached_vendor = vendor
        return _cached


def clear_gpu_backend_cache() -> None:
    """Drop cached backend (tests or hot-reload experiments only)."""
    global _cached, _cached_vendor
    with _lock:
        _cached = None
        _cached_vendor = None
