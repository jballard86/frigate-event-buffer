"""Load and cache the active :class:`GpuBackend` from merged config."""

from __future__ import annotations

import threading

from frigate_buffer.services.gpu_backends.types import GpuBackend

_lock = threading.Lock()
_cached: GpuBackend | None = None


def _normalized_vendor(config: dict) -> str:
    raw = config.get("GPU_VENDOR") or "nvidia"
    if isinstance(raw, str):
        s = raw.strip().lower()
        return s if s else "nvidia"
    return "nvidia"


def get_gpu_backend(config: dict) -> GpuBackend:
    """Return the process-wide backend for ``GPU_VENDOR`` (cached after first call).

    Only ``nvidia`` is implemented; other values raise :class:`ValueError`.
    Config is normally validated in :func:`frigate_buffer.config.load_config`;
    this function re-checks so tests and partial dicts fail consistently.
    """
    vendor = _normalized_vendor(config)
    if vendor != "nvidia":
        raise ValueError(
            f"GPU_VENDOR={vendor!r} is not supported; only 'nvidia' is available "
            "(see docs/Multi_GPU_Support_Integration_Plan/)."
        )
    global _cached
    if _cached is not None:
        return _cached
    with _lock:
        if _cached is None:
            from frigate_buffer.services.gpu_backends.nvidia import build_nvidia_backend

            _cached = build_nvidia_backend()
        return _cached


def clear_gpu_backend_cache() -> None:
    """Drop cached backend (tests or hot-reload experiments only)."""
    global _cached
    with _lock:
        _cached = None
