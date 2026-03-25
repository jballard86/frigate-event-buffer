"""Vendor-specific GPU backends (decode, runtime, FFmpeg helpers)."""

from frigate_buffer.services.gpu_backends.lock import GPU_LOCK
from frigate_buffer.services.gpu_backends.registry import (
    clear_gpu_backend_cache,
    get_gpu_backend,
)

__all__ = ["GPU_LOCK", "clear_gpu_backend_cache", "get_gpu_backend"]
