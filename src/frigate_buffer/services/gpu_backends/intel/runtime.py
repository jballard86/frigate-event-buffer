"""Intel Arc / XPU runtime: diagnostics, cache, device strings."""

from __future__ import annotations

import logging
import shutil
import subprocess

logger = logging.getLogger("frigate-buffer")


def tensor_device_for_decode(gpu_id: int = 0) -> str:
    """Torch device for tensors after native CPU decode; XPU when IPEX exposes it."""
    try:
        import torch

        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return f"xpu:{gpu_id}"
    except Exception:
        pass
    return "cpu"


def log_gpu_status() -> None:
    """Log Intel GPU visibility hints (vainfo when present)."""
    vainfo = shutil.which("vainfo")
    if vainfo:
        try:
            proc = subprocess.run(
                [vainfo, "--display", "drm"],
                capture_output=True,
                timeout=5,
            )
            out = (proc.stdout or b"").decode("utf-8", errors="replace")
            if "Driver version" in out or "VAProfile" in out:
                logger.info(
                    "GPU status: vainfo OK (Intel/VAAPI; QSV via frigate_intel_decode)"
                )
                return
        except (subprocess.TimeoutExpired, OSError) as e:
            logger.warning("vainfo failed: %s", e)
    logger.info(
        "vainfo missing or inconclusive; Intel QSV needs drivers and FFmpeg QSV."
    )


def empty_cache() -> None:
    """Release XPU cache when available (no-op on CPU-only torch)."""
    try:
        import torch

        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.empty_cache()
    except Exception:
        pass


def memory_summary(*, abbreviated: bool = False) -> str | None:
    """Return XPU memory summary when available."""
    try:
        import torch

        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return torch.xpu.memory_summary(abbreviated=abbreviated)
    except Exception:
        pass
    return None


def default_detection_device(config: dict) -> str | None:
    """DETECTION_DEVICE if set, else xpu:N when available, else None."""
    raw = (config.get("DETECTION_DEVICE") or "").strip()
    if raw:
        return raw
    try:
        import torch

        if hasattr(torch, "xpu") and torch.xpu.is_available():
            idx = int(
                config.get(
                    "GPU_DEVICE_INDEX",
                    config.get("CUDA_DEVICE_INDEX", 0),
                )
            )
            return f"xpu:{idx}"
    except Exception:
        pass
    return None


class IntelRuntime:
    """Concrete :class:`GpuRuntimeProto` for Intel XPU / CPU fallback."""

    def log_gpu_status(self) -> None:
        log_gpu_status()

    def empty_cache(self) -> None:
        empty_cache()

    def memory_summary(self, *, abbreviated: bool = False) -> str | None:
        return memory_summary(abbreviated=abbreviated)

    def tensor_device_for_decode(self, gpu_id: int = 0) -> str:
        return tensor_device_for_decode(gpu_id)

    def default_detection_device(self, config: dict) -> str | None:
        return default_detection_device(config)


intel_runtime = IntelRuntime()
