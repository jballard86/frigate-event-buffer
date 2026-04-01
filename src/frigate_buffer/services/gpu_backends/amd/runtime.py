"""AMD ROCm runtime: CUDA-compatible torch API, diagnostics, cache."""

from __future__ import annotations

import logging
import shutil
import subprocess

logger = logging.getLogger("frigate-buffer")


def tensor_device_for_decode(gpu_id: int = 0) -> str:
    """Torch device for decoded frames; ROCm builds expose GPUs as ``cuda:N``."""
    try:
        import torch

        if torch.cuda.is_available():
            return f"cuda:{int(gpu_id)}"
    except Exception:
        pass
    return "cpu"


def log_gpu_status() -> None:
    """Log ROCm / VAAPI hints when CLI tools are present (no hard failure)."""
    rocm_smi = shutil.which("rocm-smi")
    if rocm_smi:
        try:
            proc = subprocess.run(
                [rocm_smi, "--showid"],
                capture_output=True,
                timeout=5,
            )
            out = (proc.stdout or b"").decode("utf-8", errors="replace")
            if "GPU" in out or "Card" in out:
                logger.info("GPU status: rocm-smi responded (ROCm stack present)")
                return
        except (subprocess.TimeoutExpired, OSError) as e:
            logger.warning("rocm-smi failed: %s", e)
    vainfo = shutil.which("vainfo")
    if vainfo:
        try:
            proc = subprocess.run(
                [vainfo, "--display", "drm"],
                capture_output=True,
                timeout=5,
            )
            out = (proc.stdout or b"").decode("utf-8", errors="replace")
            if "VAProfile" in out:
                logger.info(
                    "GPU status: vainfo OK (VAAPI; decode via frigate_amd_decode)"
                )
                return
        except (subprocess.TimeoutExpired, OSError) as e:
            logger.warning("vainfo failed: %s", e)
    logger.info(
        "AMD GPU: rocm-smi/vainfo missing or inconclusive; "
        "ROCm + drivers needed for decode (see docs/INSTALL.md)."
    )


def empty_cache() -> None:
    """Release ROCm device cache when ``torch.cuda`` is available."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def memory_summary(*, abbreviated: bool = False) -> str | None:
    """Return CUDA memory summary (ROCm uses the same API)."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.memory_summary(abbreviated=abbreviated)
    except Exception:
        pass
    return None


def default_detection_device(config: dict) -> str | None:
    """DETECTION_DEVICE if set, else ``cuda:N`` when a GPU is visible."""
    raw = (config.get("DETECTION_DEVICE") or "").strip()
    if raw:
        return raw
    try:
        import torch

        if torch.cuda.is_available():
            idx = int(
                config.get(
                    "GPU_DEVICE_INDEX",
                    config.get("CUDA_DEVICE_INDEX", 0),
                )
            )
            return f"cuda:{idx}"
    except Exception:
        pass
    return None


class AmdRuntime:
    """Concrete :class:`GpuRuntimeProto` for ROCm (``cuda:`` device strings)."""

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


amd_runtime = AmdRuntime()
