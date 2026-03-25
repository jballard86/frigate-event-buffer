"""NVIDIA CUDA runtime: diagnostics, cache, and device strings for decode/inference."""

from __future__ import annotations

import logging
import subprocess

logger = logging.getLogger("frigate-buffer")


def tensor_device_for_decode(gpu_id: int = 0) -> str:
    """Torch device string for VRAM tensors from NVDEC (matches PyNv default)."""
    return f"cuda:{gpu_id}"


def log_gpu_status() -> None:
    """At startup: log GPU visibility for NVDEC troubleshooting (nvidia-smi)."""
    nvidia_smi = __import__("shutil").which("nvidia-smi")
    if nvidia_smi:
        try:
            proc = subprocess.run(
                [nvidia_smi, "--query-gpu=count", "--format=csv,noheader"],
                capture_output=True,
                timeout=5,
            )
            count_str = (proc.stdout or b"").decode("utf-8", errors="replace").strip()
            gpu_count = count_str.split("\n")[0] if count_str else "?"
            proc2 = subprocess.run(
                [nvidia_smi, "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True,
                timeout=5,
            )
            driver = (proc2.stdout or b"").decode(
                "utf-8", errors="replace"
            ).strip().split("\n")[0] or "?"
            logger.info(
                "GPU status: nvidia-smi OK, GPUs=%s, driver=%s (NVDEC for decode)",
                gpu_count,
                driver,
            )
        except (subprocess.TimeoutExpired, OSError) as e:
            logger.warning(
                "nvidia-smi found but failed: %s; container may not have GPU access.",
                e,
            )
    else:
        logger.info("nvidia-smi not found; NVDEC decode requires GPU.")


def empty_cache() -> None:
    """Release cached CUDA allocations when the decoder releases large batches."""
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def memory_summary(*, abbreviated: bool = False) -> str | None:
    """Return CUDA memory summary for debug logs, or None if CUDA unavailable."""
    import torch

    if not torch.cuda.is_available():
        return None
    return torch.cuda.memory_summary(abbreviated=abbreviated)


def default_detection_device(config: dict) -> str | None:
    """DETECTION_DEVICE if set, else cuda:N from GPU index, else None."""
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


class NvidiaRuntime:
    """Concrete :class:`GpuRuntimeProto` for CUDA (delegates to torch.cuda)."""

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


nvidia_runtime = NvidiaRuntime()
