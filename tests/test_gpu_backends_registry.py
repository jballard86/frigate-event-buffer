"""Tests for gpu_backends registry (get_gpu_backend cache, multi-vendor)."""

from __future__ import annotations

import threading

import pytest

from frigate_buffer.services.gpu_backends.registry import (
    clear_gpu_backend_cache,
    get_gpu_backend,
)
from frigate_buffer.services.gpu_backends.types import GpuBackend


@pytest.fixture(autouse=True)
def _clear_backend_cache():
    clear_gpu_backend_cache()
    yield
    clear_gpu_backend_cache()


def test_get_gpu_backend_default_vendor_is_nvidia() -> None:
    backend = get_gpu_backend({})
    assert isinstance(backend, GpuBackend)


def test_get_gpu_backend_explicit_nvidia() -> None:
    backend = get_gpu_backend({"GPU_VENDOR": "nvidia"})
    assert isinstance(backend, GpuBackend)


def test_get_gpu_backend_normalizes_vendor_case() -> None:
    b1 = get_gpu_backend({"GPU_VENDOR": "NVIDIA"})
    b2 = get_gpu_backend({"GPU_VENDOR": "nvidia"})
    assert b1 is b2


def test_get_gpu_backend_unsupported_vendor_raises() -> None:
    with pytest.raises(ValueError, match="not supported"):
        get_gpu_backend({"GPU_VENDOR": "matrox"})


def test_get_gpu_backend_returns_same_cached_instance() -> None:
    cfg = {"GPU_VENDOR": "nvidia", "GPU_DEVICE_INDEX": 0}
    a = get_gpu_backend(cfg)
    b = get_gpu_backend({"GPU_VENDOR": "nvidia", "GPU_DEVICE_INDEX": 1})
    assert a is b


def test_get_gpu_backend_intel_returns_distinct_backend() -> None:
    clear_gpu_backend_cache()
    n = get_gpu_backend({"GPU_VENDOR": "nvidia"})
    clear_gpu_backend_cache()
    i = get_gpu_backend({"GPU_VENDOR": "intel"})
    assert i.create_decoder is not n.create_decoder
    assert i.runtime is not n.runtime


def test_get_gpu_backend_intel_exposes_intel_ffmpeg_helpers() -> None:
    from frigate_buffer.services.gpu_backends.intel.ffmpeg_encode import (
        intel_ffmpeg_compilation_encode,
    )
    from frigate_buffer.services.gpu_backends.intel.gif_ffmpeg import intel_gif_ffmpeg

    clear_gpu_backend_cache()
    b = get_gpu_backend({"GPU_VENDOR": "intel"})
    assert b.ffmpeg_compilation_encode is intel_ffmpeg_compilation_encode
    assert b.gif_ffmpeg is intel_gif_ffmpeg


def test_get_gpu_backend_amd_returns_distinct_backend() -> None:
    clear_gpu_backend_cache()
    n = get_gpu_backend({"GPU_VENDOR": "nvidia"})
    clear_gpu_backend_cache()
    a = get_gpu_backend({"GPU_VENDOR": "amd"})
    assert a.create_decoder is not n.create_decoder
    assert a.runtime is not n.runtime


def test_get_gpu_backend_amd_exposes_amd_ffmpeg_helpers() -> None:
    from frigate_buffer.services.gpu_backends.amd.ffmpeg_encode import (
        amd_ffmpeg_compilation_encode,
    )
    from frigate_buffer.services.gpu_backends.amd.gif_ffmpeg import amd_gif_ffmpeg

    clear_gpu_backend_cache()
    b = get_gpu_backend({"GPU_VENDOR": "amd"})
    assert b.ffmpeg_compilation_encode is amd_ffmpeg_compilation_encode
    assert b.gif_ffmpeg is amd_gif_ffmpeg


def test_get_gpu_backend_switches_vendor_when_config_changes() -> None:
    from frigate_buffer.services.gpu_backends.nvidia import create_decoder as nvidia_cd

    clear_gpu_backend_cache()
    a = get_gpu_backend({"GPU_VENDOR": "intel"})
    b = get_gpu_backend({"GPU_VENDOR": "nvidia"})
    assert a is not b
    assert b.create_decoder is nvidia_cd


def test_gpu_backend_exposes_nvidia_components() -> None:
    from frigate_buffer.services.gpu_backends.nvidia import (
        create_decoder,
        nvidia_ffmpeg_compilation_encode,
        nvidia_gif_ffmpeg,
        nvidia_runtime,
    )

    b = get_gpu_backend({})
    assert b.create_decoder is create_decoder
    assert b.runtime is nvidia_runtime
    assert b.ffmpeg_compilation_encode is nvidia_ffmpeg_compilation_encode
    assert b.gif_ffmpeg is nvidia_gif_ffmpeg


def test_clear_gpu_backend_cache_forces_new_instance() -> None:
    a = get_gpu_backend({})
    clear_gpu_backend_cache()
    b = get_gpu_backend({})
    assert a is not b


def test_registry_create_decoder_matches_gpu_decoder_shim() -> None:
    """Stable entrypoint: public gpu_decoder re-export matches registry bundle."""
    from frigate_buffer.services import gpu_decoder

    b = get_gpu_backend({})
    assert b.create_decoder is gpu_decoder.create_decoder


@pytest.mark.parametrize(
    "vendor_cfg",
    [
        {},
        {"GPU_VENDOR": None},
        {"GPU_VENDOR": ""},
        {"GPU_VENDOR": "   "},
    ],
)
def test_get_gpu_backend_blank_or_missing_vendor_is_nvidia(
    vendor_cfg: dict,
) -> None:
    backend = get_gpu_backend(vendor_cfg)
    assert isinstance(backend, GpuBackend)


def test_concurrent_get_gpu_backend_returns_same_singleton() -> None:
    """Concurrent first calls must yield one backend (double-checked locking)."""
    clear_gpu_backend_cache()
    holders: list[GpuBackend | None] = [None, None]
    barrier = threading.Barrier(2)

    def run(idx: int) -> None:
        barrier.wait()
        holders[idx] = get_gpu_backend({})

    t0 = threading.Thread(target=run, args=(0,))
    t1 = threading.Thread(target=run, args=(1,))
    t0.start()
    t1.start()
    t0.join()
    t1.join()
    assert holders[0] is not None
    assert holders[0] is holders[1]
