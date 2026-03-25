"""
Shim for GPU decode: re-exports NVIDIA PyNvVideoCodec implementation.

Implementation lives in ``gpu_backends.nvidia.decoder`` (sole PyNv import site).
Decode must run under ``GPU_LOCK`` from ``gpu_backends.lock``.
"""

from __future__ import annotations

from frigate_buffer.services.gpu_backends.nvidia.decoder import (
    DECODER_MAX_WIDTH,
    DecoderContext,
    _create_simple_decoder,
    create_decoder,
)

__all__ = [
    "DECODER_MAX_WIDTH",
    "DecoderContext",
    "_create_simple_decoder",
    "create_decoder",
]
