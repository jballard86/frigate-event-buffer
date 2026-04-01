#pragma once

#include <torch/extension.h>

#include <cstdint>
#include <optional>
#include <string>

struct AVFrame;

namespace intel_xpu_zerocopy {

struct ZeroCopyResult {
  torch::Tensor tensor;
  bool used_zero_copy{false};
  std::string debug;
};

/**
 * Attempt a true zero-copy decode conversion for Intel:
 * - Map an FFmpeg hw frame to DRM PRIME (NV12) to get DMA-BUF fds.
 * - Import the DMA-BUF memory into the Intel GPU runtime.
 * - Convert NV12 -> RGB BCHW uint8 on device.
 *
 * Returns std::nullopt when the fast path is not available for this frame.
 *
 * Notes:
 * - This code is compiled only when INTEL_DECODE_XPU_ZEROCOPY is enabled.
 * - The returned tensor (when present) is always on XPU.
 */
std::optional<ZeroCopyResult> try_nv12_drmprime_to_bchw_rgb_xpu(
    AVFrame* src_hw_frame,
    int device_index,
    int width,
    int height);

}  // namespace intel_xpu_zerocopy

