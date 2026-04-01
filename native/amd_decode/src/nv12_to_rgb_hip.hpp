#pragma once

#include <cstdint>

#include <hip/hip_runtime_api.h>

namespace frigate_amd_decode {

/** Launch NV12 (device linear Y + interleaved UV) -> uint8 BCHW RGB on HIP. */
hipError_t launch_nv12_to_bchw_rgb(
    const uint8_t* y_plane,
    int y_pitch,
    const uint8_t* uv_plane,
    int uv_pitch,
    uint8_t* out_bchw,
    int width,
    int height,
    int device_index,
    hipStream_t stream);

}  // namespace frigate_amd_decode
