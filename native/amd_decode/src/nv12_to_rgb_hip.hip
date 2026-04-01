/**
 * HIP kernel: NV12 (linear) -> uint8 BCHW RGB (contiguous, batch 1).
 * BT.709 for height >= 720, else BT.601 (limited range).
 */

#include "nv12_to_rgb_hip.hpp"

#include <hip/hip_runtime.h>

namespace frigate_amd_decode {

__device__ __forceinline__ void yuv_to_rgb_bt601_lr(
    float y, float u, float v, uint8_t* r, uint8_t* g, uint8_t* b) {
  float c = y - 16.0f;
  float d = u - 128.0f;
  float e = v - 128.0f;
  float rf = (298.0f * c + 409.0f * e + 128.0f) / 256.0f;
  float gf = (298.0f * c - 100.0f * d - 208.0f * e + 128.0f) / 256.0f;
  float bf = (298.0f * c + 516.0f * d + 128.0f) / 256.0f;
  rf = fminf(255.0f, fmaxf(0.0f, rf));
  gf = fminf(255.0f, fmaxf(0.0f, gf));
  bf = fminf(255.0f, fmaxf(0.0f, bf));
  *r = static_cast<uint8_t>(rf);
  *g = static_cast<uint8_t>(gf);
  *b = static_cast<uint8_t>(bf);
}

__device__ __forceinline__ void yuv_to_rgb_bt709_lr(
    float y, float u, float v, uint8_t* r, uint8_t* g, uint8_t* b) {
  float c = y - 16.0f;
  float d = u - 128.0f;
  float e = v - 128.0f;
  float rf = (298.0f * c + 459.0f * e + 128.0f) / 256.0f;
  float gf = (298.0f * c - 55.0f * d - 137.0f * e + 128.0f) / 256.0f;
  float bf = (298.0f * c + 541.0f * d + 128.0f) / 256.0f;
  rf = fminf(255.0f, fmaxf(0.0f, rf));
  gf = fminf(255.0f, fmaxf(0.0f, gf));
  bf = fminf(255.0f, fmaxf(0.0f, bf));
  *r = static_cast<uint8_t>(rf);
  *g = static_cast<uint8_t>(gf);
  *b = static_cast<uint8_t>(bf);
}

__global__ void nv12_to_bchw_rgb_kernel(
    const uint8_t* y_plane,
    int y_pitch,
    const uint8_t* uv_plane,
    int uv_pitch,
    uint8_t* out_bchw,
    int width,
    int height,
    int use_bt709) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int yy = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || yy >= height) {
    return;
  }
  const uint8_t yv =
      y_plane[static_cast<size_t>(yy) * static_cast<size_t>(y_pitch) + static_cast<size_t>(x)];
  const int uv_row = yy / 2;
  const int uv_col = (x / 2) * 2;
  const size_t uv_off =
      static_cast<size_t>(uv_row) * static_cast<size_t>(uv_pitch) + static_cast<size_t>(uv_col);
  const uint8_t uu = uv_plane[uv_off];
  const uint8_t vv = uv_plane[uv_off + 1u];
  uint8_t rr = 0;
  uint8_t gg = 0;
  uint8_t bb = 0;
  if (use_bt709) {
    yuv_to_rgb_bt709_lr(static_cast<float>(yv), static_cast<float>(uu), static_cast<float>(vv), &rr, &gg, &bb);
  } else {
    yuv_to_rgb_bt601_lr(static_cast<float>(yv), static_cast<float>(uu), static_cast<float>(vv), &rr, &gg, &bb);
  }
  const size_t plane = static_cast<size_t>(width) * static_cast<size_t>(height);
  const size_t idx = static_cast<size_t>(yy) * static_cast<size_t>(width) + static_cast<size_t>(x);
  out_bchw[idx] = rr;
  out_bchw[plane + idx] = gg;
  out_bchw[2u * plane + idx] = bb;
}

hipError_t launch_nv12_to_bchw_rgb(
    const uint8_t* y_plane,
    int y_pitch,
    const uint8_t* uv_plane,
    int uv_pitch,
    uint8_t* out_bchw,
    int width,
    int height,
    int device_index,
    hipStream_t stream) {
  hipError_t e = hipSetDevice(device_index);
  if (e != hipSuccess) {
    return e;
  }
  const int use_bt709 = height >= 720 ? 1 : 0;
  dim3 block(16, 16);
  dim3 grid((static_cast<unsigned>(width) + 15u) / 16u, (static_cast<unsigned>(height) + 15u) / 16u);
  hipLaunchKernelGGL(
      nv12_to_bchw_rgb_kernel,
      grid,
      block,
      0,
      stream,
      y_plane,
      y_pitch,
      uv_plane,
      uv_pitch,
      out_bchw,
      width,
      height,
      use_bt709);
  return hipGetLastError();
}

}  // namespace frigate_amd_decode
