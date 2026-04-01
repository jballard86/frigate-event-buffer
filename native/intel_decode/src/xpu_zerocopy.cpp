// Experimental Intel XPU zero-copy decode path.
//
// This module is intentionally Intel-only and compiled only when
// INTEL_DECODE_XPU_ZEROCOPY=ON. On failure the caller surfaces a decode error
// (no host readback fallback for QSV frames in that build).

#include "xpu_zerocopy.hpp"

#include <cstdlib>
#include <cstring>
#include <stdexcept>

extern "C" {
#include <libavutil/hwcontext_drm.h>
}

// DPC++ / SYCL headers (provided by intel-oneapi-compiler-dpcpp-cpp).
#include <sycl/sycl.hpp>

// Level Zero headers (provided by intel level-zero runtime/dev packages).
// The exact external-memory import structs are backend-provided; this include
// is used for context/device interop and external memory import chaining.
#include <level_zero/ze_api.h>

namespace intel_xpu_zerocopy {

namespace {

constexpr uint32_t kDrmFormatNv12 = 0x3231564e;

// Best-effort external memory import of a DMA-BUF FD into a Level Zero device allocation.
// Returns nullptr on failure.
//
// NOTE: This requires external memory support in the Level Zero runtime.
static void* import_dmabuf_fd_to_ze_device_ptr(
    ze_context_handle_t ctx,
    ze_device_handle_t dev,
    int dma_buf_fd,
    size_t size_bytes) {
  if (dma_buf_fd < 0 || size_bytes == 0) {
    return nullptr;
  }

  ze_device_mem_alloc_desc_t dev_desc{};
  dev_desc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
  dev_desc.pNext = nullptr;
  dev_desc.flags = 0;
  dev_desc.ordinal = 0;

  // External memory import extension (dma-buf FD).
  // Many runtimes expose ze_external_memory_import_fd_t; if the runtime headers
  // do not, this feature cannot be built.
  ze_external_memory_import_fd_t import_desc{};
  import_desc.stype = ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMPORT_FD;
  import_desc.pNext = nullptr;
  import_desc.flags = ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF;
  import_desc.fd = dma_buf_fd;

  dev_desc.pNext = &import_desc;

  void* ptr = nullptr;
  const ze_result_t r = zeMemAllocDevice(ctx, &dev_desc, size_bytes, /*alignment=*/1, dev, &ptr);
  if (r != ZE_RESULT_SUCCESS) {
    return nullptr;
  }
  return ptr;
}

static sycl::device pick_sycl_gpu_device_by_index(int device_index) {
  std::vector<sycl::device> gpus = sycl::device::get_devices(sycl::info::device_type::gpu);
  if (gpus.empty()) {
    throw std::runtime_error("SYCL: no GPU devices available");
  }
  const int idx = std::max(0, std::min<int>(device_index, static_cast<int>(gpus.size()) - 1));
  return gpus.at(static_cast<size_t>(idx));
}

static torch::Tensor nv12_usm_to_rgb_bchw_xpu(
    sycl::queue& q,
    const uint8_t* y_plane,
    const uint8_t* uv_plane,
    int y_pitch,
    int uv_pitch,
    int width,
    int height,
    int device_index) {
  auto opts = torch::TensorOptions().dtype(torch::kUInt8).device(torch::Device(torch::kXPU, device_index));
  torch::Tensor out = torch::empty({1, 3, height, width}, opts);

  uint8_t* out_ptr = reinterpret_cast<uint8_t*>(out.data_ptr<uint8_t>());
  const int w = width;
  const int h = height;

  // Kernel: NV12 -> RGB (full range BT.601-ish). Keep it simple; correctness over micro-opts.
  q.submit([&](sycl::handler& cgh) {
     cgh.parallel_for(sycl::range<2>(static_cast<size_t>(h), static_cast<size_t>(w)), [=](sycl::id<2> id) {
       const int y = static_cast<int>(id[0]);
       const int x = static_cast<int>(id[1]);

       const int yv = static_cast<int>(y_plane[y * y_pitch + x]);
       const int uv_x = (x & ~1);
       const int uv_y = (y / 2);
       const uint8_t u8 = uv_plane[uv_y * uv_pitch + uv_x + 0];
       const uint8_t v8 = uv_plane[uv_y * uv_pitch + uv_x + 1];

       const int c = yv - 16;
       const int d = static_cast<int>(u8) - 128;
       const int e = static_cast<int>(v8) - 128;

       auto clamp_u8 = [](int v) -> uint8_t {
         if (v < 0) return 0;
         if (v > 255) return 255;
         return static_cast<uint8_t>(v);
       };

       const int r = (298 * c + 409 * e + 128) >> 8;
       const int g = (298 * c - 100 * d - 208 * e + 128) >> 8;
       const int b = (298 * c + 516 * d + 128) >> 8;

       // BCHW layout: [0,0,y,x] contiguous.
       const size_t base = static_cast<size_t>(y) * static_cast<size_t>(w) + static_cast<size_t>(x);
       out_ptr[0ull * static_cast<size_t>(h) * static_cast<size_t>(w) + base] = clamp_u8(r);
       out_ptr[1ull * static_cast<size_t>(h) * static_cast<size_t>(w) + base] = clamp_u8(g);
       out_ptr[2ull * static_cast<size_t>(h) * static_cast<size_t>(w) + base] = clamp_u8(b);
     });
   }).wait_and_throw();

  return out;
}

}  // namespace

std::optional<ZeroCopyResult> try_nv12_drmprime_to_bchw_rgb_xpu(
    AVFrame* src_hw_frame,
    int device_index,
    int width,
    int height) {
  (void)width;
  (void)height;

  if (src_hw_frame == nullptr) {
    return std::nullopt;
  }

  if (src_hw_frame->data[0] == nullptr) {
    // Not a DRM_PRIME frame; caller must map it first.
    return std::nullopt;
  }

  auto* desc = reinterpret_cast<AVDRMFrameDescriptor*>(src_hw_frame->data[0]);
  if (desc == nullptr || desc->nb_layers < 1 || desc->nb_objects < 1) {
    return std::nullopt;
  }
  const AVDRMLayerDescriptor& layer0 = desc->layers[0];
  if (layer0.nb_planes < 2) {
    return std::nullopt;
  }
  if (layer0.format != static_cast<int>(kDrmFormatNv12)) {
    return std::nullopt;
  }

  const int obj_y = layer0.planes[0].object_index;
  const int obj_uv = layer0.planes[1].object_index;
  if (obj_y < 0 || obj_uv < 0 || obj_y >= desc->nb_objects || obj_uv >= desc->nb_objects) {
    return std::nullopt;
  }

  const int fd_y = desc->objects[obj_y].fd;
  const int fd_uv = desc->objects[obj_uv].fd;
  if (fd_y < 0 || fd_uv < 0) {
    return std::nullopt;
  }

  // Sizes are per-object; for NV12 they are often the same object for both planes.
  const size_t size_y = static_cast<size_t>(desc->objects[obj_y].size);
  const size_t size_uv = static_cast<size_t>(desc->objects[obj_uv].size);

  try {
    // Pick a SYCL device and obtain Level Zero interop handles.
    const sycl::device dev = pick_sycl_gpu_device_by_index(device_index);
    sycl::context ctx(dev);
    sycl::queue q(ctx, dev);

    const auto ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx);
    const auto ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(dev);

    void* ze_ptr_y = import_dmabuf_fd_to_ze_device_ptr(ze_ctx, ze_dev, fd_y, size_y);
    void* ze_ptr_uv = import_dmabuf_fd_to_ze_device_ptr(ze_ctx, ze_dev, fd_uv, size_uv);
    if (ze_ptr_y == nullptr || ze_ptr_uv == nullptr) {
      if (ze_ptr_y) zeMemFree(ze_ctx, ze_ptr_y);
      if (ze_ptr_uv) zeMemFree(ze_ctx, ze_ptr_uv);
      return std::nullopt;
    }

    const auto* y_plane = reinterpret_cast<const uint8_t*>(reinterpret_cast<uintptr_t>(ze_ptr_y) +
                                                          static_cast<uintptr_t>(layer0.planes[0].offset));
    const auto* uv_plane = reinterpret_cast<const uint8_t*>(reinterpret_cast<uintptr_t>(ze_ptr_uv) +
                                                           static_cast<uintptr_t>(layer0.planes[1].offset));

    const int y_pitch = static_cast<int>(layer0.planes[0].pitch);
    const int uv_pitch = static_cast<int>(layer0.planes[1].pitch);

    // Width/height come from the mapped DRM frame.
    const int w = src_hw_frame->width;
    const int h = src_hw_frame->height;
    if (w <= 0 || h <= 0) {
      zeMemFree(ze_ctx, ze_ptr_y);
      zeMemFree(ze_ctx, ze_ptr_uv);
      return std::nullopt;
    }

    torch::Tensor out = nv12_usm_to_rgb_bchw_xpu(q, y_plane, uv_plane, y_pitch, uv_pitch, w, h, device_index);

    // Free imported allocations; the output tensor owns its own storage.
    zeMemFree(ze_ctx, ze_ptr_y);
    zeMemFree(ze_ctx, ze_ptr_uv);

    ZeroCopyResult res;
    res.tensor = out;
    res.used_zero_copy = true;
    res.debug = "xpu_zerocopy_ok";
    return res;
  } catch (const std::exception&) {
    return std::nullopt;
  }
}

}  // namespace intel_xpu_zerocopy

