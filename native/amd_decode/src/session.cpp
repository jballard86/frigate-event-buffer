#include "session.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_drm.h>
#include <libavutil/imgutils.h>
#include <libavutil/mathematics.h>
#include <libavutil/pixfmt.h>
#include <libswscale/swscale.h>
}

#if defined(AMD_DECODE_WITH_HIP) && AMD_DECODE_WITH_HIP
#include <hip/hip_runtime_api.h>

#include "nv12_to_rgb_hip.hpp"
#endif

namespace {

#if defined(AMD_DECODE_WITH_HIP) && AMD_DECODE_WITH_HIP
/** Destroys imported DMA-BUF handles when leaving the VAAPI→DRM→HIP conversion scope. */
struct HipImportedMemories {
  std::vector<hipExternalMemory_t> handles;
  ~HipImportedMemories() {
    for (hipExternalMemory_t e : handles) {
      if (e != nullptr) {
        (void)hipDestroyExternalMemory(e);
      }
    }
  }
};
#endif

std::string av_error_string(int errnum) {
  char buf[AV_ERROR_MAX_STRING_SIZE];
  if (av_strerror(errnum, buf, sizeof(buf)) != 0) {
    return "unknown FFmpeg error " + std::to_string(errnum);
  }
  return std::string(buf);
}

struct PacketDeleter {
  void operator()(AVPacket* p) const { av_packet_free(&p); }
};

struct FrameDeleter {
  void operator()(AVFrame* p) const { av_frame_free(&p); }
};

struct SwsCtxDeleter {
  void operator()(SwsContext* p) const { sws_freeContext(p); }
};

enum AVPixelFormat amd_vaapi_get_format(AVCodecContext* /*ctx*/, const enum AVPixelFormat* pix_fmts) {
  while (*pix_fmts != AV_PIX_FMT_NONE) {
    if (*pix_fmts == AV_PIX_FMT_VAAPI) {
      return *pix_fmts;
    }
    ++pix_fmts;
  }
  return AV_PIX_FMT_NONE;
}

std::string vaapi_device_from_index(int device_index) {
  const char* env = std::getenv("FRIGATE_AMD_VAAPI_DEVICE");
  if (env != nullptr && env[0] != '\0') {
    return std::string(env);
  }
  const int base = 128;
  const int n = base + std::max(0, device_index);
  return "/dev/dri/renderD" + std::to_string(n);
}

/** DRM_FORMAT_NV12 from linux/drm_fourcc.h (avoid hard dependency on libdrm headers). */
constexpr uint32_t kDrmFormatNv12 = 0x3231564e;

}  // namespace

AmdDecoderSession::AmdDecoderSession(const std::string& path, int device_index)
    : path_(path), device_index_(device_index) {
  const char* force_sw = std::getenv("FRIGATE_AMD_DECODE_FORCE_SW");
  const bool prefer_vaapi = !(force_sw != nullptr && force_sw[0] == '1');
  open_demuxer_(path);
  setup_decoder_(prefer_vaapi);
  if (using_vaapi_) {
    init_drm_map_context_();
#if defined(AMD_DECODE_WITH_HIP) && AMD_DECODE_WITH_HIP
    int n = 0;
    if (drm_frames_ctx_ != nullptr && hipGetDeviceCount(&n) == hipSuccess && n > 0 &&
        device_index_ >= 0 && device_index_ < n) {
      zero_copy_ready_ = true;
    }
#endif
  }
  compute_frame_count_();
  if (video_width_ > 4096) {
    throw std::runtime_error("video width exceeds DECODER_MAX_WIDTH 4096 (AMD decode)");
  }
  seek_to_timestamp_(0);
  flush_decoder_();
  seq_cursor_ = 0;
}

AmdDecoderSession::~AmdDecoderSession() {
  if (dec_ctx_) {
    avcodec_free_context(&dec_ctx_);
  }
  if (hw_device_ctx_) {
    av_buffer_unref(&hw_device_ctx_);
  }
  if (drm_frames_ctx_) {
    av_buffer_unref(&drm_frames_ctx_);
  }
  if (drm_device_ctx_) {
    av_buffer_unref(&drm_device_ctx_);
  }
  if (fmt_) {
    avformat_close_input(&fmt_);
  }
}

void AmdDecoderSession::open_demuxer_(const std::string& path) {
  int r = avformat_open_input(&fmt_, path.c_str(), nullptr, nullptr);
  if (r < 0) {
    throw std::runtime_error("avformat_open_input: " + av_error_string(r));
  }
  if (avformat_find_stream_info(fmt_, nullptr) < 0) {
    throw std::runtime_error("avformat_find_stream_info failed");
  }
  for (unsigned i = 0; i < fmt_->nb_streams; ++i) {
    if (fmt_->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
      video_stream_idx_ = static_cast<int>(i);
      video_st_ = fmt_->streams[i];
      break;
    }
  }
  if (video_stream_idx_ < 0 || video_st_ == nullptr) {
    throw std::runtime_error("no video stream in file");
  }
  AVCodecParameters* par = video_st_->codecpar;
  video_width_ = par->width;
  video_height_ = par->height;
}

void AmdDecoderSession::init_drm_map_context_() {
  std::string dev = vaapi_device_from_index(device_index_);
  int rr = av_hwdevice_ctx_create(&drm_device_ctx_, AV_HWDEVICE_TYPE_DRM, dev.c_str(), nullptr, 0);
  if (rr < 0) {
    drm_device_ctx_ = nullptr;
    return;
  }
  drm_frames_ctx_ = av_hwframe_ctx_alloc(drm_device_ctx_);
  if (drm_frames_ctx_ == nullptr) {
    av_buffer_unref(&drm_device_ctx_);
    drm_device_ctx_ = nullptr;
    return;
  }
  auto* fc = reinterpret_cast<AVHWFramesContext*>(drm_frames_ctx_->data);
  fc->format = AV_PIX_FMT_DRM_PRIME;
  fc->sw_format = AV_PIX_FMT_NV12;
  fc->width = video_width_;
  fc->height = video_height_;
  if (av_hwframe_ctx_init(drm_frames_ctx_) < 0) {
    av_buffer_unref(&drm_frames_ctx_);
    av_buffer_unref(&drm_device_ctx_);
    drm_frames_ctx_ = nullptr;
    drm_device_ctx_ = nullptr;
  }
}

void AmdDecoderSession::setup_decoder_(bool prefer_vaapi) {
  AVCodecParameters* par = video_st_->codecpar;
  const AVCodec* dec = avcodec_find_decoder(par->codec_id);
  if (dec == nullptr) {
    throw std::runtime_error("no suitable FFmpeg decoder for video stream");
  }

  bool have_vaapi = false;
  if (prefer_vaapi) {
    std::string dev = vaapi_device_from_index(device_index_);
    int hwr = av_hwdevice_ctx_create(
        &hw_device_ctx_, AV_HWDEVICE_TYPE_VAAPI, dev.c_str(), nullptr, 0);
    if (hwr >= 0) {
      have_vaapi = true;
    } else {
      av_buffer_unref(&hw_device_ctx_);
      hw_device_ctx_ = nullptr;
    }
  }

  dec_ctx_ = avcodec_alloc_context3(dec);
  if (!dec_ctx_) {
    throw std::runtime_error("avcodec_alloc_context3 failed");
  }
  if (avcodec_parameters_to_context(dec_ctx_, par) < 0) {
    avcodec_free_context(&dec_ctx_);
    throw std::runtime_error("avcodec_parameters_to_context failed");
  }

  if (have_vaapi && hw_device_ctx_ != nullptr) {
    dec_ctx_->hw_device_ctx = av_buffer_ref(hw_device_ctx_);
    dec_ctx_->get_format = amd_vaapi_get_format;
    using_vaapi_ = true;
  } else {
    using_vaapi_ = false;
  }

  int oerr = avcodec_open2(dec_ctx_, dec, nullptr);
  if (oerr < 0) {
    if (have_vaapi) {
      dec_ctx_->hw_device_ctx = nullptr;
      dec_ctx_->get_format = nullptr;
      avcodec_free_context(&dec_ctx_);
      if (hw_device_ctx_) {
        av_buffer_unref(&hw_device_ctx_);
        hw_device_ctx_ = nullptr;
      }
      dec_ctx_ = avcodec_alloc_context3(dec);
      if (!dec_ctx_) {
        throw std::runtime_error("avcodec_alloc_context3 failed (sw retry)");
      }
      if (avcodec_parameters_to_context(dec_ctx_, par) < 0) {
        avcodec_free_context(&dec_ctx_);
        throw std::runtime_error("avcodec_parameters_to_context failed (sw retry)");
      }
      using_vaapi_ = false;
      if (avcodec_open2(dec_ctx_, dec, nullptr) < 0) {
        avcodec_free_context(&dec_ctx_);
        throw std::runtime_error("avcodec_open2 failed (software fallback)");
      }
    } else {
      avcodec_free_context(&dec_ctx_);
      throw std::runtime_error("avcodec_open2 failed");
    }
  }
}

void AmdDecoderSession::compute_frame_count_() {
  nb_frames_ = video_st_->nb_frames;
  if (nb_frames_ > 0) {
    return;
  }
  AVRational fps = video_st_->avg_frame_rate;
  if (fps.num > 0 && video_st_->duration != AV_NOPTS_VALUE) {
    double sec = video_st_->duration * av_q2d(video_st_->time_base);
    double f = av_q2d(fps);
    nb_frames_ = static_cast<int64_t>(std::llround(std::max(0.0, sec * f)));
  }
  if (nb_frames_ <= 0) {
    int64_t count = 0;
    std::unique_ptr<AVPacket, PacketDeleter> pkt(av_packet_alloc());
    if (!pkt) {
      throw std::runtime_error("av_packet_alloc failed");
    }
    while (av_read_frame(fmt_, pkt.get()) >= 0) {
      if (pkt->stream_index == video_stream_idx_) {
        ++count;
      }
      av_packet_unref(pkt.get());
    }
    nb_frames_ = std::max<int64_t>(1, count);
    if (av_seek_frame(fmt_, video_stream_idx_, 0, AVSEEK_FLAG_BACKWARD) < 0) {
      throw std::runtime_error("av_seek_frame rewind after packet count failed");
    }
    avcodec_flush_buffers(dec_ctx_);
  }
}

int64_t AmdDecoderSession::timestamp_for_index_(int64_t index) const {
  AVRational fps = video_st_->avg_frame_rate;
  if (fps.num <= 0) {
    fps = {30000, 1001};
  }
  return av_rescale_q(index, av_inv_q(fps), video_st_->time_base);
}

void AmdDecoderSession::seek_to_timestamp_(int64_t ts) {
  int r = av_seek_frame(fmt_, video_stream_idx_, ts, AVSEEK_FLAG_BACKWARD);
  if (r < 0) {
    throw std::runtime_error("av_seek_frame: " + av_error_string(r));
  }
}

void AmdDecoderSession::flush_decoder_() { avcodec_flush_buffers(dec_ctx_); }

#if defined(AMD_DECODE_WITH_HIP) && AMD_DECODE_WITH_HIP

torch::Tensor AmdDecoderSession::avframe_vaapi_drm_to_cuda_(AVFrame* src) {
  if (drm_frames_ctx_ == nullptr) {
    throw std::runtime_error("drm_frames_ctx not initialized");
  }
  std::unique_ptr<AVFrame, FrameDeleter> mapped(av_frame_alloc());
  if (!mapped) {
    throw std::runtime_error("av_frame_alloc failed (drm map)");
  }
  mapped->format = AV_PIX_FMT_DRM_PRIME;
  mapped->width = src->width;
  mapped->height = src->height;
  mapped->hw_frames_ctx = av_buffer_ref(drm_frames_ctx_);
  if (av_hwframe_get_buffer(drm_frames_ctx_, mapped.get(), 0) < 0) {
    throw std::runtime_error("av_hwframe_get_buffer (drm) failed");
  }
  const int map_flags = AV_HWFRAME_MAP_READ | AV_HWFRAME_MAP_OVERWRITE;
  if (av_hwframe_map(mapped.get(), src, map_flags) < 0) {
    throw std::runtime_error("av_hwframe_map (vaapi->drm_prime) failed");
  }
  if (mapped->data[0] == nullptr) {
    throw std::runtime_error("mapped DRM frame has no descriptor");
  }
  auto* desc = reinterpret_cast<AVDRMFrameDescriptor*>(mapped->data[0]);
  if (desc->nb_layers < 1 || desc->nb_objects < 1) {
    throw std::runtime_error("DRM descriptor missing layers/objects");
  }
  const AVDRMLayerDescriptor& layer = desc->layers[0];
  if (layer.nb_planes < 2) {
    throw std::runtime_error("DRM mapped frame has fewer than 2 planes (expected NV12)");
  }
  if (layer.format != static_cast<int>(kDrmFormatNv12)) {
    throw std::runtime_error("DRM mapped frame format is not NV12 (zero-copy path)");
  }
  const AVDRMPlaneDescriptor& yp = layer.planes[0];
  const AVDRMPlaneDescriptor& uvp = layer.planes[1];
  if (yp.object_index < 0 || yp.object_index >= desc->nb_objects || uvp.object_index < 0 ||
      uvp.object_index >= desc->nb_objects) {
    throw std::runtime_error("invalid DRM plane object_index");
  }
  const AVDRMObjectDescriptor& yobj = desc->objects[yp.object_index];
  const AVDRMObjectDescriptor& uvobj = desc->objects[uvp.object_index];
  if (yobj.fd < 0 || yobj.size <= 0 || uvobj.fd < 0 || uvobj.size <= 0) {
    throw std::runtime_error("invalid DRM object fd/size");
  }

  HipImportedMemories imported;
  imported.handles.reserve(2);

  auto import_one = [&](int fd, size_t size, void** base_out) {
    hipExternalMemoryHandleDesc hd{};
    hd.type = hipExternalMemoryHandleTypeOpaqueFd;
    hd.handle.fd = fd;
    hd.size = size;
    hipExternalMemory_t ext = nullptr;
    hipError_t he = hipImportExternalMemory(&ext, &hd);
    if (he != hipSuccess) {
      throw std::runtime_error(std::string("hipImportExternalMemory failed: ") + hipGetErrorString(he));
    }
    imported.handles.push_back(ext);
    hipExternalMemoryBufferDesc bd{};
    bd.size = size;
    bd.offset = 0;
    void* p = nullptr;
    he = hipExternalMemoryGetMappedBuffer(&p, ext, &bd);
    if (he != hipSuccess) {
      throw std::runtime_error(std::string("hipExternalMemoryGetMappedBuffer failed: ") + hipGetErrorString(he));
    }
    *base_out = p;
  };

  void* y_base = nullptr;
  void* uv_base = nullptr;
  import_one(yobj.fd, static_cast<size_t>(yobj.size), &y_base);
  if (yobj.fd == uvobj.fd) {
    uv_base = y_base;
  } else {
    import_one(uvobj.fd, static_cast<size_t>(uvobj.size), &uv_base);
  }

  auto* y_ptr = static_cast<const uint8_t*>(y_base) + static_cast<size_t>(yp.offset);
  auto* uv_ptr = static_cast<const uint8_t*>(uv_base) + static_cast<size_t>(uvp.offset);
  const int y_pitch = static_cast<int>(yp.pitch);
  const int uv_pitch = static_cast<int>(uvp.pitch);
  const int w = src->width;
  const int h = src->height;

  torch::Tensor out = torch::empty(
      {1, 3, h, w},
      torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA, device_index_));
  void* out_ptr = out.data_ptr();
  hipError_t ke = frigate_amd_decode::launch_nv12_to_bchw_rgb(
      y_ptr,
      y_pitch,
      uv_ptr,
      uv_pitch,
      static_cast<uint8_t*>(out_ptr),
      w,
      h,
      device_index_,
      nullptr);
  if (ke != hipSuccess) {
    throw std::runtime_error(std::string("launch_nv12_to_bchw_rgb failed: ") + hipGetErrorString(ke));
  }
  if (hipDeviceSynchronize() != hipSuccess) {
    throw std::runtime_error("hipDeviceSynchronize after NV12 kernel failed");
  }
  return out;
}

#endif  // AMD_DECODE_WITH_HIP

torch::Tensor AmdDecoderSession::avframe_to_bchw_rgb_cpu_(AVFrame* frame) {
  AVFrame* sw = frame;
  std::unique_ptr<AVFrame, FrameDeleter> transferred;

  if (frame->hw_frames_ctx != nullptr) {
    transferred.reset(av_frame_alloc());
    if (!transferred) {
      throw std::runtime_error("av_frame_alloc failed for hw transfer");
    }
    int tr = av_hwframe_transfer_data(transferred.get(), frame, 0);
    if (tr < 0) {
      throw std::runtime_error("av_hwframe_transfer_data: " + av_error_string(tr));
    }
    sw = transferred.get();
  }

  const int w = sw->width;
  const int h = sw->height;
  if (w <= 0 || h <= 0) {
    throw std::runtime_error("invalid decoded frame dimensions");
  }

  std::unique_ptr<SwsContext, SwsCtxDeleter> sws(sws_getContext(
      w,
      h,
      static_cast<AVPixelFormat>(sw->format),
      w,
      h,
      AV_PIX_FMT_RGB24,
      SWS_BILINEAR,
      nullptr,
      nullptr,
      nullptr));
  if (!sws) {
    throw std::runtime_error("sws_getContext failed");
  }

  std::vector<uint8_t> rgb(static_cast<size_t>(w) * static_cast<size_t>(h) * 3u);
  uint8_t* dst_slices[1] = {rgb.data()};
  int dst_linesize[1] = {3 * w};
  const int scale_ret = sws_scale(
      sws.get(), sw->data, sw->linesize, 0, h, dst_slices, dst_linesize);
  if (scale_ret != h) {
    throw std::runtime_error("sws_scale did not convert full frame");
  }

  auto opts = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
  torch::Tensor hwc = torch::from_blob(rgb.data(), {h, w, 3}, opts).clone();
  torch::Tensor chw = hwc.permute({2, 0, 1}).contiguous();
  return chw.unsqueeze(0);
}

torch::Tensor AmdDecoderSession::avframe_to_bchw_rgb_(AVFrame* src) {
#if defined(AMD_DECODE_WITH_HIP) && AMD_DECODE_WITH_HIP
  if (zero_copy_ready_ && using_vaapi_ && src->format == AV_PIX_FMT_VAAPI && src->hw_frames_ctx != nullptr) {
    const char* zc = std::getenv("FRIGATE_AMD_DECODE_DISABLE_ZEROCOPY");
    if (!(zc != nullptr && zc[0] == '1')) {
      try {
        return avframe_vaapi_drm_to_cuda_(src);
      } catch (const std::exception&) {
        const char* strict = std::getenv("FRIGATE_AMD_DECODE_STRICT_ZEROCOPY");
        if (strict != nullptr && strict[0] == '1') {
          throw;
        }
        return avframe_to_bchw_rgb_cpu_(src);
      }
    }
  }
#endif
  return avframe_to_bchw_rgb_cpu_(src);
}

torch::Tensor AmdDecoderSession::decode_one_displayed_frame_() {
  std::unique_ptr<AVPacket, PacketDeleter> pkt(av_packet_alloc());
  std::unique_ptr<AVFrame, FrameDeleter> frame(av_frame_alloc());
  if (!pkt || !frame) {
    throw std::runtime_error("av_packet_alloc / av_frame_alloc failed");
  }

  for (;;) {
    int r = av_read_frame(fmt_, pkt.get());
    if (r == AVERROR_EOF) {
      throw std::runtime_error("unexpected EOF while decoding video frame");
    }
    if (r < 0) {
      throw std::runtime_error("av_read_frame: " + av_error_string(r));
    }
    if (pkt->stream_index != video_stream_idx_) {
      av_packet_unref(pkt.get());
      continue;
    }
    int send_r = avcodec_send_packet(dec_ctx_, pkt.get());
    av_packet_unref(pkt.get());
    if (send_r < 0) {
      throw std::runtime_error("avcodec_send_packet: " + av_error_string(send_r));
    }
    int recv_r = avcodec_receive_frame(dec_ctx_, frame.get());
    if (recv_r == AVERROR(EAGAIN)) {
      continue;
    }
    if (recv_r < 0) {
      throw std::runtime_error("avcodec_receive_frame: " + av_error_string(recv_r));
    }
    return avframe_to_bchw_rgb_(frame.get());
  }
}

void AmdDecoderSession::seek_to_index(int64_t index) {
  if (nb_frames_ <= 0) {
    return;
  }
  index = std::max<int64_t>(0, std::min<int64_t>(index, nb_frames_ - 1));
  seek_to_timestamp_(timestamp_for_index_(index));
  flush_decoder_();
  seq_cursor_ = index;
}

int64_t AmdDecoderSession::get_index_from_time_in_seconds(double t_sec) const {
  if (nb_frames_ <= 0) {
    return 0;
  }
  AVRational fps = video_st_->avg_frame_rate;
  if (fps.num <= 0) {
    fps = {30000, 1001};
  }
  double f = av_q2d(fps);
  int64_t idx = static_cast<int64_t>(std::llround(std::max(0.0, t_sec) * f));
  return std::max<int64_t>(0, std::min<int64_t>(idx, nb_frames_ - 1));
}

torch::Tensor AmdDecoderSession::get_frame_at_index(int64_t index) {
  if (nb_frames_ <= 0) {
    auto opts = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
    return torch::empty({1, 3, 0, 0}, opts);
  }
  seek_to_index(index);
  torch::Tensor t = decode_one_displayed_frame_();
  seq_cursor_ = index + 1;
  return t;
}

torch::Tensor AmdDecoderSession::get_frames(const std::vector<int64_t>& indices) {
  if (indices.empty()) {
    auto opts = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
    return torch::empty({0, 3, 0, 0}, opts);
  }
  std::vector<torch::Tensor> parts;
  parts.reserve(indices.size());
  for (int64_t idx : indices) {
    parts.push_back(get_frame_at_index(idx));
  }
  if (parts.size() == 1) {
    return parts[0];
  }
  return torch::cat(parts, 0);
}

std::vector<torch::Tensor> AmdDecoderSession::get_batch_frames(int64_t count) {
  if (count <= 0) {
    return {};
  }
  std::vector<torch::Tensor> out;
  out.reserve(static_cast<size_t>(count));
  for (int64_t i = 0; i < count; ++i) {
    if (seq_cursor_ >= nb_frames_) {
      throw std::runtime_error("get_batch_frames past end of stream");
    }
    torch::Tensor t = decode_one_displayed_frame_();
    seq_cursor_ += 1;
    out.push_back(t);
  }
  return out;
}
