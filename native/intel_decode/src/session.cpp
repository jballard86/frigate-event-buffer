#include "session.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#if defined(INTEL_DECODE_XPU_ZEROCOPY) && INTEL_DECODE_XPU_ZEROCOPY
#include "xpu_zerocopy.hpp"
#endif

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_drm.h>
#include <libavutil/imgutils.h>
#include <libavutil/mathematics.h>
#include <libswscale/swscale.h>
}

namespace {

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

std::string drm_device_from_index(int device_index) {
  const char* env = std::getenv("FRIGATE_INTEL_DRM_DEVICE");
  if (env != nullptr && env[0] != '\0') {
    return std::string(env);
  }
  const int base = 128;
  const int n = base + std::max(0, device_index);
  return "/dev/dri/renderD" + std::to_string(n);
}

/** DRM_FORMAT_NV12 from linux/drm_fourcc.h (avoid hard dependency on libdrm headers). */
constexpr uint32_t kDrmFormatNv12 = 0x3231564e;

const AVCodec* pick_qsv_decoder(enum AVCodecID id) {
  switch (id) {
    case AV_CODEC_ID_H264:
      return avcodec_find_decoder_by_name("h264_qsv");
    case AV_CODEC_ID_HEVC:
      return avcodec_find_decoder_by_name("hevc_qsv");
    default:
      return nullptr;
  }
}

}  // namespace

IntelDecoderSession::IntelDecoderSession(const std::string& path, int device_index)
    : path_(path), device_index_(device_index) {
  open_demuxer_(path);
  setup_decoder_();
  if (using_hw_) {
    init_drm_map_context_();
  }
  compute_frame_count_();
  if (video_width_ > 4096) {
    throw std::runtime_error("video width exceeds DECODER_MAX_WIDTH 4096 (Intel decode)");
  }
  seek_to_timestamp_(0);
  flush_decoder_();
  seq_cursor_ = 0;
}

IntelDecoderSession::~IntelDecoderSession() {
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

void IntelDecoderSession::open_demuxer_(const std::string& path) {
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

void IntelDecoderSession::setup_decoder_() {
  AVCodecParameters* par = video_st_->codecpar;
  const AVCodec* dec = pick_qsv_decoder(par->codec_id);
  if (dec == nullptr) {
    throw std::runtime_error(
        "Intel decode requires QSV (h264_qsv/hevc_qsv); unsupported codec for this stream");
  }
  int hwr = av_hwdevice_ctx_create(&hw_device_ctx_, AV_HWDEVICE_TYPE_QSV, "auto", nullptr, 0);
  if (hwr < 0) {
    av_buffer_unref(&hw_device_ctx_);
    hw_device_ctx_ = nullptr;
    throw std::runtime_error("Intel decode: QSV device init failed: " + av_error_string(hwr));
  }
  using_hw_ = true;

  dec_ctx_ = avcodec_alloc_context3(dec);
  if (!dec_ctx_) {
    throw std::runtime_error("avcodec_alloc_context3 failed");
  }
  if (avcodec_parameters_to_context(dec_ctx_, par) < 0) {
    throw std::runtime_error("avcodec_parameters_to_context failed");
  }

  if (using_hw_ && hw_device_ctx_) {
    dec_ctx_->hw_device_ctx = av_buffer_ref(hw_device_ctx_);
  }

  if (avcodec_open2(dec_ctx_, dec, nullptr) < 0) {
    throw std::runtime_error("avcodec_open2 failed");
  }
}

void IntelDecoderSession::init_drm_map_context_() {
  std::string dev = drm_device_from_index(device_index_);
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

void IntelDecoderSession::compute_frame_count_() {
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

int64_t IntelDecoderSession::timestamp_for_index_(int64_t index) const {
  AVRational fps = video_st_->avg_frame_rate;
  if (fps.num <= 0) {
    fps = {30000, 1001};
  }
  return av_rescale_q(index, av_inv_q(fps), video_st_->time_base);
}

void IntelDecoderSession::seek_to_timestamp_(int64_t ts) {
  int r = av_seek_frame(fmt_, video_stream_idx_, ts, AVSEEK_FLAG_BACKWARD);
  if (r < 0) {
    throw std::runtime_error("av_seek_frame: " + av_error_string(r));
  }
}

void IntelDecoderSession::flush_decoder_() { avcodec_flush_buffers(dec_ctx_); }

torch::Tensor IntelDecoderSession::avframe_to_bchw_rgb_(AVFrame* src) {
  AVFrame* frame = src;
  std::unique_ptr<AVFrame, FrameDeleter> transferred;

  if (!drm_map_ready_ && using_hw_ && drm_frames_ctx_ != nullptr && src->hw_frames_ctx != nullptr) {
    probe_drm_map_(src);
  }

#if defined(INTEL_DECODE_XPU_ZEROCOPY) && INTEL_DECODE_XPU_ZEROCOPY
  // Attempt true zero-copy (DRM PRIME -> DMA-BUF import -> on-XPU NV12->RGB).
  // Intel uses QSV-only decode; when this build flag is on, do not fall back to
  // host readback + swscale for hw frames.
  if (using_hw_ && drm_map_ready_ && drm_frames_ctx_ != nullptr && src->hw_frames_ctx != nullptr) {
    std::unique_ptr<AVFrame, FrameDeleter> mapped(av_frame_alloc());
    if (!mapped) {
      throw std::runtime_error("Intel zero-copy required: av_frame_alloc failed");
    }
    mapped->format = AV_PIX_FMT_DRM_PRIME;
    mapped->width = src->width;
    mapped->height = src->height;
    mapped->hw_frames_ctx = av_buffer_ref(drm_frames_ctx_);
    if (mapped->hw_frames_ctx == nullptr) {
      throw std::runtime_error("Intel zero-copy required: av_buffer_ref(drm_frames_ctx) failed");
    }
    if (av_hwframe_get_buffer(drm_frames_ctx_, mapped.get(), 0) < 0) {
      throw std::runtime_error("Intel zero-copy required: av_hwframe_get_buffer failed");
    }
    const int map_flags = AV_HWFRAME_MAP_READ | AV_HWFRAME_MAP_OVERWRITE;
    if (av_hwframe_map(mapped.get(), src, map_flags) < 0) {
      throw std::runtime_error("Intel zero-copy required: av_hwframe_map to DRM PRIME failed");
    }
    if (mapped->data[0] == nullptr) {
      throw std::runtime_error("Intel zero-copy required: mapped DRM PRIME missing descriptor");
    }
    auto zc = intel_xpu_zerocopy::try_nv12_drmprime_to_bchw_rgb_xpu(
        mapped.get(), device_index_, mapped->width, mapped->height);
    if (zc.has_value() && zc->used_zero_copy) {
      zero_copy_active_ = true;
      xpu_output_active_ = (zc->tensor.device().type() == torch::kXPU);
      return zc->tensor;
    }
    throw std::runtime_error("Intel zero-copy required: fast path unavailable for mapped frame");
  }
#endif

#if defined(INTEL_DECODE_XPU_ZEROCOPY) && INTEL_DECODE_XPU_ZEROCOPY
  // XPU zero-copy builds: never use host readback + swscale for QSV hw frames.
  if (using_hw_ && src->hw_frames_ctx != nullptr) {
    throw std::runtime_error("Intel zero-copy required: refusing CPU hwframe transfer");
  }
#endif

  if (src->hw_frames_ctx != nullptr) {
    transferred.reset(av_frame_alloc());
    if (!transferred) {
      throw std::runtime_error("av_frame_alloc failed for hw transfer");
    }
    int tr = av_hwframe_transfer_data(transferred.get(), src, 0);
    if (tr < 0) {
      throw std::runtime_error("av_hwframe_transfer_data: " + av_error_string(tr));
    }
    frame = transferred.get();
  }

  const int w = frame->width;
  const int h = frame->height;
  if (w <= 0 || h <= 0) {
    throw std::runtime_error("invalid decoded frame dimensions");
  }

  std::unique_ptr<SwsContext, SwsCtxDeleter> sws(sws_getContext(
      w,
      h,
      static_cast<AVPixelFormat>(frame->format),
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
      sws.get(), frame->data, frame->linesize, 0, h, dst_slices, dst_linesize);
  if (scale_ret != h) {
    throw std::runtime_error("sws_scale did not convert full frame");
  }

  auto opts = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
  torch::Tensor hwc = torch::from_blob(rgb.data(), {h, w, 3}, opts).clone();
  torch::Tensor chw = hwc.permute({2, 0, 1}).contiguous();
  torch::Tensor out = chw.unsqueeze(0);

#if defined(INTEL_DECODE_XPU_ZEROCOPY) && INTEL_DECODE_XPU_ZEROCOPY
  const char* disable = std::getenv("FRIGATE_INTEL_DECODE_DISABLE_XPU_OUTPUT");
  if (!(disable != nullptr && disable[0] == '1')) {
    try {
      // Note: this enables native XPU output when built against an XPU-capable torch.
      // The true zero-copy path (DRM PRIME -> XPU import) is handled separately.
      out = out.to(torch::Device(torch::kXPU, device_index_), /*non_blocking=*/true);
      xpu_output_active_ = (out.device().type() == torch::kXPU);
    } catch (const std::exception&) {
      // Keep CPU output if XPU is unavailable at runtime.
    }
  }
#endif

  return out;
}

void IntelDecoderSession::probe_drm_map_(AVFrame* src) {
  // Best-effort probe: can we map the hw frame to DRM PRIME NV12?
  // This is intentionally side-effect free for the decode API.
  std::unique_ptr<AVFrame, FrameDeleter> mapped(av_frame_alloc());
  if (!mapped) {
    return;
  }
  mapped->format = AV_PIX_FMT_DRM_PRIME;
  mapped->width = src->width;
  mapped->height = src->height;
  mapped->hw_frames_ctx = av_buffer_ref(drm_frames_ctx_);
  if (av_hwframe_get_buffer(drm_frames_ctx_, mapped.get(), 0) < 0) {
    return;
  }
  const int map_flags = AV_HWFRAME_MAP_READ | AV_HWFRAME_MAP_OVERWRITE;
  if (av_hwframe_map(mapped.get(), src, map_flags) < 0) {
    return;
  }
  if (mapped->data[0] == nullptr) {
    return;
  }
  auto* desc = reinterpret_cast<AVDRMFrameDescriptor*>(mapped->data[0]);
  if (desc->nb_layers < 1 || desc->nb_objects < 1) {
    return;
  }
  const AVDRMLayerDescriptor& layer = desc->layers[0];
  if (layer.nb_planes < 2) {
    return;
  }
  if (layer.format != static_cast<int>(kDrmFormatNv12)) {
    return;
  }
  drm_map_ready_ = true;
}

torch::Tensor IntelDecoderSession::decode_one_displayed_frame_() {
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

void IntelDecoderSession::seek_to_index(int64_t index) {
  if (nb_frames_ <= 0) {
    return;
  }
  index = std::max<int64_t>(0, std::min<int64_t>(index, nb_frames_ - 1));
  seek_to_timestamp_(timestamp_for_index_(index));
  flush_decoder_();
  seq_cursor_ = index;
}

int64_t IntelDecoderSession::get_index_from_time_in_seconds(double t_sec) const {
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

torch::Tensor IntelDecoderSession::get_frame_at_index(int64_t index) {
  if (nb_frames_ <= 0) {
    auto opts = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
    return torch::empty({1, 3, 0, 0}, opts);
  }
  seek_to_index(index);
  torch::Tensor t = decode_one_displayed_frame_();
  seq_cursor_ = index + 1;
  return t;
}

torch::Tensor IntelDecoderSession::get_frames(const std::vector<int64_t>& indices) {
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

std::vector<torch::Tensor> IntelDecoderSession::get_batch_frames(int64_t count) {
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
