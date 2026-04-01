#pragma once

#include <torch/extension.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct AVFormatContext;
struct AVCodecContext;
struct AVBufferRef;
struct AVFrame;
struct AVStream;

/**
 * FFmpeg-backed AMD decode: VAAPI (DRM render node) when available, else SW.
 * When built with HIP and DRM map succeeds, frames are uint8 BCHW RGB on the
 * ROCm device (torch::kCUDA index). Otherwise hw frames use av_hwframe_transfer_data
 * + sws_scale and output CPU tensors (Python may .to(cuda) on ROCm).
 */
class AmdDecoderSession {
 public:
  AmdDecoderSession(const std::string& path, int device_index);
  ~AmdDecoderSession();

  AmdDecoderSession(const AmdDecoderSession&) = delete;
  AmdDecoderSession& operator=(const AmdDecoderSession&) = delete;

  int64_t frame_count() const { return nb_frames_; }
  int64_t len() const { return nb_frames_; }

  torch::Tensor get_frame_at_index(int64_t index);
  torch::Tensor get_frames(const std::vector<int64_t>& indices);
  std::vector<torch::Tensor> get_batch_frames(int64_t count);

  void seek_to_index(int64_t index);
  int64_t get_index_from_time_in_seconds(double t_sec) const;

  /** True when VAAPI hw device + hw frames are active (else software decode). */
  bool uses_hw_decode() const { return using_vaapi_; }

  /** True when VAAPI + DRM map + HIP produce device tensors (no CPU readback). */
  bool uses_zero_copy_decode() const { return zero_copy_ready_; }

 private:
  void open_demuxer_(const std::string& path);
  void setup_decoder_(bool prefer_vaapi);
  void init_drm_map_context_();
  void compute_frame_count_();
  int64_t timestamp_for_index_(int64_t index) const;
  void seek_to_timestamp_(int64_t ts);
  void flush_decoder_();
  torch::Tensor decode_one_displayed_frame_();
  torch::Tensor avframe_to_bchw_rgb_(AVFrame* frame);
  torch::Tensor avframe_to_bchw_rgb_cpu_(AVFrame* frame);

#if defined(AMD_DECODE_WITH_HIP) && AMD_DECODE_WITH_HIP
  torch::Tensor avframe_vaapi_drm_to_cuda_(AVFrame* src);
#endif

  std::string path_;
  int device_index_{0};

  AVFormatContext* fmt_{nullptr};
  int video_stream_idx_{-1};
  AVStream* video_st_{nullptr};

  AVCodecContext* dec_ctx_{nullptr};
  AVBufferRef* hw_device_ctx_{nullptr};
  AVBufferRef* drm_device_ctx_{nullptr};
  AVBufferRef* drm_frames_ctx_{nullptr};

  int64_t nb_frames_{0};
  int video_width_{0};
  int video_height_{0};

  int64_t seq_cursor_{0};

  bool using_vaapi_{false};
  bool zero_copy_ready_{false};
};
