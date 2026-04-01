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
 * FFmpeg-backed decode session: **QSV only** (h264_qsv / hevc_qsv). There is no
 * software libavcodec fallback; unsupported codecs or QSV init failure throws.
 */
class IntelDecoderSession {
 public:
  IntelDecoderSession(const std::string& path, int device_index);
  ~IntelDecoderSession();

  IntelDecoderSession(const IntelDecoderSession&) = delete;
  IntelDecoderSession& operator=(const IntelDecoderSession&) = delete;

  int64_t frame_count() const { return nb_frames_; }
  int64_t len() const { return nb_frames_; }

  torch::Tensor get_frame_at_index(int64_t index);
  torch::Tensor get_frames(const std::vector<int64_t>& indices);
  std::vector<torch::Tensor> get_batch_frames(int64_t count);

  void seek_to_index(int64_t index);
  int64_t get_index_from_time_in_seconds(double t_sec) const;

  /** True when this session uses QSV (always true after successful construction). */
  bool uses_hw_decode() const { return using_hw_; }

  /** True when hw frames can be mapped to DRM PRIME (for zero-copy experiments). */
  bool can_map_to_drm_prime() const { return drm_map_ready_; }

  /** True when this session returns XPU tensors (built against XPU torch + runtime XPU available). */
  bool uses_xpu_output() const { return xpu_output_active_; }

  /** True only when the experimental DRM PRIME -> XPU import path is active. */
  bool uses_zero_copy_decode() const { return zero_copy_active_; }

 private:
  void open_demuxer_(const std::string& path);
  void setup_decoder_();
  void init_drm_map_context_();
  void compute_frame_count_();
  int64_t timestamp_for_index_(int64_t index) const;
  void seek_to_timestamp_(int64_t ts);
  void flush_decoder_();
  torch::Tensor decode_one_displayed_frame_();
  torch::Tensor avframe_to_bchw_rgb_(AVFrame* frame);
  void probe_drm_map_(AVFrame* src);

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

  /** Next sequential frame index for get_batch_frames. */
  int64_t seq_cursor_{0};

  bool using_hw_{false};
  bool drm_map_ready_{false};
  bool xpu_output_active_{false};
  bool zero_copy_active_{false};
};
