/**
 * pybind11 entry for Intel decode session (gpu-02 Phase 2).
 *
 * QSV-only decode (h264_qsv/hevc_qsv); no software libavcodec fallback.
 */

#include "session.hpp"

#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace {

torch::Tensor decode_first_frame_bchw_rgb(const std::string& path) {
  IntelDecoderSession session(path, 0);
  return session.get_frame_at_index(0);
}

}  // namespace

PYBIND11_MODULE(frigate_intel_decode, m) {
  m.doc() = "Intel decode: FFmpeg QSV only (h264_qsv/hevc_qsv); DecoderSession API.";

  m.def("version", []() { return std::string("0.2.0-phase2"); });

  py::class_<IntelDecoderSession>(m, "IntelDecoderSession")
      .def(py::init<const std::string&, int>(), py::arg("path"), py::arg("device_index") = 0)
      .def("__len__", &IntelDecoderSession::len)
      .def("frame_count", &IntelDecoderSession::frame_count)
      .def("get_frames", &IntelDecoderSession::get_frames, py::arg("indices"))
      .def("get_frame_at_index", &IntelDecoderSession::get_frame_at_index, py::arg("index"))
      .def("get_batch_frames", &IntelDecoderSession::get_batch_frames, py::arg("count"))
      .def("seek_to_index", &IntelDecoderSession::seek_to_index, py::arg("index"))
      .def(
          "get_index_from_time_in_seconds",
          &IntelDecoderSession::get_index_from_time_in_seconds,
          py::arg("t_sec"))
      .def("uses_hw_decode", &IntelDecoderSession::uses_hw_decode)
      .def("can_map_to_drm_prime", &IntelDecoderSession::can_map_to_drm_prime)
      .def("uses_xpu_output", &IntelDecoderSession::uses_xpu_output)
      .def("uses_zero_copy_decode", &IntelDecoderSession::uses_zero_copy_decode);

  m.def("decode_first_frame_bchw_rgb", &decode_first_frame_bchw_rgb, py::arg("path"));
  // Back-compat alias (historical name from early spike).
  m.def("decode_first_frame_bchw_rgb_sw", &decode_first_frame_bchw_rgb, py::arg("path"));
}
