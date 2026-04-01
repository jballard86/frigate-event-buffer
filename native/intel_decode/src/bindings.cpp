/**
 * pybind11 entry for Intel decode session (gpu-02 Phase 2).
 *
 * QSV hardware decode when available; software FFmpeg fallback or
 * FRIGATE_INTEL_DECODE_FORCE_SW=1. Output tensors are CPU uint8 BCHW RGB.
 */

#include "session.hpp"

#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace {

torch::Tensor decode_first_frame_bchw_rgb_sw(const std::string& path) {
  IntelDecoderSession session(path, 0);
  return session.get_frame_at_index(0);
}

}  // namespace

PYBIND11_MODULE(frigate_intel_decode, m) {
  m.doc() = "Intel decode: FFmpeg QSV (preferred) or SW; DecoderSession API; CPU BCHW RGB.";

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
      .def("uses_hw_decode", &IntelDecoderSession::uses_hw_decode);

  m.def("decode_first_frame_bchw_rgb_sw", &decode_first_frame_bchw_rgb_sw, py::arg("path"));
}
