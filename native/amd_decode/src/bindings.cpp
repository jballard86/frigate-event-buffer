/**
 * pybind11 entry for AMD decode session (gpu-03).
 *
 * VAAPI hw decode + HIP zero-copy only (no CPU fallback path).
 */

#include "session.hpp"

#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace {

torch::Tensor decode_first_frame_bchw_rgb(const std::string& path) {
  AmdDecoderSession session(path, 0);
  return session.get_frame_at_index(0);
}

}  // namespace

PYBIND11_MODULE(frigate_amd_decode, m) {
  m.doc() = "AMD decode: FFmpeg VAAPI + HIP zero-copy (DRM PRIME -> ROCm tensor).";

  m.def("version", []() { return std::string("0.2.0-zerocopy-hip"); });

  py::class_<AmdDecoderSession>(m, "AmdDecoderSession")
      .def(py::init<const std::string&, int>(), py::arg("path"), py::arg("device_index") = 0)
      .def("__len__", &AmdDecoderSession::len)
      .def("frame_count", &AmdDecoderSession::frame_count)
      .def("get_frames", &AmdDecoderSession::get_frames, py::arg("indices"))
      .def("get_frame_at_index", &AmdDecoderSession::get_frame_at_index, py::arg("index"))
      .def("get_batch_frames", &AmdDecoderSession::get_batch_frames, py::arg("count"))
      .def("seek_to_index", &AmdDecoderSession::seek_to_index, py::arg("index"))
      .def(
          "get_index_from_time_in_seconds",
          &AmdDecoderSession::get_index_from_time_in_seconds,
          py::arg("t_sec"))
      .def("uses_hw_decode", &AmdDecoderSession::uses_hw_decode)
      .def("zero_copy_capable", &AmdDecoderSession::zero_copy_capable)
      .def("uses_zero_copy_decode", &AmdDecoderSession::uses_zero_copy_decode);

  m.def("decode_first_frame_bchw_rgb", &decode_first_frame_bchw_rgb, py::arg("path"));
}
