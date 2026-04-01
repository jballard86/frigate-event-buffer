# gpu-02: pybind11 + libtorch + FFmpeg. IntelDecoderSession prefers QSV decode;
# software fallback when QSV init fails or FRIGATE_INTEL_DECODE_FORCE_SW=1.

cmake_minimum_required(VERSION 3.18)
project(frigate_intel_decode LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(WIN32)
  set(_intel_decode_default OFF)
else()
  set(_intel_decode_default ON)
endif()
option(INTEL_DECODE_BUILD "Build frigate_intel_decode Python extension" ${_intel_decode_default})

if(NOT INTEL_DECODE_BUILD)
  message(STATUS "INTEL_DECODE_BUILD=OFF — skipping frigate_intel_decode (set ON with deps to build).")
  return()
endif()

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)

include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.11.1
)
FetchContent_MakeAvailable(pybind11)

find_package(PkgConfig REQUIRED)
pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET
  libavcodec>=58
  libavformat>=58
  libavutil>=56
  libswscale>=5
)

pybind11_add_module(frigate_intel_decode MODULE src/bindings.cpp src/session.cpp)
target_link_libraries(
  frigate_intel_decode PRIVATE
  "${TORCH_LIBRARIES}"
  PkgConfig::FFMPEG
)
target_include_directories(frigate_intel_decode PRIVATE "${TORCH_INCLUDE_DIRS}")

set_target_properties(
  frigate_intel_decode PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
)
