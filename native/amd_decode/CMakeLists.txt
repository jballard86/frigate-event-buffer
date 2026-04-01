# gpu-03: pybind11 + libtorch + FFmpeg. VAAPI decode; optional HIP zero-copy
# (VAAPI -> DRM PRIME -> hipImportExternalMemory + NV12 HIP kernel -> torch CUDA tensor).

cmake_minimum_required(VERSION 3.18)
project(frigate_amd_decode LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(WIN32)
  set(_amd_decode_default OFF)
else()
  set(_amd_decode_default ON)
endif()
option(AMD_DECODE_BUILD "Build frigate_amd_decode Python extension" ${_amd_decode_default})

if(NOT AMD_DECODE_BUILD)
  message(STATUS "AMD_DECODE_BUILD=OFF — skipping frigate_amd_decode (set ON with deps to build).")
  return()
endif()

option(AMD_DECODE_HIP "Enable ROCm HIP zero-copy decode (requires ROCm in CMAKE_PREFIX_PATH)" ON)

set(_amd_hip_sources "")
if(AMD_DECODE_HIP AND NOT WIN32)
  if(DEFINED ENV{ROCM_PATH})
    list(APPEND CMAKE_PREFIX_PATH "$ENV{ROCM_PATH}" "$ENV{ROCM_PATH}/hip")
  endif()
  list(APPEND CMAKE_PREFIX_PATH "/opt/rocm" "/opt/rocm/hip")
  find_package(hip CONFIG QUIET)
  if(hip_FOUND)
    enable_language(HIP)
    if(NOT DEFINED CMAKE_HIP_ARCHITECTURES OR CMAKE_HIP_ARCHITECTURES STREQUAL "")
      set(CMAKE_HIP_ARCHITECTURES "gfx906;gfx1030;gfx1100;gfx1101;gfx1200" CACHE STRING "AMD GPU targets for HIP" FORCE)
    endif()
    list(APPEND _amd_hip_sources "${CMAKE_CURRENT_SOURCE_DIR}/src/nv12_to_rgb_hip.hip")
    message(STATUS "AMD decode: HIP enabled (zero-copy path). CMAKE_HIP_ARCHITECTURES=${CMAKE_HIP_ARCHITECTURES}")
  else()
    message(FATAL_ERROR "AMD decode requires HIP (zero-copy only). Install ROCm/HIP or set a valid ROCM_PATH/CMAKE_PREFIX_PATH.")
  endif()
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

pybind11_add_module(
  frigate_amd_decode
  MODULE
  src/bindings.cpp
  src/session.cpp
  ${_amd_hip_sources}
)

if(_amd_hip_sources)
  target_compile_definitions(frigate_amd_decode PRIVATE AMD_DECODE_WITH_HIP=1)
  set_source_files_properties(src/nv12_to_rgb_hip.hip PROPERTIES LANGUAGE HIP)
  target_link_libraries(frigate_amd_decode PRIVATE hip::host)
endif()

target_link_libraries(
  frigate_amd_decode PRIVATE
  "${TORCH_LIBRARIES}"
  PkgConfig::FFMPEG
)
target_include_directories(frigate_amd_decode PRIVATE "${TORCH_INCLUDE_DIRS}")

set_target_properties(
  frigate_amd_decode PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
)
