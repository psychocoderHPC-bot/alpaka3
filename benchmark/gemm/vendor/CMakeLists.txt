#
# Copyright 2026 alpaka authors
# SPDX-License-Identifier: MPL-2.0
#
# Optional benchmark-only reference baseline. Disabled by default; when off this
# file defines the option and returns without fetching anything.
#
cmake_minimum_required(VERSION 3.25)

option(
    ALPAKA_GEMM_VENDOR_BASELINE
    "Build the optional alpakaVendor reference GEMM baseline (fetches alpakaVendor at a pinned commit)."
    OFF
)

if(NOT ALPAKA_GEMM_VENDOR_BASELINE)
    return()
endif()

include(FetchContent)

# Exact commit of https://github.com/psychocoderHPC/alpakaVendor, branch dev.
set(ALPAKA_GEMM_VENDOR_SHA
    "ccf128e3c1ff85ae0b1fc5d9b08fd9bc44f02aeb"
    CACHE STRING
    "Pinned alpakaVendor commit used by the GEMM reference baseline."
)

option(
    ALPAKA_GEMM_VENDOR_OPENBLAS
    "Use the OpenBLAS host backend of alpakaVendor (required by this baseline for now)."
    ON
)

if(NOT ALPAKA_GEMM_VENDOR_OPENBLAS)
    message(
        FATAL_ERROR
        "gemm_vendor_bench currently requires -DALPAKA_GEMM_VENDOR_OPENBLAS=ON. The CUDA, HIP and oneMKL "
        "backends of alpakaVendor need a backend-specific device in the functor and are not wired yet."
    )
endif()

find_package(PkgConfig REQUIRED)
pkg_check_modules(ALPAKA_GEMM_OPENBLAS QUIET IMPORTED_TARGET openblas)
if(NOT ALPAKA_GEMM_OPENBLAS_FOUND)
    message(
        FATAL_ERROR
        "ALPAKA_GEMM_VENDOR_BASELINE=ON requires the OpenBLAS development package with a pkg-config "
        "module named 'openblas' (Debian/Ubuntu: libopenblas-dev). No such module was found. "
        "Reconfigure with -DALPAKA_GEMM_VENDOR_BASELINE=OFF to build the in-tree benchmark only."
    )
endif()

# Configure alpakaVendor before FetchContent_MakeAvailable().
set(alpakaV_TESTS OFF CACHE BOOL "" FORCE)
set(alpakaV_DOCS OFF CACHE BOOL "" FORCE)
set(alpakaV_DEP_FFTW OFF CACHE BOOL "" FORCE)
set(alpakaV_DEP_OPENBLAS ON CACHE BOOL "" FORCE)
set(alpakaV_DEP_CUFFT OFF CACHE BOOL "" FORCE)
set(alpakaV_DEP_CUBLAS OFF CACHE BOOL "" FORCE)
set(alpakaV_DEP_ROCFFT OFF CACHE BOOL "" FORCE)
set(alpakaV_DEP_ROCBLAS OFF CACHE BOOL "" FORCE)
set(alpakaV_DEP_ONEMKL OFF CACHE BOOL "" FORCE)

# alpakaVendor's own CMakeLists unconditionally fetches its pinned alpaka3.
# Point that FetchContent at the checkout under test so the baseline compiles
# against exactly these headers. The alpaka target is guarded against duplicate
# creation, so the nested add_subdirectory() is safe.
get_filename_component(ALPAKA_GEMM_CHECKOUT_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
if(NOT EXISTS "${ALPAKA_GEMM_CHECKOUT_ROOT}/include/alpaka/alpaka.hpp")
    message(
        FATAL_ERROR
        "Could not locate the alpaka3 source tree at '${ALPAKA_GEMM_CHECKOUT_ROOT}'. The vendor baseline "
        "must be built from the checkout, e.g. cmake -S benchmark/gemm "
        "-Dalpaka_USE_SOURCE_TREE=ON -DALPAKA_GEMM_VENDOR_BASELINE=ON."
    )
endif()
set(FETCHCONTENT_SOURCE_DIR_ALPAKA3
    "${ALPAKA_GEMM_CHECKOUT_ROOT}"
    CACHE PATH
    "Reuse the alpaka3 checkout under test instead of alpakaVendor's pinned alpaka3."
    FORCE
)

FetchContent_Declare(
    alpakaVendor
    GIT_REPOSITORY https://github.com/psychocoderHPC/alpakaVendor.git
    GIT_TAG "${ALPAKA_GEMM_VENDOR_SHA}"
    GIT_SHALLOW FALSE
)
FetchContent_MakeAvailable(alpakaVendor)

add_executable(
    gemm_vendor_bench
    "${CMAKE_CURRENT_LIST_DIR}/../src/gemm_bench.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/../src/json.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/vendor_baseline.cpp"
)
target_compile_features(gemm_vendor_bench PRIVATE cxx_std_20)
target_include_directories(gemm_vendor_bench PRIVATE "${CMAKE_CURRENT_LIST_DIR}/../src" "${CMAKE_CURRENT_LIST_DIR}")
target_link_libraries(gemm_vendor_bench PRIVATE alpaka::alpaka alpakaVendor::alpakaVendor)
target_compile_definitions(
    gemm_vendor_bench
    PRIVATE
        GEMM_USE_VENDOR_BASELINE=1
        GEMM_DEFAULT_MANIFEST="${CMAKE_CURRENT_LIST_DIR}/../workloads.json"
        GEMM_SOURCE_DIR="${ALPAKA_GEMM_CHECKOUT_ROOT}"
)
alpaka_finalize(gemm_vendor_bench)
if(COMMAND alpaka_gemm_apply_native_isa)
    alpaka_gemm_apply_native_isa(gemm_vendor_bench)
endif()
set_target_properties(gemm_vendor_bench PROPERTIES FOLDER benchmarks/gemm)
