#
# Copyright 2026 alpaka authors
# SPDX-License-Identifier: MPL-2.0
#
cmake_minimum_required(VERSION 3.25)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
project(alpakaGemmBenchmark LANGUAGES CXX)

# Also loadable when this directory is configured standalone (not via the root
# CMakeLists.txt); include_guard keeps the option and helper single-definition.
include("${CMAKE_CURRENT_LIST_DIR}/../../cmake/alpakaGemm.cmake")

if(NOT TARGET alpaka::alpaka)
    option(alpaka_USE_SOURCE_TREE "Use alpaka's source tree instead of an alpaka installation" OFF)
    if(alpaka_USE_SOURCE_TREE)
        # Don't build the benchmarks recursively
        set(alpaka_BENCHMARKS OFF)
        add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/../.." "${CMAKE_BINARY_DIR}/alpaka")
    else()
        find_package(alpaka REQUIRED)
    endif()
endif()

# The harness itself is portable C++20; the M2 adapter drives the real
# common-source GEMM kernel through the public host API.
add_executable(gemm_bench "src/gemm_bench.cpp" "src/json.cpp")
target_compile_features(gemm_bench PRIVATE cxx_std_20)
target_include_directories(gemm_bench PRIVATE "src")
target_link_libraries(gemm_bench PRIVATE alpaka::alpaka)
alpaka_finalize(gemm_bench)

get_filename_component(GEMM_CHECKOUT_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
target_compile_definitions(
    gemm_bench
    PRIVATE GEMM_DEFAULT_MANIFEST="${CMAKE_CURRENT_LIST_DIR}/workloads.json" GEMM_SOURCE_DIR="${GEMM_CHECKOUT_ROOT}"
)

option(GEMM_BENCH_WERROR "Treat warnings in the GEMM benchmark harness as errors." OFF)
# Host warning flags must not reach nvcc: nvcc's -Werror takes a warning list and
# rejects the following CUDA option when these flags are used for CUDA sources.
if(alpaka_COMPILE_PEDANTIC OR GEMM_BENCH_WERROR)
    target_compile_options(gemm_bench PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic;-Werror>")
else()
    target_compile_options(gemm_bench PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>")
endif()

alpaka_gemm_apply_native_isa(gemm_bench)
set_target_properties(gemm_bench PROPERTIES FOLDER benchmarks/gemm)

# Run as bounded ctest entries. The full manifest is a benchmark, not a CI test.
# Build-tree output is outside the source checkout for out-of-source builds, so the
# driver's outside-checkout guard stays satisfied without weakening it.
set(GEMM_BENCH_CTEST_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/gemm-bench-out")
file(MAKE_DIRECTORY "${GEMM_BENCH_CTEST_OUTPUT}/full" "${GEMM_BENCH_CTEST_OUTPUT}/smoke")
add_test(
    NAME gemm_bench
    COMMAND gemm_bench --subset ci-fast --filter square-64 --max-cases 1 --output-dir "${GEMM_BENCH_CTEST_OUTPUT}/full"
)
add_test(
    NAME gemm_bench_smoke
    COMMAND
        gemm_bench --smoke --subset ci-fast --filter tiny-1x1x1 --max-cases 1 --output-dir
        "${GEMM_BENCH_CTEST_OUTPUT}/smoke"
)
set_tests_properties(gemm_bench gemm_bench_smoke PROPERTIES TIMEOUT 30)

# Optional benchmark-only reference baseline (default OFF, no fetch when off).
add_subdirectory(vendor)
