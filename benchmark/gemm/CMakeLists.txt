#
# Copyright 2026 alpaka authors
# SPDX-License-Identifier: MPL-2.0
#
cmake_minimum_required(VERSION 3.25)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
project(alpakaGemmBenchmark LANGUAGES CXX)

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

set_target_properties(gemm_bench PROPERTIES FOLDER benchmarks/gemm)

# Run as bounded ctest entries. The full manifest is a benchmark, not a CI test.
add_test(
    NAME gemm_bench
    COMMAND gemm_bench --subset ci-fast --filter square-64 --output-dir "${CMAKE_CURRENT_BINARY_DIR}/gemm_bench_ctest"
)
add_test(
    NAME gemm_bench_smoke
    COMMAND
        gemm_bench --smoke --subset ci-fast --filter tiny-1x1x1 --output-dir
        "${CMAKE_CURRENT_BINARY_DIR}/gemm_bench_smoke_ctest"
)

# Optional benchmark-only reference baseline (default OFF, no fetch when off).
add_subdirectory(vendor)
