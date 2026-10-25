#
# Copyright 2026 alpaka authors
# SPDX-License-Identifier: MPL-2.0
#
cmake_minimum_required(VERSION 3.25)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
project(alpakaGemmBenchmark LANGUAGES CXX)

# The harness itself is portable C++20 and does not require a GPU backend to
# compile. The real GEMM implementation plugs in behind GemmFunctor later.
add_executable(gemm_bench "src/gemm_bench.cpp" "src/json.cpp")
target_compile_features(gemm_bench PRIVATE cxx_std_20)
target_include_directories(gemm_bench PRIVATE "src")

get_filename_component(GEMM_CHECKOUT_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
target_compile_definitions(
    gemm_bench
    PRIVATE GEMM_DEFAULT_MANIFEST="${CMAKE_CURRENT_LIST_DIR}/workloads.json" GEMM_SOURCE_DIR="${GEMM_CHECKOUT_ROOT}"
)

option(GEMM_BENCH_WERROR "Treat warnings in the GEMM benchmark harness as errors." OFF)
if(alpaka_COMPILE_PEDANTIC OR GEMM_BENCH_WERROR)
    target_compile_options(gemm_bench PRIVATE -Wall -Wextra -Wpedantic -Werror)
else()
    target_compile_options(gemm_bench PRIVATE -Wall -Wextra -Wpedantic)
endif()

set_target_properties(gemm_bench PROPERTIES FOLDER benchmarks/gemm)
