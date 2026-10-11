#
# Copyright 2026 René Widera
# SPDX-License-Identifier: ISC
#
cmake_minimum_required(VERSION 3.25)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
project(dct8x8 LANGUAGES CXX)

if(NOT TARGET alpaka::alpaka)
    option(alpaka_USE_SOURCE_TREE "Use alpaka's source tree instead of an alpaka installation" OFF)
    if(alpaka_USE_SOURCE_TREE)
        # Don't build the benchmarks recursively
        set(alpaka_BUILD_BENCHMARKS OFF)
        add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/../.." "${CMAKE_BINARY_DIR}/alpaka")
    else()
        find_package(alpaka REQUIRED)
    endif()
endif()

# alpaka version of dct8x8
add_executable(dct8x8_alpaka "src/main_alpaka.cpp")
target_include_directories(dct8x8_alpaka PRIVATE "src")
target_link_libraries(dct8x8_alpaka PUBLIC alpaka::alpaka)
target_link_libraries(dct8x8_alpaka PRIVATE Catch2::Catch2WithMain Catch2::Catch2)
alpaka_finalize(dct8x8_alpaka)

# Small irregular image plus a boundary case (a bare 8x8 block and a non-multiple size) and the
# 8192x8192 measured workload; both transform directions are exercised.
add_test(NAME dct8x8_alpaka COMMAND "dct8x8_alpaka" --width 999 --height 777 --repeat 20 --samples 2)
add_test(NAME dct8x8_alpaka_inverse COMMAND "dct8x8_alpaka" --width 999 --height 777 --repeat 20 --samples 2 --inverse)
