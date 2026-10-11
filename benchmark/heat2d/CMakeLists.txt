#
# Copyright 2026 René Widera
# SPDX-License-Identifier: ISC
#
cmake_minimum_required(VERSION 3.25)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
project(heat2d LANGUAGES CXX)

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

# alpaka version of heat2d
add_executable(heat2d_alpaka "src/main_alpaka.cpp")
target_include_directories(heat2d_alpaka PRIVATE "src")
target_link_libraries(heat2d_alpaka PUBLIC alpaka::alpaka)
target_link_libraries(heat2d_alpaka PRIVATE Catch2::Catch2WithMain Catch2::Catch2)
alpaka_finalize(heat2d_alpaka)

# Small fast case that still exercises the periodic stencil, the swap and the host reference.
add_test(
    NAME heat2d_alpaka
    COMMAND "heat2d_alpaka" --lx 64 --ly 64 --iters 8 --repeat 2
)
