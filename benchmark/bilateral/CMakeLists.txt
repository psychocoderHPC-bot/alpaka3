#
# Copyright 2026 René Widera
# SPDX-License-Identifier: ISC
#
cmake_minimum_required(VERSION 3.25)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
project(bilateral LANGUAGES CXX)

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

# alpaka version of bilateral
add_executable(bilateral_alpaka "src/main_alpaka.cpp")
target_include_directories(bilateral_alpaka PRIVATE "src")
target_link_libraries(bilateral_alpaka PUBLIC alpaka::alpaka)
target_link_libraries(bilateral_alpaka PRIVATE Catch2::Catch2WithMain Catch2::Catch2)
alpaka_finalize(bilateral_alpaka)

# The full default run (2960x1440, repeat=1000) is for measurement. The registered test uses a small
# image and validation-only mode so it stays fast on a CPU backend while still exercising all three
# reference radii and the host reference.
add_test(NAME bilateral_alpaka COMMAND "bilateral_alpaka" --width 128 --height 96 --verify-only)
