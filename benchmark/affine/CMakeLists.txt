#
# Copyright 2026 René Widera
# SPDX-License-Identifier: ISC
#
cmake_minimum_required(VERSION 3.25)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
project(affine LANGUAGES CXX)

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

# alpaka version of affine
add_executable(affine_alpaka "src/main_alpaka.cpp")
target_include_directories(affine_alpaka PRIVATE "src")
target_link_libraries(affine_alpaka PUBLIC alpaka::alpaka)
target_link_libraries(affine_alpaka PRIVATE Catch2::Catch2WithMain Catch2::Catch2)
alpaka_finalize(affine_alpaka)

# Fast correctness subset; exercises an irregular size plus the pinned default.
add_test(NAME affine_alpaka COMMAND "affine_alpaka" --width 127 --height 129 --repeat 4 --samples 2)
