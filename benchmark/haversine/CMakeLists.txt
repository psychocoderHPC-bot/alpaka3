#
# Copyright 2026 René Widera
# SPDX-License-Identifier: ISC
#
cmake_minimum_required(VERSION 3.25)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
project(haversine LANGUAGES CXX)

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

# alpaka version of haversine
add_executable(haversine_alpaka "src/main_alpaka.cpp")
target_include_directories(haversine_alpaka PRIVATE "src")
target_link_libraries(haversine_alpaka PUBLIC alpaka::alpaka)
target_link_libraries(haversine_alpaka PRIVATE Catch2::Catch2WithMain Catch2::Catch2)
alpaka_finalize(haversine_alpaka)

# Fast correctness subset; exercises a small irregular size plus a nonzero seed.
add_test(NAME haversine_alpaka COMMAND "haversine_alpaka" --cities 100000 --repeat 4 --samples 2 --seed 7)
