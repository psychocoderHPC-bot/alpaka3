#
# Copyright 2026 René Widera
# SPDX-License-Identifier: ISC
#
cmake_minimum_required(VERSION 3.25)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
project(morphology LANGUAGES CXX)

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

# alpaka version of morphology
add_executable(morphology_alpaka "src/main_alpaka.cpp")
target_include_directories(morphology_alpaka PRIVATE "src")
target_link_libraries(morphology_alpaka PUBLIC alpaka::alpaka)
target_link_libraries(morphology_alpaka PRIVATE Catch2::Catch2WithMain Catch2::Catch2)
alpaka_finalize(morphology_alpaka)

add_test(NAME morphology_alpaka COMMAND "morphology_alpaka" --width 129 --height 127 --hsize 5 --vsize 5 --repeat 2 --samples 2)
# Boundary case for the maximum supported structuring element and the per-thread deque capacity.
add_test(
    NAME morphology_alpaka_hsize256
    COMMAND "morphology_alpaka" --width 129 --height 127 --hsize 256 --vsize 256 --repeat 2 --samples 1 --seed 12345
)
