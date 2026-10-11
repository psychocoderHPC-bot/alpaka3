#
# Copyright 2026 René Widera
# SPDX-License-Identifier: ISC
#
cmake_minimum_required(VERSION 3.25)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
project(histogram LANGUAGES CXX)

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

# alpaka version of histogram
add_executable(histogram_alpaka "src/main_alpaka.cpp")
target_include_directories(histogram_alpaka PRIVATE "src")
target_link_libraries(histogram_alpaka PUBLIC alpaka::alpaka)
target_link_libraries(histogram_alpaka PRIVATE Catch2::Catch2WithMain Catch2::Catch2)
alpaka_finalize(histogram_alpaka)

add_test(NAME histogram_alpaka COMMAND "histogram_alpaka" --pixels 100000 --repeat 4 --samples 2)
