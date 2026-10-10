#
# Copyright 2026 René Widera
# SPDX-License-Identifier: ISC
#
cmake_minimum_required(VERSION 3.25)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
project(mandelbrot LANGUAGES CXX)

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

# alpaka version of mandelbrot
add_executable(mandelbrot_alpaka "src/main_alpaka.cpp")
target_include_directories(mandelbrot_alpaka PRIVATE "src")
target_link_libraries(mandelbrot_alpaka PUBLIC alpaka::alpaka)
target_link_libraries(mandelbrot_alpaka PRIVATE Catch2::Catch2WithMain Catch2::Catch2)
alpaka_finalize(mandelbrot_alpaka)

add_test(NAME mandelbrot_alpaka COMMAND "mandelbrot_alpaka" --rows 64 --cols 64 --max-iter 100 --repeat 2)
