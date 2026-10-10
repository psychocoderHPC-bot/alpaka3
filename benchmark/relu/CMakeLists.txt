#
# Copyright 2026 Rene Widera
# SPDX-License-Identifier: ISC
#
cmake_minimum_required(VERSION 3.25)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
project(relu LANGUAGES CXX)

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

# alpaka version of relu
add_executable(relu_alpaka "src/main_alpaka.cpp")
target_include_directories(relu_alpaka PRIVATE "src")
target_link_libraries(relu_alpaka PUBLIC alpaka::alpaka)
target_link_libraries(relu_alpaka PRIVATE Catch2::Catch2WithMain Catch2::Catch2)
alpaka_finalize(relu_alpaka)

add_test(NAME relu_alpaka COMMAND "relu_alpaka" --size 4096 --repeat 2)
# Odd size that is not a multiple of the SIMD frame width, exercising the scalar remainder path of both kernels.
add_test(NAME relu_alpaka_odd COMMAND "relu_alpaka" --size 257 --repeat 1)
