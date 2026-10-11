#
# Copyright 2026 René Widera
# SPDX-License-Identifier: ISC
#
cmake_minimum_required(VERSION 3.25)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
project(jacobi LANGUAGES CXX)

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

# alpaka version of jacobi
add_executable(jacobi_alpaka "src/main_alpaka.cpp")
target_include_directories(jacobi_alpaka PRIVATE "src")
target_link_libraries(jacobi_alpaka PUBLIC alpaka::alpaka)
target_link_libraries(jacobi_alpaka PRIVATE Catch2::Catch2WithMain Catch2::Catch2)
alpaka_finalize(jacobi_alpaka)

# Small fast case that still exercises the full convergence loop and the host reference.
add_test(NAME jacobi_alpaka COMMAND "jacobi_alpaka" --n 64 --max-iters 2000 --repeat 2)
