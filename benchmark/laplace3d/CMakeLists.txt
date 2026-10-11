#
# Copyright 2026 René Widera
# SPDX-License-Identifier: ISC
#
cmake_minimum_required(VERSION 3.25)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
project(laplace3d LANGUAGES CXX)

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

# alpaka version of laplace3d
add_executable(laplace3d_alpaka "src/main_alpaka.cpp")
target_include_directories(laplace3d_alpaka PRIVATE "src")
target_link_libraries(laplace3d_alpaka PUBLIC alpaka::alpaka)
target_link_libraries(laplace3d_alpaka PRIVATE Catch2::Catch2WithMain Catch2::Catch2)
alpaka_finalize(laplace3d_alpaka)

# Small fast case that still exercises the seven-point stencil, the Dirichlet boundary and the swap.
add_test(
    NAME laplace3d_alpaka
    COMMAND "laplace3d_alpaka" --nx 64 --ny 32 --nz 16 --iters 8 --repeat 2
)
