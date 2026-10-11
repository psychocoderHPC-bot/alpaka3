#
# Copyright 2026 Rene Widera
# SPDX-License-Identifier: ISC
#
cmake_minimum_required(VERSION 3.25)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
project(ising LANGUAGES CXX)

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

# alpaka version of ising
add_executable(ising_alpaka "src/main_alpaka.cpp")
target_include_directories(ising_alpaka PRIVATE "src")
target_link_libraries(ising_alpaka PUBLIC alpaka::alpaka)
target_link_libraries(ising_alpaka PRIVATE Catch2::Catch2WithMain Catch2::Catch2)
alpaka_finalize(ising_alpaka)

# Small fast case exercising the periodic stencil with an irregular (small) lattice.
add_test(NAME ising_alpaka COMMAND "ising_alpaka" --lattice-n 64 --lattice-m 64 --nwarmup 10 --niters 50 --repeat 1)
# Non-square, even lattice to exercise the row/column packing and the odd/even row parity path.
add_test(NAME ising_alpaka_rect COMMAND "ising_alpaka" --lattice-n 34 --lattice-m 22 --nwarmup 5 --niters 30 --repeat 1)
