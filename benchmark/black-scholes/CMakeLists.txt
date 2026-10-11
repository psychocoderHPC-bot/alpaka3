#
# Copyright 2026 René Widera
# SPDX-License-Identifier: ISC
#
cmake_minimum_required(VERSION 3.25)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
project(black-scholes LANGUAGES CXX)

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

# alpaka version of black-scholes
add_executable(black-scholes_alpaka "src/main_alpaka.cpp")
target_include_directories(black-scholes_alpaka PRIVATE "src")
target_link_libraries(black-scholes_alpaka PUBLIC alpaka::alpaka)
target_link_libraries(black-scholes_alpaka PRIVATE Catch2::Catch2WithMain Catch2::Catch2)
alpaka_finalize(black-scholes_alpaka)

# Small fast case, a case that is not a multiple of the block size, and a
# boundary case smaller than one block.
add_test(NAME black-scholes_alpaka_small COMMAND "black-scholes_alpaka" --options 3700 --repeat 20 --samples 2)
add_test(NAME black-scholes_alpaka_odd COMMAND "black-scholes_alpaka" --options 12345 --repeat 20 --samples 2)
add_test(NAME black-scholes_alpaka_boundary COMMAND "black-scholes_alpaka" --options 7 --repeat 20 --samples 2)
