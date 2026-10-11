#
# Copyright 2026 Rene Widera
# SPDX-License-Identifier: ISC
#
cmake_minimum_required(VERSION 3.25)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
project(matrix-rotate LANGUAGES CXX)

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

# alpaka version of matrix-rotate
add_executable(matrix-rotate_alpaka "src/main_alpaka.cpp")
target_include_directories(matrix-rotate_alpaka PRIVATE "src")
target_link_libraries(matrix-rotate_alpaka PUBLIC alpaka::alpaka)
target_link_libraries(matrix-rotate_alpaka PRIVATE Catch2::Catch2WithMain Catch2::Catch2)
alpaka_finalize(matrix-rotate_alpaka)

# A small odd size exercises the pyramid-ring addressing and the exact comparison.
add_test(
    NAME matrix-rotate_alpaka
    COMMAND "matrix-rotate_alpaka" --size 17 --repeat 3
)
