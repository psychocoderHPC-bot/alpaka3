#
# Copyright 2026 Rene Widera
# SPDX-License-Identifier: ISC
#
cmake_minimum_required(VERSION 3.25)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
project(cross LANGUAGES CXX)

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

# alpaka version of cross
add_executable(cross_alpaka "src/main_alpaka.cpp")
target_include_directories(cross_alpaka PRIVATE "src")
target_link_libraries(cross_alpaka PUBLIC alpaka::alpaka)
target_link_libraries(cross_alpaka PRIVATE Catch2::Catch2WithMain Catch2::Catch2)
alpaka_finalize(cross_alpaka)

add_test(
    NAME cross_alpaka
    COMMAND "cross_alpaka" --size 4096 --repeat 2
)
