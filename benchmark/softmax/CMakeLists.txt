#
# Copyright 2026 Rene Widera
# SPDX-License-Identifier: ISC
#
cmake_minimum_required(VERSION 3.25)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
project(softmax LANGUAGES CXX)

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

# alpaka version of softmax
add_executable(softmax_alpaka "src/main_alpaka.cpp")
target_include_directories(softmax_alpaka PRIVATE "src")
target_link_libraries(softmax_alpaka PUBLIC alpaka::alpaka)
target_link_libraries(softmax_alpaka PRIVATE Catch2::Catch2WithMain Catch2::Catch2)
alpaka_finalize(softmax_alpaka)

add_test(
    NAME softmax_alpaka
    COMMAND "softmax_alpaka" --size 4096 --slice-size 128 --repeat 2
)
