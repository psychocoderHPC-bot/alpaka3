#
# Copyright 2026 Rene Widera
# SPDX-License-Identifier: ISC
#
cmake_minimum_required(VERSION 3.25)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
project(axpby LANGUAGES CXX)

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

# alpaka version of axpby
add_executable(axpby_alpaka "src/main_alpaka.cpp")
target_include_directories(axpby_alpaka PRIVATE "src")
target_link_libraries(axpby_alpaka PUBLIC alpaka::alpaka)
target_link_libraries(axpby_alpaka PRIVATE Catch2::Catch2WithMain Catch2::Catch2)
alpaka_finalize(axpby_alpaka)

add_test(
    NAME axpby_alpaka
    COMMAND "axpby_alpaka" --tensors 4 --fixed-length 4093 --chunk-size 512 --repeat 2
)
