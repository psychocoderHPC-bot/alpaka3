#
# Copyright 2026 René Widera
# SPDX-License-Identifier: ISC
#
cmake_minimum_required(VERSION 3.25)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
project(memcpy LANGUAGES CXX)

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

# alpaka version of memcpy
add_executable(memcpy_alpaka "src/main_alpaka.cpp")
target_include_directories(memcpy_alpaka PRIVATE "src")
target_link_libraries(memcpy_alpaka PUBLIC alpaka::alpaka)
target_link_libraries(memcpy_alpaka PRIVATE Catch2::Catch2WithMain Catch2::Catch2)
alpaka_finalize(memcpy_alpaka)

add_test(NAME memcpy_alpaka COMMAND "memcpy_alpaka" --sizes 64,1000,65536 --repeat 4 --samples 2)
