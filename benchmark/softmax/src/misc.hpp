/* Copyright 2026 Rene Widera
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

/** @file Command line parsing helpers for the softmax benchmark.
 * @attention The license differs from the benchmark implementation because this file contains no parts of the
 * HeCBench code base.
 */

#include <catch2/catch_session.hpp>

#include <cstdlib>
#include <iostream>
#include <string>

namespace softmax
{
    // Defaults from the HeCBench run recipe: `<numSlice> <sliceSize> <implementation> <repeat>` with
    // 100000 784 0 100 (see src/softmax-cuda/Makefile and benchmarks.yaml at the pinned SHA).
    constexpr int defaultNumSlice = 100'000;
    constexpr int defaultSliceSize = 784;
    constexpr int defaultRepeat = 100;

    /** Parse the command line and clamp the parameters to sane values. */
    inline int parseCmd(int argc, char* argv[], int& numSlice, int& sliceSize, int& repeat, std::string& dtype)
    {
        numSlice = defaultNumSlice;
        sliceSize = defaultSliceSize;
        repeat = defaultRepeat;
        dtype = "float";

        Catch::Session session;
        using namespace Catch::Clara;

        auto cli = session.cli()
                   | Opt(numSlice, "numSlice")["--size"]("Number of independent rows (slices)")
                   | Opt(sliceSize, "sliceSize")["--slice-size"]("Number of elements per row (slice)")
                   | Opt(repeat, "repeat")["--repeat"]("Number of measured repetitions")
                   | Opt(dtype, "dtype")["--dtype"]("Data type: float, double or both");

        session.cli(cli);

        int const rc = session.applyCommandLine(argc, argv);
        if(rc != 0)
            return rc;

        if(numSlice <= 0)
        {
            std::cerr << "Error: number of slices must be greater than zero.\n";
            return EXIT_FAILURE;
        }
        if(sliceSize <= 0)
        {
            std::cerr << "Error: slice size must be greater than zero.\n";
            return EXIT_FAILURE;
        }
        if(repeat <= 0)
        {
            std::cerr << "Error: number of repetitions must be greater than zero.\n";
            return EXIT_FAILURE;
        }
        // A dedicated warm-up enqueue is performed before the measured iterations, see timeKernel().
        return EXIT_SUCCESS;
    }
} // namespace softmax
