/* Copyright 2026 Rene Widera
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

/** @file Command line parsing helpers for the relu benchmark.
 * @attention The license differs from the benchmark implementation because this file contains no parts of the
 * HeCBench/TensorFlow code base.
 */

#include <catch2/catch_session.hpp>

#include <cstdlib>
#include <iostream>

namespace relu
{
    constexpr int defaultCount = 10000000;
    constexpr int defaultRepeat = 100;

    /** Parse the command line and clamp the parameters to sane values. */
    inline int parseCmd(int argc, char* argv[], int& count, int& repeat)
    {
        count = defaultCount;
        repeat = defaultRepeat;

        Catch::Session session;
        using namespace Catch::Clara;

        auto cli = session.cli()
                   | Opt(count, "count")["--size"]("Number of elements")
                   | Opt(repeat, "repeat")["--repeat"]("Number of measured repetitions");

        session.cli(cli);

        int const rc = session.applyCommandLine(argc, argv);
        if(rc != 0)
            return rc;

        if(count <= 0)
        {
            std::cerr << "Error: number of elements must be greater than zero.\n";
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
} // namespace relu
