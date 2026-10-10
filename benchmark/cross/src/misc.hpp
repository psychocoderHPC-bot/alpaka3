/* Copyright 2026 Rene Widera
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

/** @file Command line parsing helpers for the cross benchmark.
 * @attention The license differs from the benchmark implementation because this file contains no parts of the
 * HeCBench code base.
 */

#include <catch2/catch_session.hpp>

#include <cstdlib>
#include <iostream>
#include <string>

namespace cross
{
    constexpr int defaultNumRows = 10'000'000;
    constexpr int defaultRepeat = 100;

    /** Parse the command line and clamp the parameters to sane values. */
    inline int parseCmd(int argc, char* argv[], int& numRows, int& repeat, std::string& dtype)
    {
        numRows = defaultNumRows;
        repeat = defaultRepeat;
        dtype = "both";

        Catch::Session session;
        using namespace Catch::Clara;

        auto cli = session.cli()
                   | Opt(numRows, "numRows")["--size"]("Number of rows, i.e. number of 3D vectors in each tensor")
                   | Opt(repeat, "repeat")["--repeat"]("Number of measured repetitions")
                   | Opt(dtype, "dtype")["--dtype"]("Data type: float, double or both");

        session.cli(cli);

        int const rc = session.applyCommandLine(argc, argv);
        if(rc != 0)
            return rc;

        if(numRows <= 0)
        {
            std::cerr << "Error: number of rows must be greater than zero.\n";
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
} // namespace cross
