/* Copyright 2026 René Widera
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

/** @file Command line parsing helpers for the mandelbrot benchmark.
 * @attention The license differs from the benchmark implementation because this file contains no parts of the
 * HeCBench code base.
 */

#include "helper.hpp"

#include <catch2/catch_session.hpp>

#include <cstdlib>
#include <iostream>

namespace mandel
{
    /** Parse the command line for size, iteration limit and repetitions.
     *
     * Defaults replicate the HeCBench reference program (`row_size = 1024`, `col_size = 1024`,
     * `max_iterations = 1000`); the number of repetitions mirrors the reference `./main <repeat>`
     * argument.
     */
    inline int parseCmd(int argc, char* argv[], int& rows, int& cols, int& maxIterations, int& repetitions)
    {
        rows = defaultRowSize;
        cols = defaultColSize;
        maxIterations = defaultMaxIterations;
        repetitions = defaultRepetitions;

        Catch::Session session;
        using namespace Catch::Clara;

        auto cli = session.cli() | Opt(rows, "rows")["--rows"]("Number of rows of the image")
                   | Opt(cols, "cols")["--cols"]("Number of columns of the image")
                   | Opt(maxIterations, "maxIterations")["--max-iter"]("Iteration limit")
                   | Opt(repetitions, "repetitions")["--repeat"]("Number of measured repetitions");

        session.cli(cli);

        int const rc = session.applyCommandLine(argc, argv);
        if(rc != 0)
            return rc;

        if(rows <= 0 || cols <= 0)
        {
            std::cerr << "Error: rows and cols must be greater than zero.\n";
            return EXIT_FAILURE;
        }
        if(maxIterations <= 0)
        {
            std::cerr << "Error: max iterations must be greater than zero.\n";
            return EXIT_FAILURE;
        }
        if(repetitions <= 0)
        {
            std::cerr << "Error: repetitions must be greater than zero.\n";
            return EXIT_FAILURE;
        }
        return EXIT_SUCCESS;
    }
} // namespace mandel
