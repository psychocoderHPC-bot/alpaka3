/* Copyright 2026 René Widera
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

/** @file Command line parsing helpers for the HeCBench `gaussian` benchmark.
 * @attention The license differs from the benchmark implementation because this file contains no parts of the
 * HeCBench code base.
 */

#include <catch2/catch_session.hpp>

#include <cstdint>
#include <cstdlib>
#include <iostream>

namespace gaussianBench
{
    //! Default matrix size. The HeCBench `benchmarks.yaml` regression uses `-s 4096`.
    constexpr std::uint32_t defaultSize = 1024u;
    //! Inner forward eliminations per measured sample.
    constexpr int defaultRepeat = 1;
    //! Number of independent measured samples.
    constexpr int defaultSamples = 5;

    struct Params
    {
        //! Dimension of the square matrix / right-hand side.
        std::uint32_t size = defaultSize;
        //! Forward-elimination runs per measured sample.
        int repeat = defaultRepeat;
        //! Independent measured samples.
        int samples = defaultSamples;
        //! Restrict the run to the correctness check (no timing).
        bool verifyOnly = false;
        //! Print the full solution (debugging aid).
        bool verbose = false;
    };

    /** Parse the command line.
     *
     * The reference `-s N` form is accepted through `--size N`; `--size`/`-s` both work.
     */
    inline int parseCmd(int argc, char* argv[], Params& params)
    {
        Catch::Session session;
        using namespace Catch::Clara;

        auto cli = session.cli() | Opt(params.size, "size")["--size"]["-s"]("Matrix dimension (square, N x N)")
                   | Opt(params.repeat, "repeat")["--repeat"]("Forward eliminations per measured sample")
                   | Opt(params.samples, "samples")["--samples"]("Independent measured samples")
                   | Opt(params.verifyOnly)["--verify-only"]("Only run the correctness check, no timing")
                   | Opt(params.verbose)["--verbose"]("Print the full solution vector");

        session.cli(cli);

        int const rc = session.applyCommandLine(argc, argv);
        if(rc != 0)
            return rc;

        if(params.size == 0u)
        {
            std::cerr << "Error: matrix size must be greater than zero.\n";
            return EXIT_FAILURE;
        }
        if(params.repeat <= 0)
        {
            std::cerr << "Error: repeat must be greater than zero.\n";
            return EXIT_FAILURE;
        }
        if(params.samples <= 0)
        {
            std::cerr << "Error: samples must be greater than zero.\n";
            return EXIT_FAILURE;
        }

        return EXIT_SUCCESS;
    }
} // namespace gaussianBench
