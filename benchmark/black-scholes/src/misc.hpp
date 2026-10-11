/* Copyright 2026 René Widera
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

/** @file Command line parsing helpers for the black-scholes benchmark.
 * @attention The license differs from the benchmark implementation because this file contains no
 * parts of the HeCBench code base.
 */

#include <catch2/catch_session.hpp>

#include <cstdint>
#include <cstdlib>
#include <iostream>

namespace blackScholesBench
{
    //! Reference `main` hardcodes 50,000,000 options per run.
    constexpr int defaultOptions = 50000000;
    //! Reference `run` target invokes `./main 100`.
    constexpr int defaultRepeat = 100;
    //! Number of independent measured samples.
    constexpr int defaultSamples = 5;
    //! Tolerance of the ported validation: the reference's own absolute tolerance `1.0e-4`
    //! against the published analytic prices. See README for the measured margin.
    constexpr double defaultTolerance = 1.0e-4;

    struct Params
    {
        int options = defaultOptions;
        int repeat = defaultRepeat;
        int samples = defaultSamples;
        //! Restrict the run to validation only (no timing loop).
        bool verifyOnly = false;
    };

    /** Parse the command line.
     *
     * The reference takes the positional `<repeat>` and hardcodes the option count. This port uses
     * named options and generates the option stream internally.
     */
    inline int parseCmd(int argc, char* argv[], Params& params)
    {
        Catch::Session session;
        using namespace Catch::Clara;

        auto cli = session.cli() | Opt(params.options, "options")["--options"]("Number of priced options")
                   | Opt(params.repeat, "repeat")["--repeat"]("Kernel repetitions per measured sample")
                   | Opt(params.samples, "samples")["--samples"]("Independent measured samples")
                   | Opt(params.verifyOnly)["--verify-only"]("Only run the correctness check, no timing");

        session.cli(cli);

        int const rc = session.applyCommandLine(argc, argv);
        if(rc != 0)
            return rc;

        if(params.options <= 0)
        {
            std::cerr << "Error: options must be greater than zero.\n";
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

} // namespace blackScholesBench
