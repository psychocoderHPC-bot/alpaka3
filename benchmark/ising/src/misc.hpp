/* Copyright 2026 Rene Widera
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

/** @file Command line parsing helpers for the ising benchmark.
 * @attention The license differs from the benchmark implementation because this file contains no parts of the
 * HeCBench code base.
 */

#include <catch2/catch_session.hpp>

#include <cstdlib>
#include <iostream>

namespace ising
{
    //! HeCBench defaults, matching `src/ising-cuda/main.cu`.
    constexpr int defaultNx = 5120;
    constexpr int defaultNy = 5120;
    constexpr int defaultNwarmup = 100;
    constexpr int defaultNiters = 1000;
    constexpr double defaultAlpha = 0.1;
    constexpr unsigned long long defaultSeed = 1234ULL;
    constexpr int defaultRepeat = 5;

    struct Params
    {
        int nx = defaultNx;
        int ny = defaultNy;
        int nwarmup = defaultNwarmup;
        int niters = defaultNiters;
        int repeat = defaultRepeat;
        double alpha = defaultAlpha;
        unsigned long long seed = defaultSeed;
    };

    /** Parse the command line and validate the parameters. */
    inline int parseCmd(int argc, char* argv[], Params& params)
    {
        Catch::Session session;
        using namespace Catch::Clara;

        auto cli = session.cli() | Opt(params.nx, "lattice-n")["--lattice-n"]("Number of lattice rows (even)")
                   | Opt(params.ny, "lattice-m")["--lattice-m"]("Number of lattice columns (even)")
                   | Opt(params.nwarmup, "nwarmup")["--nwarmup"]("Number of warm-up iterations")
                   | Opt(params.niters, "niters")["--niters"]("Number of trial iterations")
                   | Opt(params.repeat, "repeat")["--repeat"]("Number of measured repetitions")
                   | Opt(params.alpha, "alpha")["--alpha"]("Coefficient of the critical temperature")
                   | Opt(params.seed, "seed")["--seed"]("Seed for the random number generation");

        session.cli(cli);

        int const rc = session.applyCommandLine(argc, argv);
        if(rc != 0)
            return rc;

        if(params.nx <= 0 || params.ny <= 0)
        {
            std::cerr << "Error: lattice dimensions must be greater than zero.\n";
            return EXIT_FAILURE;
        }
        if(params.nx % 2 != 0 || params.ny % 2 != 0)
        {
            std::cerr << "Error: lattice dimensions must be even values.\n";
            return EXIT_FAILURE;
        }
        if(params.nwarmup < 0 || params.niters <= 0)
        {
            std::cerr << "Error: nwarmup must be non-negative and niters positive.\n";
            return EXIT_FAILURE;
        }
        if(params.repeat <= 0)
        {
            std::cerr << "Error: repeat must be greater than zero.\n";
            return EXIT_FAILURE;
        }
        if(params.alpha <= 0.0)
        {
            std::cerr << "Error: alpha must be greater than zero.\n";
            return EXIT_FAILURE;
        }
        return EXIT_SUCCESS;
    }
} // namespace ising
