/* Copyright 2026 René Widera
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

/** @file Command line parsing helpers for the HeCBench `heat2d` benchmark.
 * @attention The license differs from the benchmark implementation because this file contains no parts of the
 * HeCBench code base.
 */

#include <catch2/catch_session.hpp>

#include <cctype>
#include <cstdlib>
#include <iostream>
#include <string>

namespace heat2d
{
    //! HeCBench `make run` uses `./main 4096 4096 1000`; keep those as defaults.
    constexpr int defaultLx = 4096;
    constexpr int defaultLy = 4096;
    constexpr int defaultIters = 1000;
    //! Number of measured timed runs; the HeCBench reference reports one value per invocation.
    constexpr int defaultRepeat = 5;

    //! Benchmark configuration handed to the example function.
    struct Config
    {
        int lx = defaultLx;
        int ly = defaultLy;
        int iters = defaultIters;
        int repeat = defaultRepeat;
    };

    namespace detail
    {
        inline bool allDigits(char const* s)
        {
            if(s == nullptr || *s == '\0')
                return false;
            for(char const* p = s; *p != '\0'; ++p)
                if(!std::isdigit(static_cast<unsigned char>(*p)))
                    return false;
            return true;
        }
    } // namespace detail

    /** Parse the command line.
     *
     * Accepts the HeCBench positional form (`<Lx> <Ly> <niter>`) and the named form
     * (`--lx L --ly L --iters N --repeat R`). The positional form is parsed before Catch2 sees the
     * arguments; named options use the Catch2/Clara parser. Unlike the reference, arbitrary
     * dimensions are accepted: the reference only rejects non-multiples of its 16x16 block size for
     * launch reasons, which the portable alpaka launch does not require.
     */
    inline int parseCmd(int argc, char* argv[], Config& cfg)
    {
        cfg = Config{};

        if(argc >= 4 && detail::allDigits(argv[1]) && detail::allDigits(argv[2]) && detail::allDigits(argv[3]))
        {
            cfg.lx = std::atoi(argv[1]);
            cfg.ly = std::atoi(argv[2]);
            cfg.iters = std::atoi(argv[3]);
            if(argc >= 5 && detail::allDigits(argv[4]))
                cfg.repeat = std::atoi(argv[4]);
            if(cfg.lx <= 0 || cfg.ly <= 0 || cfg.iters < 0 || cfg.repeat <= 0)
            {
                std::cerr << "Error: Lx, Ly and repeat must be greater than zero; niter must not be negative.\n";
                return EXIT_FAILURE;
            }
            return EXIT_SUCCESS;
        }

        Catch::Session session;
        using namespace Catch::Clara;

        auto cli = session.cli() | Opt(cfg.lx, "lx")["--lx"]("Grid size in x")
                   | Opt(cfg.ly, "ly")["--ly"]("Grid size in y")
                   | Opt(cfg.iters, "iters")["--iters"]("Number of heat-equation iterations")
                   | Opt(cfg.repeat, "repeat")["--repeat"]("Number of measured timed runs");

        session.cli(cli);

        int const rc = session.applyCommandLine(argc, argv);
        if(rc != 0)
            return rc;

        if(cfg.lx <= 0 || cfg.ly <= 0)
        {
            std::cerr << "Error: Lx and Ly must be greater than zero.\n";
            return EXIT_FAILURE;
        }
        if(cfg.iters < 0)
        {
            std::cerr << "Error: iters must not be negative.\n";
            return EXIT_FAILURE;
        }
        if(cfg.repeat <= 0)
        {
            std::cerr << "Error: repeat must be greater than zero.\n";
            return EXIT_FAILURE;
        }
        return EXIT_SUCCESS;
    }
} // namespace heat2d
