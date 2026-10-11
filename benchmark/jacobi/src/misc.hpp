/* Copyright 2026 René Widera
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

/** @file Command line parsing helpers for the jacobi benchmark.
 * @attention The license differs from the benchmark implementation because this file contains no parts of the
 * HeCBench code base.
 */

#include <catch2/catch_session.hpp>

#include <cctype>
#include <cstdlib>
#include <iostream>
#include <string>

namespace jacobi
{
    //! HeCBench `jacobi` hard codes `#define N 2048`; keep it as the default.
    constexpr int defaultN = 2048;
    //! HeCBench tolerance `1.e-5f`.
    constexpr float defaultTolerance = 1.e-5f;
    //! HeCBench iteration cap `max_iters = 10000`.
    constexpr int defaultMaxIters = 10000;
    //! Number of measured full solves; the HeCBench reference prints one value per invocation.
    constexpr int defaultRepeat = 5;

    //! Benchmark configuration handed to the example function.
    struct Config
    {
        int n = defaultN;
        float tolerance = defaultTolerance;
        int maxIters = defaultMaxIters;
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
     * Accepts a positional form (`<n> [repeat]`) and named options (`--n N --tolerance T --max-iters M
     * --repeat R`). The positional form is parsed before Catch2 sees the arguments; named options use
     * the Catch2/Clara parser.
     */
    inline int parseCmd(int argc, char* argv[], Config& cfg)
    {
        cfg = Config{};

        if(argc >= 2 && detail::allDigits(argv[1]))
        {
            cfg.n = std::atoi(argv[1]);
            if(argc >= 3 && detail::allDigits(argv[2]))
                cfg.repeat = std::atoi(argv[2]);
            if(cfg.n < 3 || cfg.repeat <= 0)
            {
                std::cerr << "Error: n must be >= 3 and repeat must be greater than zero.\n";
                return EXIT_FAILURE;
            }
            return EXIT_SUCCESS;
        }

        Catch::Session session;
        using namespace Catch::Clara;

        auto cli = session.cli() | Opt(cfg.n, "n")["--n"]("Grid side length")
                   | Opt(cfg.tolerance, "tolerance")["--tolerance"]("L2 convergence tolerance")
                   | Opt(cfg.maxIters, "max-iters")["--max-iters"]("Maximum number of Jacobi iterations")
                   | Opt(cfg.repeat, "repeat")["--repeat"]("Number of measured full solves");

        session.cli(cli);

        int const rc = session.applyCommandLine(argc, argv);
        if(rc != 0)
            return rc;

        if(cfg.n < 3)
        {
            std::cerr << "Error: n must be >= 3.\n";
            return EXIT_FAILURE;
        }
        if(!(cfg.tolerance > 0.0f))
        {
            std::cerr << "Error: tolerance must be greater than zero.\n";
            return EXIT_FAILURE;
        }
        if(cfg.maxIters <= 0)
        {
            std::cerr << "Error: max-iters must be greater than zero.\n";
            return EXIT_FAILURE;
        }
        if(cfg.repeat <= 0)
        {
            std::cerr << "Error: repeat must be greater than zero.\n";
            return EXIT_FAILURE;
        }
        return EXIT_SUCCESS;
    }
} // namespace jacobi
