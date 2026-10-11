/* Copyright 2026 René Widera
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

/** @file Command line parsing helpers for the HeCBench `laplace3d` benchmark.
 * @attention The license differs from the benchmark implementation because this file contains no parts of the
 * HeCBench code base.
 */

#include <catch2/catch_session.hpp>

#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <iostream>

namespace laplace3d
{
    //! HeCBench `make run` uses `./main 128 128 128 100 1`; keep those as defaults.
    constexpr int defaultNx = 128;
    constexpr int defaultNy = 128;
    constexpr int defaultNz = 128;
    constexpr int defaultIters = 100;
    //! Number of measured timed runs; the HeCBench reference reports one value per invocation.
    constexpr int defaultRepeat = 5;

    //! Benchmark configuration handed to the example function.
    struct Config
    {
        int nx = defaultNx;
        int ny = defaultNy;
        int nz = defaultNz;
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
     * Accepts the HeCBench positional form (`<Nx> <Ny> <Nz> <niter>`) and the named form
     * (`--nx N --ny N --nz N --iters N --repeat R`). The positional form is parsed before Catch2
     * sees the arguments; named options use the Catch2/Clara parser. Unlike the reference, arbitrary
     * dimensions are accepted: the reference rejects non-multiples of its 32x8 launch block, which
     * is a native-launch detail the portable alpaka launch does not require.
     */
    inline int parseCmd(int argc, char* argv[], Config& cfg)
    {
        cfg = Config{};

        if(argc >= 5 && detail::allDigits(argv[1]) && detail::allDigits(argv[2]) && detail::allDigits(argv[3])
           && detail::allDigits(argv[4]))
        {
            cfg.nx = std::atoi(argv[1]);
            cfg.ny = std::atoi(argv[2]);
            cfg.nz = std::atoi(argv[3]);
            cfg.iters = std::atoi(argv[4]);
            if(argc >= 6 && detail::allDigits(argv[5]))
                cfg.repeat = std::atoi(argv[5]);
            if(cfg.nx <= 0 || cfg.ny <= 0 || cfg.nz <= 0 || cfg.iters < 0 || cfg.repeat <= 0)
            {
                std::cerr << "Error: Nx, Ny, Nz and repeat must be greater than zero; niter must not be negative.\n";
                return EXIT_FAILURE;
            }
            return EXIT_SUCCESS;
        }

        Catch::Session session;
        using namespace Catch::Clara;

        auto cli = session.cli() | Opt(cfg.nx, "nx")["--nx"]("Grid size in x")
                   | Opt(cfg.ny, "ny")["--ny"]("Grid size in y")
                   | Opt(cfg.nz, "nz")["--nz"]("Grid size in z")
                   | Opt(cfg.iters, "iters")["--iters"]("Number of Jacobi iterations")
                   | Opt(cfg.repeat, "repeat")["--repeat"]("Number of measured timed runs");

        session.cli(cli);

        int const rc = session.applyCommandLine(argc, argv);
        if(rc != 0)
            return rc;

        if(cfg.nx <= 0 || cfg.ny <= 0 || cfg.nz <= 0)
        {
            std::cerr << "Error: Nx, Ny and Nz must be greater than zero.\n";
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
} // namespace laplace3d
