/* Copyright 2026 René Widera
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

/** @file Command line parsing helpers for the stddev benchmark.
 * @attention The license differs from the benchmark implementation because this file contains no parts of the
 * HeCBench code base.
 */

#include <catch2/catch_session.hpp>

#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>

namespace stddevBench
{
    //! HeCBench default: number of columns (`D`), a multiple of 32 in the reference.
    constexpr uint32_t defaultD = 65536u;
    //! HeCBench default: number of rows (`N`).
    constexpr uint32_t defaultN = 16384u;
    //! HeCBench default: repeat count of the timed kernel pipeline.
    constexpr int defaultRepeat = 100;
    //! Independent measured samples per configuration.
    constexpr int defaultSamples = 5;

    struct Params
    {
        uint32_t D = defaultD;
        uint32_t N = defaultN;
        int repeat = defaultRepeat;
        int samples = defaultSamples;
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
     * Accepts the HeCBench positional form `stddev_alpaka <D> <N> <repeat>` and the named form.
     * Named options take precedence when present.
     */
    inline int parseCmd(int argc, char* argv[], Params& params)
    {
        // Consume the HeCBench positional form before Catch sees it.
        if(argc == 4 && detail::allDigits(argv[1]) && detail::allDigits(argv[2]) && detail::allDigits(argv[3]))
        {
            params.D = static_cast<uint32_t>(std::strtoul(argv[1], nullptr, 10));
            params.N = static_cast<uint32_t>(std::strtoul(argv[2], nullptr, 10));
            params.repeat = std::atoi(argv[3]);
        }
        else
        {
            Catch::Session session;
            using namespace Catch::Clara;

            auto cli = session.cli() | Opt(params.D, "columns")["--D"]("Number of columns (matrix width)")
                       | Opt(params.N, "rows")["--N"]("Number of rows (matrix height)")
                       | Opt(params.repeat, "repeat")["--repeat"]("Kernel pipeline repetitions per sample")
                       | Opt(params.samples, "samples")["--samples"]("Independent measured samples");

            session.cli(cli);

            int const rc = session.applyCommandLine(argc, argv);
            if(rc != 0)
                return rc;
        }

        if(params.D == 0u)
        {
            std::cerr << "Error: D (columns) must be greater than zero.\n";
            return EXIT_FAILURE;
        }
        if(params.N < 2u)
        {
            std::cerr << "Error: N (rows) must be at least 2 for sample stddev.\n";
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
} // namespace stddevBench
