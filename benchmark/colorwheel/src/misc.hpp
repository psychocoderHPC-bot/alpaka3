/* Copyright 2026 René Widera
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

/** @file Command line parsing helpers for the colorwheel benchmark.
 * @attention The license differs from the benchmark implementation because this file contains no parts of the
 * HeCBench code base.
 */

#include <catch2/catch_session.hpp>

#include <cctype>
#include <cstdlib>
#include <iostream>

namespace colorwheel
{
    //! HeCBench reference run: `./main 10 4096 100`.
    constexpr float defaultTrueRange = 10.0f;
    constexpr int defaultSize = 4096;
    constexpr int defaultRepeat = 100;
    //! Number of independent measured samples.
    constexpr int defaultSamples = 5;

    struct Params
    {
        float trueRange = defaultTrueRange;
        int size = defaultSize;
        int repeat = defaultRepeat;
        int samples = defaultSamples;
        //! Restrict the run to validation only (no timing loop).
        bool verifyOnly = false;
    };

    namespace detail
    {
        inline bool isNumber(char const* s)
        {
            if(s == nullptr || *s == '\0')
                return false;
            for(char const* p = s; *p != '\0'; ++p)
                if(!std::isdigit(static_cast<unsigned char>(*p)) && *p != '.')
                    return false;
            return true;
        }
    } // namespace detail

    /** Parse the command line.
     *
     * Accepts the HeCBench positional form `colorwheel_alpaka <range> <size> <repeat>` (this is what
     * the upstream `Makefile` `run` target uses) as well as named options
     * `--range R --size N --repeat M --samples S --verify-only`.
     */
    inline int parseCmd(int argc, char* argv[], Params& params)
    {
        // Positional HeCBench form `<range> <size> <repeat>`.
        if(argc == 4 && detail::isNumber(argv[1]) && detail::isNumber(argv[2]) && detail::isNumber(argv[3]))
        {
            params.trueRange = static_cast<float>(std::atof(argv[1]));
            params.size = std::atoi(argv[2]);
            params.repeat = std::atoi(argv[3]);
        }
        else
        {
            Catch::Session session;
            using namespace Catch::Clara;

            auto cli = session.cli() | Opt(params.trueRange, "range")["--range"]("Colour-wheel range")
                       | Opt(params.size, "size")["--size"]("Image side length in pixels")
                       | Opt(params.repeat, "repeat")["--repeat"]("Kernel launches per measured sample")
                       | Opt(params.samples, "samples")["--samples"]("Independent measured samples")
                       | Opt(params.verifyOnly)["--verify-only"]("Only run the correctness check, no timing");

            session.cli(cli);

            int const rc = session.applyCommandLine(argc, argv);
            if(rc != 0)
                return rc;
        }

        if(params.trueRange <= 0.0f)
        {
            std::cerr << "Error: range must be greater than zero.\n";
            return EXIT_FAILURE;
        }
        if(params.size < 3)
        {
            std::cerr << "Error: size must be at least 3 (a non-empty image with an interior).\n";
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

} // namespace colorwheel
