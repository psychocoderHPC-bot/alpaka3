/* Copyright 2026 René Widera
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

/** @file Command line parsing helpers for the histogram benchmark.
 * @attention The license differs from the benchmark implementation because this file contains no parts of the
 * HeCBench code base.
 */

#include "helper.hpp"

#include <catch2/catch_session.hpp>

#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>

namespace histogramBench
{
    //! Default number of pixels; the HeCBench reference defaults to width=1920, height=1080 = 2,073,600.
    constexpr std::size_t defaultNumPixels = 1920u * 1080u;
    //! Inner histogram computations per measured sample, matching the reference `--i` default of 100.
    constexpr int defaultRepeat = 100;
    //! Number of independent measured samples.
    constexpr int defaultSamples = 5;
    //! Entropy reduction of the reference `--entropy` default.
    constexpr int defaultEntropy = 0;

    struct Params
    {
        //! Number of pixels (width * height in the reference).
        std::size_t numPixels = defaultNumPixels;
        int repeat = defaultRepeat;
        int samples = defaultSamples;
        int entropy = defaultEntropy;
        //! Restrict the run to validation only (no timing loop).
        bool verifyOnly = false;
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
     * Accepts named options. The HeCBench positional form `./main <iterations>` has no stable analogue
     * because this benchmark requires an explicit pixel count, so only named options are accepted.
     */
    inline int parseCmd(int argc, char* argv[], Params& params)
    {
        Catch::Session session;
        using namespace Catch::Clara;

        auto cli = session.cli() | Opt(params.numPixels, "numPixels")["--pixels"]("Number of pixels to histogram")
                   | Opt(params.repeat, "repeat")["--repeat"]("Inner histogram computations per sample")
                   | Opt(params.samples, "samples")["--samples"]("Independent measured samples")
                   | Opt(params.entropy, "entropy")["--entropy"]("Entropy reduction of the generated image")
                   | Opt(params.verifyOnly)["--verify-only"]("Only run the correctness check, no timing");

        session.cli(cli);

        int const rc = session.applyCommandLine(argc, argv);
        if(rc != 0)
            return rc;

        if(params.numPixels == 0)
        {
            std::cerr << "Error: number of pixels must be greater than zero.\n";
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
        if(params.entropy < -1)
        {
            std::cerr << "Error: entropy must be >= -1.\n";
            return EXIT_FAILURE;
        }

        return EXIT_SUCCESS;
    }

} // namespace histogramBench
