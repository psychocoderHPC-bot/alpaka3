/* Copyright 2026 René Widera
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

/** @file Command line parsing helpers for the median filter benchmark.
 * @attention The license differs from the benchmark implementation because this file contains no parts of the
 * HeCBench code base.
 */

#include <catch2/catch_session.hpp>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>

namespace medianfilterBench
{
    //! Default image size, matching the HeCBench reference input (`SierrasRGB.ppm`, 1920x1080).
    constexpr std::uint32_t defaultWidth = 1920u;
    constexpr std::uint32_t defaultHeight = 1080u;
    //! Median filter passes per measured sample, comparable to the reference `<repeat>` argument.
    constexpr int defaultRepeat = 100;
    //! Number of independent measured samples.
    constexpr int defaultSamples = 5;

    struct Params
    {
        std::uint32_t width = defaultWidth;
        std::uint32_t height = defaultHeight;
        int repeat = defaultRepeat;
        int samples = defaultSamples;
        //! Restrict the run to validation only (no timing loop).
        bool verifyOnly = false;
    };

    /** Parse the command line.
     *
     * The reference takes a positional `<PPM image> <repeat>`. This port generates its image internally and
     * therefore uses named options only. The pinned HeCBench reference is a fixed 3x3 filter; there is no
     * window-size option in the reference, so none is exposed here.
     */
    inline int parseCmd(int argc, char* argv[], Params& params)
    {
        Catch::Session session;
        using namespace Catch::Clara;

        auto cli = session.cli() | Opt(params.width, "width")["--width"]("Image width in pixels")
                   | Opt(params.height, "height")["--height"]("Image height in pixels")
                   | Opt(params.repeat, "repeat")["--repeat"]("Median filter passes per measured sample")
                   | Opt(params.samples, "samples")["--samples"]("Independent measured samples")
                   | Opt(params.verifyOnly)["--verify-only"]("Only run the correctness check, no timing");

        session.cli(cli);

        int const rc = session.applyCommandLine(argc, argv);
        if(rc != 0)
            return rc;

        if(params.width == 0u)
        {
            std::cerr << "Error: width must be greater than zero.\n";
            return EXIT_FAILURE;
        }
        if(params.height == 0u)
        {
            std::cerr << "Error: height must be greater than zero.\n";
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

} // namespace medianfilterBench
