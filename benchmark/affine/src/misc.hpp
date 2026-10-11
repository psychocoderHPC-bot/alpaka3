/* Copyright 2026 René Widera
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

/** @file Command line parsing helpers for the HeCBench `affine` benchmark.
 * @attention The license differs from the benchmark implementation because this file contains no parts of the
 * HeCBench code base.
 */

#include <catch2/catch_session.hpp>

#include <cstdint>
#include <cstdlib>
#include <iostream>

namespace affineBench
{
    //! Default image width and height, matching the pinned HeCBench `X_SIZE`/`Y_SIZE` (512x512).
    constexpr std::uint32_t defaultWidth = 512u;
    constexpr std::uint32_t defaultHeight = 512u;
    //! Transform passes per measured sample.
    constexpr int defaultRepeat = 100;
    //! Number of independent measured samples.
    constexpr int defaultSamples = 5;

    struct Params
    {
        //! Image width in pixels.
        std::uint32_t width = defaultWidth;
        //! Image height in pixels.
        std::uint32_t height = defaultHeight;
        //! Transform passes per measured sample, comparable to the reference positional `<iterations>`.
        int repeat = defaultRepeat;
        //! Independent measured samples.
        int samples = defaultSamples;
        //! Restrict the run to the correctness check (no timing).
        bool verifyOnly = false;
    };

    /** Parse the command line.
     *
     * The reference takes positional `<input image> <output image> <iterations>`. This port generates
     * its input internally and therefore uses named options only. The transform itself (rotation,
     * scale and translation) is fixed by the pinned reference and not configurable, so it is not
     * exposed here.
     */
    inline int parseCmd(int argc, char* argv[], Params& params)
    {
        Catch::Session session;
        using namespace Catch::Clara;

        auto cli = session.cli() | Opt(params.width, "width")["--width"]("Image width in pixels")
                   | Opt(params.height, "height")["--height"]("Image height in pixels")
                   | Opt(params.repeat, "repeat")["--repeat"]("Transform passes per measured sample")
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

} // namespace affineBench
