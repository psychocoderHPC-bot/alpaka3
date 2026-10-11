/* Copyright 2026 René Widera
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

/** @file Command line parsing helpers for the bilateral filter benchmark.
 * @attention The license differs from the benchmark implementation because this file contains no parts of the
 * HeCBench code base.
 */

#include <catch2/catch_session.hpp>

#include <cstdint>
#include <cstdlib>
#include <iostream>

namespace bilateralBench
{
    //! Default image size, matching the pinned HeCBench reference invocation (`main 2960 1440 0.5 0.5 1000`).
    constexpr int defaultWidth = 2960;
    constexpr int defaultHeight = 1440;
    //! Default range-kernel variance (third reference argument).
    constexpr float defaultVarianceI = 0.5f;
    //! Default spatial-kernel variance (fourth reference argument).
    constexpr float defaultVarianceSpatial = 0.5f;
    //! Inner filter passes per measured sample (fifth reference argument).
    constexpr int defaultRepeat = 1000;
    //! Number of independent measured samples.
    constexpr int defaultSamples = 5;
    //! Run every reference radius when `radius < 0`.
    constexpr int allRadii = -1;

    struct Params
    {
        int width = defaultWidth;
        int height = defaultHeight;
        float varianceI = defaultVarianceI;
        float varianceSpatial = defaultVarianceSpatial;
        int repeat = defaultRepeat;
        int samples = defaultSamples;
        //! Window radius; `< 0` runs the reference set {3, 6, 9}.
        int radius = allRadii;
        //! Restrict the run to validation only (no timing loop).
        bool verifyOnly = false;
    };

    /** Parse the command line.
     *
     * The reference takes positional `<width> <height> <intensity> <spatial> <repeat>`. This port keeps the same
     * defaults but exposes named options and an optional single radius for quick checks. `--verify-only`
     * skips timing and runs only the correctness check.
     */
    inline int parseCmd(int argc, char* argv[], Params& params)
    {
        Catch::Session session;
        using namespace Catch::Clara;

        auto cli = session.cli() | Opt(params.width, "width")["--width"]("Image width in pixels (reference: 2960)")
                   | Opt(params.height, "height")["--height"]("Image height in pixels (reference: 1440)")
                   | Opt(params.varianceI, "intensity")["--intensity"]("Range-kernel variance (reference: 0.5)")
                   | Opt(params.varianceSpatial, "spatial")["--spatial"]("Spatial-kernel variance (reference: 0.5)")
                   | Opt(params.repeat, "repeat")["--repeat"]("Inner filter passes per measured sample")
                   | Opt(params.samples, "samples")["--samples"]("Independent measured samples")
                   | Opt(params.radius, "radius")["--radius"]("Window radius, or -1 for the reference set {3,6,9}")
                   | Opt(params.verifyOnly)["--verify-only"]("Only run the correctness check, no timing");

        session.cli(cli);

        int const rc = session.applyCommandLine(argc, argv);
        if(rc != 0)
            return rc;

        if(params.width <= 0)
        {
            std::cerr << "Error: width must be greater than zero.\n";
            return EXIT_FAILURE;
        }
        if(params.height <= 0)
        {
            std::cerr << "Error: height must be greater than zero.\n";
            return EXIT_FAILURE;
        }
        if(!(params.varianceI > 0.0f))
        {
            std::cerr << "Error: intensity variance must be positive.\n";
            return EXIT_FAILURE;
        }
        if(!(params.varianceSpatial > 0.0f))
        {
            std::cerr << "Error: spatial variance must be positive.\n";
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
        if(params.radius >= 0 && params.radius > params.width - 1)
        {
            std::cerr << "Error: radius must not exceed width-1.\n";
            return EXIT_FAILURE;
        }
        if(params.radius >= 0 && params.radius > params.height - 1)
        {
            std::cerr << "Error: radius must not exceed height-1.\n";
            return EXIT_FAILURE;
        }
        if(params.radius < 0 && (params.width < 9 + 1 || params.height < 9 + 1))
        {
            std::cerr << "Error: width and height must be at least 10 for the reference radii {3,6,9}.\n";
            return EXIT_FAILURE;
        }

        return EXIT_SUCCESS;
    }

} // namespace bilateralBench
