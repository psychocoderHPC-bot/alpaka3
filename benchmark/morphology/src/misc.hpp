/* Copyright 2026 René Widera
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

/** @file Command line parsing helpers for the morphology benchmark.
 * @attention The license differs from the benchmark implementation because this file contains no parts of the
 * HeCBench code base.
 */

#include "helper.hpp"

#include <catch2/catch_session.hpp>

#include <cstdint>
#include <cstdlib>
#include <iostream>

namespace morphologyBench
{
    //! Default image size of the measured workload: the HeCBench-derived irregular 999x777 frame.
    constexpr int defaultWidth = 999;
    constexpr int defaultHeight = 777;
    //! Default structuring-element size (`hsize == vsize`); the measured workload uses ~70.
    constexpr int defaultSelSize = 71;
    //! Inner `dilate`+`erode` pairs per measured sample; the reference `run` target uses 100.
    constexpr int defaultRepeat = 100;
    //! Number of independent measured samples.
    constexpr int defaultSamples = 5;
    //! Zero selects the reference's single-white-pixel image; nonzero selects a deterministic random image.
    constexpr unsigned int defaultSeed = 0u;

    //! Largest structuring element the portable kernel supports (bound on the internal deque).
    constexpr int maxSelSize = 256;

    struct Params
    {
        int width = defaultWidth;
        int height = defaultHeight;
        int hsize = defaultSelSize;
        int vsize = defaultSelSize;
        int repeat = defaultRepeat;
        int samples = defaultSamples;
        unsigned int seed = defaultSeed;
        //! Restrict the run to validation only (no timing loop).
        bool verifyOnly = false;
    };

    inline bool isPositive(int value)
    {
        return value > 0;
    }

    /** Parse the command line.
     *
     * The reference takes positional arguments `<kernel width> <kernel height> <image width> <image
     * height> <repeat>`. This port uses named options and generates the image internally.
     */
    inline int parseCmd(int argc, char* argv[], Params& params)
    {
        Catch::Session session;
        using namespace Catch::Clara;

        auto cli = session.cli() | Opt(params.hsize, "hsize")["--hsize"]("Structuring-element width in pixels")
                   | Opt(params.vsize, "vsize")["--vsize"]("Structuring-element height in pixels")
                   | Opt(params.width, "width")["--width"]("Image width in pixels")
                   | Opt(params.height, "height")["--height"]("Image height in pixels")
                   | Opt(params.repeat, "repeat")["--repeat"]("Dilate+erode pairs per measured sample")
                   | Opt(params.samples, "samples")["--samples"]("Independent measured samples")
                   | Opt(params.seed, "seed")["--seed"]("0 = single-pixel image, else deterministic random image")
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
        if(!isPositive(params.hsize) || !isPositive(params.vsize))
        {
            std::cerr << "Error: hsize and vsize must be greater than zero.\n";
            return EXIT_FAILURE;
        }
        if(params.hsize > maxSelSize || params.vsize > maxSelSize)
        {
            std::cerr << "Error: hsize and vsize must be <= " << maxSelSize << ".\n";
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

} // namespace morphologyBench
