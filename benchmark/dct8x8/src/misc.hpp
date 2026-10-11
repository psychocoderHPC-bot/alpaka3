/* Copyright 2026 René Widera
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

/** @file Command line parsing helpers for the dct8x8 benchmark.
 * @attention The license differs from the benchmark implementation because this file contains no parts of the
 * HeCBench code base.
 */

#include <catch2/catch_session.hpp>

#include <cstdint>
#include <cstdlib>
#include <iostream>

namespace dct8x8Bench
{
    //! Default image size of the measured workload: the HeCBench `run` target uses 8192x8192.
    constexpr int defaultWidth = 8192;
    constexpr int defaultHeight = 8192;
    //! Reference `run` target transforms the image 100 times per direction.
    constexpr int defaultRepeat = 100;
    //! Number of independent measured samples.
    constexpr int defaultSamples = 5;
    //! Seed of the pinned reference `main` (`srand(2009)`).
    constexpr unsigned int defaultSeed = 2009u;
    //! The pinned reference transform definition is a relative-L2 comparison against a CPU
    //! reference with threshold `1e-6`; the same threshold is kept here.
    constexpr double defaultTolerance = 1e-6;

    struct Params
    {
        int width = defaultWidth;
        int height = defaultHeight;
        int repeat = defaultRepeat;
        int samples = defaultSamples;
        unsigned int seed = defaultSeed;
        //! Restrict the run to validation only (no timing loop).
        bool verifyOnly = false;
        //! Forward DCT (666 in the reference) unless set to inverse (777).
        bool inverse = false;
    };

    /** Parse the command line.
     *
     * The reference takes positional arguments `<image width> <image height> <repeat>`. This port
     * uses named options and generates the image internally; `--inverse` selects the inverse
     * transform, matching the second half of the reference `main`.
     */
    inline int parseCmd(int argc, char* argv[], Params& params)
    {
        Catch::Session session;
        using namespace Catch::Clara;

        auto cli = session.cli() | Opt(params.width, "width")["--width"]("Image width in pixels")
                   | Opt(params.height, "height")["--height"]("Image height in pixels")
                   | Opt(params.repeat, "repeat")["--repeat"]("Transforms per measured sample")
                   | Opt(params.samples, "samples")["--samples"]("Independent measured samples")
                   | Opt(params.seed, "seed")["--seed"]("Input seed (reference uses 2009)")
                   | Opt(params.inverse)["--inverse"]("Run the inverse DCT (default: forward DCT)")
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

} // namespace dct8x8Bench
