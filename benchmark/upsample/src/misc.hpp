/* Copyright 2026 René Widera
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

/** @file Command line parsing helpers for the upsample benchmark.
 * @attention The license differs from the benchmark implementation because this file contains no parts of the
 * HeCBench code base.
 */

#include <catch2/catch_session.hpp>

#include <cstdint>
#include <cstdlib>
#include <iostream>

namespace upsampleBench
{
    //! Default problem size: the HeCBench `run` target uses `32 64 64 64 100`.
    constexpr int defaultBatch = 32;
    constexpr int defaultChannels = 64;
    constexpr int defaultHeight = 64;
    constexpr int defaultWidth = 64;
    //! Reference `run` target repeats the pass 100 times per measurement.
    constexpr int defaultRepeat = 100;
    //! Number of independent measured samples.
    constexpr int defaultSamples = 5;
    //! Seed of the internal deterministic input generator (the reference calls `srand(0)`).
    constexpr std::uint64_t defaultSeed = 0u;
    /** Per-tile relative-L2 acceptance threshold.
     *
     * The forward pass is an exact 2x2 replication and the backward pass sums one 2x2 tile in a fixed
     * order, so a correct portable port reproduces the reference arithmetic exactly. The threshold
     * leaves room for a SIMD lane-reordering backend (one rounding unit, ~6e-8 for fp32) while still
     * rejecting any wrong tile mapping, border rule or offset. The per-tile metric is defined in
     * `helper.hpp`.
     */
    constexpr double defaultTolerance = 1e-8;

    struct Params
    {
        int batch = defaultBatch;
        int channels = defaultChannels;
        int height = defaultHeight;
        int width = defaultWidth;
        int repeat = defaultRepeat;
        int samples = defaultSamples;
        std::uint64_t seed = defaultSeed;
        //! Restrict the run to validation only (no timing loop).
        bool verifyOnly = false;
    };

    /** Parse the command line.
     *
     * The reference takes positional arguments `<batch size> <number of channels> <height> <width>
     * <repeat>`. This port uses named options and generates its input internally.
     */
    inline int parseCmd(int argc, char* argv[], Params& params)
    {
        Catch::Session session;
        using namespace Catch::Clara;

        auto cli = session.cli() | Opt(params.batch, "batch")["--batch"]("Batch size N")
                   | Opt(params.channels, "channels")["--channels"]("Number of channels C")
                   | Opt(params.height, "height")["--height"]("Input height H")
                   | Opt(params.width, "width")["--width"]("Input width W")
                   | Opt(params.repeat, "repeat")["--repeat"]("Passes per measured sample")
                   | Opt(params.samples, "samples")["--samples"]("Independent measured samples")
                   | Opt(params.seed, "seed")["--seed"]("Input seed (reference uses 0)")
                   | Opt(params.verifyOnly)["--verify-only"]("Only run the correctness check, no timing");

        session.cli(cli);

        int const rc = session.applyCommandLine(argc, argv);
        if(rc != 0)
            return rc;

        if(params.batch <= 0)
        {
            std::cerr << "Error: batch must be greater than zero.\n";
            return EXIT_FAILURE;
        }
        if(params.channels <= 0)
        {
            std::cerr << "Error: channels must be greater than zero.\n";
            return EXIT_FAILURE;
        }
        if(params.height <= 0)
        {
            std::cerr << "Error: height must be greater than zero.\n";
            return EXIT_FAILURE;
        }
        if(params.width <= 0)
        {
            std::cerr << "Error: width must be greater than zero.\n";
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

} // namespace upsampleBench
