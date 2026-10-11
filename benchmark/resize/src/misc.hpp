/* Copyright 2026 René Widera
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/** @file Command line parsing helpers for the resize benchmark.
 * @attention The license differs from the benchmark implementation because this file contains no
 * parts of the HeCBench code base.
 */

#include <catch2/catch_session.hpp>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace resizeBench
{
    //! Default input/output sizes, matching the HeCBench reference `run` target `1920 1080 256 256`.
    constexpr std::uint32_t defaultInWidth = 1920u;
    constexpr std::uint32_t defaultInHeight = 1080u;
    constexpr std::uint32_t defaultOutWidth = 256u;
    constexpr std::uint32_t defaultOutHeight = 256u;
    //! Default number of images, matching the reference `run` target (`512` channels).
    constexpr std::uint32_t defaultChannels = 512u;
    //! Kernel passes per measured sample, comparable to the reference `<repeat>` argument.
    constexpr int defaultRepeat = 100;
    //! Number of independent measured samples.
    constexpr int defaultSamples = 5;

    struct Params
    {
        std::uint32_t inWidth = defaultInWidth;
        std::uint32_t inHeight = defaultInHeight;
        std::uint32_t outWidth = defaultOutWidth;
        std::uint32_t outHeight = defaultOutHeight;
        std::uint32_t channels = defaultChannels;
        int repeat = defaultRepeat;
        int samples = defaultSamples;
        //! Pixel type sizes to run: 1 (u8), 2 (u16), 4 (u32). Default: all three.
        std::vector<std::uint32_t> typeSizes{1u, 2u, 4u};
        //! Run the nearest kernel (default true).
        bool nearest = true;
        //! Run the bilinear kernel (default true).
        bool bilinear = true;
        //! Restrict the run to validation only (no timing loop).
        bool verifyOnly = false;

        [[nodiscard]] std::vector<bool> modes() const
        {
            std::vector<bool> m;
            if(nearest)
                m.push_back(false);
            if(bilinear)
                m.push_back(true);
            return m;
        }

        [[nodiscard]] std::vector<std::uint32_t> types() const
        {
            return typeSizes;
        }

        [[nodiscard]] std::string typeName() const
        {
            std::string s;
            for(std::size_t i = 0; i < typeSizes.size(); ++i)
                s += (i ? "," : "") + std::to_string(typeSizes[i] * 8u);
            return s;
        }

        [[nodiscard]] std::string modeName() const
        {
            std::string s;
            if(nearest)
                s += "nearest";
            if(bilinear)
                s += (s.empty() ? "" : ",") + std::string("bilinear");
            return s;
        }
    };

    /** Parse the command line.
     *
     * The reference takes positional `<in_width> <in_height> <out_width> <out_height> <channels>
     * <repeat>`. This port additionally allows selecting the pixel types and the resize modes.
     */
    inline int parseCmd(int argc, char* argv[], Params& params)
    {
        Catch::Session session;
        using namespace Catch::Clara;

        auto cli
            = session.cli()
              | Opt(params.inWidth, "inWidth")["--in-width"]("Input image width in pixels")
              | Opt(params.inHeight, "inHeight")["--in-height"]("Input image height in pixels")
              | Opt(params.outWidth, "outWidth")["--out-width"]("Output image width in pixels")
              | Opt(params.outHeight, "outHeight")["--out-height"]("Output image height in pixels")
              | Opt(params.channels, "channels")["--channels"]("Number of single-channel images")
              | Opt(params.repeat, "repeat")["--repeat"]("Kernel passes per measured sample")
              | Opt(params.samples, "samples")["--samples"]("Independent measured samples")
              | Opt(params.typeSizes, "types")["--types"](
                  "Pixel type sizes in bytes to run (1,2,4); repeatable")
              | Opt(params.nearest)["--nearest"]("Run the nearest-neighbour kernel")
              | Opt(params.bilinear)["--bilinear"]("Run the bilinear kernel")
              | Opt(params.verifyOnly)["--verify-only"]("Only run the correctness check, no timing");

        session.cli(cli);

        int const rc = session.applyCommandLine(argc, argv);
        if(rc != 0)
            return rc;

        if(params.inWidth == 0u || params.inHeight == 0u || params.outWidth == 0u || params.outHeight == 0u)
        {
            std::cerr << "Error: image dimensions must be greater than zero.\n";
            return EXIT_FAILURE;
        }
        if(params.channels == 0u)
        {
            std::cerr << "Error: number of channels must be greater than zero.\n";
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
        if(params.typeSizes.empty())
        {
            std::cerr << "Error: at least one pixel type must be selected.\n";
            return EXIT_FAILURE;
        }
        for(std::uint32_t t : params.typeSizes)
        {
            if(t != 1u && t != 2u && t != 4u)
            {
                std::cerr << "Error: pixel type size must be 1, 2 or 4 bytes.\n";
                return EXIT_FAILURE;
            }
        }
        if(!params.nearest && !params.bilinear)
        {
            std::cerr << "Error: at least one of --nearest or --bilinear must be selected.\n";
            return EXIT_FAILURE;
        }
        return EXIT_SUCCESS;
    }

} // namespace resizeBench
