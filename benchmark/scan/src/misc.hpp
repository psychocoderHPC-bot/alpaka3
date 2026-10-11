/* Copyright 2026 René Widera
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

/** @file Command line parsing helpers for the `scan` benchmark.
 * @attention The license differs from the benchmark implementation because this file contains no parts of the
 * HeCBench code base.
 */

#include <catch2/catch_session.hpp>

#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>

namespace scanBench
{
    //! Default number of input elements; the HeCBench `run` target uses `268435456 100`.
    constexpr std::uint64_t defaultNumElements = 1ull << 26;
    //! Chunk size (elements per independent block scan); matches the reference default set.
    constexpr std::uint32_t defaultChunk = 2048u;
    //! Inner scans per measured sample.
    constexpr int defaultRepeat = 10;
    //! Number of independent measured samples.
    constexpr int defaultSamples = 5;

    struct Params
    {
        //! Number of input elements requested; the actual scanned count is rounded up per chunk.
        std::uint64_t numElements = defaultNumElements;
        //! Requested chunk size; 0 selects every reference chunk size (128, 256, 512, 1024, 2048).
        std::uint32_t chunk = defaultChunk;
        //! Inner scans per measured sample.
        int repeat = defaultRepeat;
        //! Independent measured samples.
        int samples = defaultSamples;
        //! Element type: "char", "short", "int", "long" or "all".
        std::string type = "int";
        //! Restrict the run to validation only (no timing loop).
        bool verifyOnly = false;
    };

    /** Parse the command line.
     *
     * Named options keep the same meaning as the pinned HeCBench reference run
     * (`./main <number of elements> <repeat>`): the scanned count and the number of inner scans.
     */
    inline int parseCmd(int argc, char* argv[], Params& params)
    {
        Catch::Session session;
        using namespace Catch::Clara;

        auto cli = session.cli()
                   | Opt(params.numElements, "numElements")["--n"]("Number of input elements to scan")
                   | Opt(params.chunk, "chunk")["--chunk"]("Elements per independent block scan (0 = all)")
                   | Opt(params.repeat, "repeat")["--repeat"]("Inner scans per measured sample")
                   | Opt(params.samples, "samples")["--samples"]("Independent measured samples")
                   | Opt(params.type, "type")["--type"]("Element type: char, short, int, long or all")
                   | Opt(params.verifyOnly)["--verify-only"]("Only run the correctness checks, no timing");

        session.cli(cli);

        int const rc = session.applyCommandLine(argc, argv);
        if(rc != 0)
            return rc;

        if(params.chunk != 0u && params.chunk != 128u && params.chunk != 256u && params.chunk != 512u
           && params.chunk != 1024u && params.chunk != 2048u)
        {
            std::cerr << "Error: chunk must be one of 0, 128, 256, 512, 1024, 2048.\n";
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
        if(params.type != "char" && params.type != "short" && params.type != "int" && params.type != "long"
           && params.type != "all")
        {
            std::cerr << "Error: type must be one of char, short, int, long, all.\n";
            return EXIT_FAILURE;
        }

        return EXIT_SUCCESS;
    }

} // namespace scanBench
