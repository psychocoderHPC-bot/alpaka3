/* Copyright 2026 Rene Widera
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

/** @file Command line parsing helpers for the axpby benchmark.
 * @attention The license differs from the benchmark implementation because this file contains no parts of the
 * HeCBench code base.
 */

#include <catch2/catch_session.hpp>

#include <cstdint>
#include <cstdlib>
#include <iostream>

namespace axpby
{
    //! Default tensor count used by the HeCBench reference run (`./main 1000`).
    constexpr int defaultNumTensors = 1000;
    //! Elements per chunk. Mirrors the reference `BLOCK_SIZE` (512) at its first chunk size.
    constexpr int defaultChunkSize = 512;
    constexpr int defaultRepeat = 5;

    /** Parse the command line and validate the parameters.
     *
     * @param fixedLength if greater than zero every tensor has exactly this length, otherwise the length is drawn
     *                    from the HeCBench `srand(123)` sequence.
     */
    inline int parseCmd(
        int argc,
        char* argv[],
        int& numTensors,
        int& chunkSize,
        int& repeat,
        int& fixedLength)
    {
        numTensors = defaultNumTensors;
        chunkSize = defaultChunkSize;
        repeat = defaultRepeat;
        fixedLength = 0;

        Catch::Session session;
        using namespace Catch::Clara;

        auto cli = session.cli()
                   | Opt(numTensors, "numTensors")["--tensors"]("Number of independent tensors")
                   | Opt(chunkSize, "chunkSize")["--chunk-size"]("Elements processed per chunk (one chunk per block)")
                   | Opt(repeat, "repeat")["--repeat"]("Number of measured repetitions")
                   | Opt(fixedLength, "fixedLength")["--fixed-length"](
                       "Force this length for every tensor (0: use the HeCBench srand(123) lengths)");

        session.cli(cli);

        int const rc = session.applyCommandLine(argc, argv);
        if(rc != 0)
            return rc;

        if(numTensors <= 0)
        {
            std::cerr << "Error: number of tensors must be greater than zero.\n";
            return EXIT_FAILURE;
        }
        if(chunkSize <= 0)
        {
            std::cerr << "Error: chunk size must be greater than zero.\n";
            return EXIT_FAILURE;
        }
        if(repeat <= 0)
        {
            std::cerr << "Error: number of repetitions must be greater than zero.\n";
            return EXIT_FAILURE;
        }
        if(fixedLength < 0)
        {
            std::cerr << "Error: fixed length must not be negative.\n";
            return EXIT_FAILURE;
        }
        if(fixedLength == 1)
        {
            std::cerr << "Error: fixed length must be either 0 or at least 2.\n";
            return EXIT_FAILURE;
        }
        return EXIT_SUCCESS;
    }
} // namespace axpby
