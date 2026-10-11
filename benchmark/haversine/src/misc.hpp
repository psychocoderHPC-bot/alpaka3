/* Copyright 2026 René Widera
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

/** @file Command line parsing helpers for the HeCBench `haversine` benchmark.
 * @attention The license differs from the benchmark implementation because this file contains no parts of the
 * HeCBench code base.
 */

#include <catch2/catch_session.hpp>

#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>

namespace haversineBench
{
    struct Params
    {
        //! Number of distinct city locations (`num_cities`, HeCBench default 2^21).
        std::uint32_t numCities = defaultNumCities;
        //! Timed repetitions of the distance kernel per measured sample.
        int repeat = 100;
        //! Independent measured samples.
        int samples = 5;
        //! Seed for the deterministic location generator (ignored when @ref input is set).
        std::uint64_t seed = 12345u;
        //! Logical workers per frame (launch-shape hint).
        std::uint32_t frame = 256u;
        //! Optional reference-format `lat lon` file; empty -> generate locations in-process.
        std::string input;
        //! Optional path to write the raw device distance array (validation aid).
        std::string dump;
        //! Run the correctness check only, no timing.
        bool verifyOnly = false;
    };

    namespace detail
    {
        inline bool startsWithDash(char const* s)
        {
            return s != nullptr && s[0] == '-' && s[1] != '\0';
        }
    } // namespace detail

    /** Parse the command line.
     *
     * Accepts the HeCBench positional form `haversine_alpaka <file> <repeat>` (a reference-format
     * location file plus a repeat count) and the named form below. When a positional file is given it
     * is loaded; otherwise the built-in deterministic generator is used.
     */
    inline int parseCmd(int argc, char* argv[], Params& params)
    {
        bool const positional = (argc == 3) && !detail::startsWithDash(argv[1]);

        if(positional)
        {
            params.input = argv[1];
            params.repeat = std::atoi(argv[2]);
        }
        else
        {
            Catch::Session session;
            using namespace Catch::Clara;

            auto cli = session.cli()
                       | Opt(params.numCities, "cities")["--cities"](
                           "Number of distinct city locations (num_cities)")
                       | Opt(params.repeat, "repeat")["--repeat"]("Kernel repetitions per sample")
                       | Opt(params.samples, "samples")["--samples"]("Independent measured samples")
                       | Opt(params.seed, "seed")["--seed"]("Seed of the deterministic location generator")
                       | Opt(params.frame, "frame")["--frame"]("Logical workers per frame (launch-shape hint)")
                       | Opt(params.input, "input")["--input"](
                           "Reference-format lat/lon file (default: built-in generator)")
                       | Opt(params.dump, "dump")["--dump"]("Write the raw device distance array to this file")
                       | Opt(params.verifyOnly)["--verify-only"]("Only run the correctness check, no timing");

            session.cli(cli);

            int const rc = session.applyCommandLine(argc, argv);
            if(rc != 0)
                return rc;
        }

        if(params.numCities == 0u)
        {
            std::cerr << "Error: --cities must be greater than zero.\n";
            return EXIT_FAILURE;
        }
        if(params.repeat <= 0)
        {
            std::cerr << "Error: --repeat must be greater than zero.\n";
            return EXIT_FAILURE;
        }
        if(params.samples <= 0)
        {
            std::cerr << "Error: --samples must be greater than zero.\n";
            return EXIT_FAILURE;
        }
        if(params.frame == 0u)
        {
            std::cerr << "Error: --frame must be greater than zero.\n";
            return EXIT_FAILURE;
        }
        return EXIT_SUCCESS;
    }
} // namespace haversineBench
