/* Copyright 2026 René Widera
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

/** @file Command line parsing helpers for the stencil1d benchmark.
 * @attention The license differs from the benchmark implementation because this file contains no parts of the
 * HeCBench code base.
 */

#include <catch2/catch_session.hpp>

#include <cctype>
#include <cstdlib>
#include <iostream>
#include <string>

namespace stencil1d
{
    //! HeCBench default run is `./main 134217728 1000`; the small fast default fits the test suite.
    constexpr int defaultLength = 1 << 20;
    constexpr int defaultRepeat = 5;

    namespace detail
    {
        inline bool allDigits(char const* s)
        {
            if(s == nullptr || *s == '\0')
                return false;
            for(char const* p = s; *p != '\0'; ++p)
                if(!std::isdigit(static_cast<unsigned char>(*p)))
                    return false;
            return true;
        }
    } // namespace detail

    /** Parse the command line.
     *
     * Accepts both the HeCBench positional form (`<length> <repeat>`) and the named form
     * (`--length N --repeat M`). Named options take precedence.
     */
    inline int parseCmd(int argc, char* argv[], int& length, int& repeat)
    {
        length = defaultLength;
        repeat = defaultRepeat;

        // Consume the HeCBench positional form `./main <length> <repeat>` before Catch sees it.
        if(argc == 3 && detail::allDigits(argv[1]) && detail::allDigits(argv[2]))
        {
            length = std::atoi(argv[1]);
            repeat = std::atoi(argv[2]);
            if(length <= 0 || repeat <= 0)
            {
                std::cerr << "Error: length and repeat must be greater than zero.\n";
                return EXIT_FAILURE;
            }
            return EXIT_SUCCESS;
        }

        Catch::Session session;
        using namespace Catch::Clara;

        auto cli = session.cli() | Opt(length, "length")["--length"]("Number of output elements")
                   | Opt(repeat, "repeat")["--repeat"]("Number of measured repetitions");

        session.cli(cli);

        int const rc = session.applyCommandLine(argc, argv);
        if(rc != 0)
            return rc;

        if(length <= 0)
        {
            std::cerr << "Error: length must be greater than zero.\n";
            return EXIT_FAILURE;
        }
        if(repeat <= 0)
        {
            std::cerr << "Error: repeat must be greater than zero.\n";
            return EXIT_FAILURE;
        }
        return EXIT_SUCCESS;
    }
} // namespace stencil1d
