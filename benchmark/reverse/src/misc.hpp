/* Copyright 2026 René Widera
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

/** @file Command line parsing helpers for the reverse benchmark.
 * @attention The license differs from the benchmark implementation because this file contains no parts of the
 * HeCBench code base.
 */

#include <catch2/catch_session.hpp>

#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>

namespace reverse
{
    //! Number of elements reversed by the HeCBench reference (`const int len = 256`).
    constexpr int defaultLength = 256;
    //! Largest supported length; the shared staging buffer is sized for this bound.
    constexpr int maxLength = 4096;
    //! Number of iteration/test cases. The reference binary is run as `./main 100`.
    constexpr int defaultIterations = 100;
    //! Number of measured repetitions of the per-case reverse loop (one value per invocation).
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
     * Accepts the HeCBench positional form (`./main <iterations>`) and the named form.
     * Named options take precedence when present.
     */
    inline int parseCmd(int argc, char* argv[], int& iterations, int& length, int& repeat)
    {
        iterations = defaultIterations;
        length = defaultLength;
        repeat = defaultRepeat;

        // Consume the HeCBench positional form `./main <iterations>` before Catch sees it.
        if(argc == 2 && detail::allDigits(argv[1]))
        {
            iterations = std::atoi(argv[1]);
            if(iterations <= 0)
            {
                std::cerr << "Error: iterations must be greater than zero.\n";
                return EXIT_FAILURE;
            }
            return EXIT_SUCCESS;
        }

        Catch::Session session;
        using namespace Catch::Clara;

        auto cli = session.cli() | Opt(iterations, "iterations")["--iterations"]("Number of test cases")
                   | Opt(length, "length")["--length"]("Number of elements to reverse")
                   | Opt(repeat, "repeat")["--repeat"]("Number of measured repetitions");

        session.cli(cli);

        int const rc = session.applyCommandLine(argc, argv);
        if(rc != 0)
            return rc;

        if(iterations <= 0)
        {
            std::cerr << "Error: iterations must be greater than zero.\n";
            return EXIT_FAILURE;
        }
        if(length <= 0)
        {
            std::cerr << "Error: length must be greater than zero.\n";
            return EXIT_FAILURE;
        }
        if(length > maxLength)
        {
            std::cerr << "Error: length must not exceed " << maxLength << ".\n";
            return EXIT_FAILURE;
        }
        if(repeat <= 0)
        {
            std::cerr << "Error: repeat must be greater than zero.\n";
            return EXIT_FAILURE;
        }
        return EXIT_SUCCESS;
    }
} // namespace reverse
