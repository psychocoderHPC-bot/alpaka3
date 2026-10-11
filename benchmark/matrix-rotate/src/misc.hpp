/* Copyright 2026 Rene Widera
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

/** @file Command line parsing helpers for the matrix-rotate benchmark.
 * @attention The license differs from the benchmark implementation because this file contains no parts of the
 * HeCBench code base.
 */

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>

namespace matrixRotate
{
    //! Matrix side length used by the HeCBench reference run (`./main 5000 100`).
    constexpr int defaultSize = 5000;
    //! Number of rotations used by the HeCBench reference run.
    constexpr int defaultRepeat = 100;

    inline bool parsePositiveInt(char const* text, int& out)
    {
        char* end = nullptr;
        long const value = std::strtol(text, &end, 10);
        if(end == text || *end != '\0' || value <= 0 || value > 1'000'000'000L)
            return false;
        out = static_cast<int>(value);
        return true;
    }

    /** Parse the command line and validate the parameters.
     *
     * Accepts the HeCBench positional form `./main <matrix size> <repeat>` as well as the named form
     * `--size <n> --repeat <r>`. The positional form is what the HeCBench reference uses.
     */
    inline int parseCmd(int argc, char* argv[], int& size, int& repeat)
    {
        size = defaultSize;
        repeat = defaultRepeat;

        int positional = 0;
        for(int i = 1; i < argc; ++i)
        {
            char const* arg = argv[i];
            if(std::strcmp(arg, "--size") == 0 || std::strcmp(arg, "-n") == 0)
            {
                if(i + 1 >= argc || !parsePositiveInt(argv[++i], size))
                {
                    std::cerr << "Error: --size requires a positive integer.\n";
                    return EXIT_FAILURE;
                }
            }
            else if(std::strcmp(arg, "--repeat") == 0 || std::strcmp(arg, "-r") == 0)
            {
                if(i + 1 >= argc || !parsePositiveInt(argv[++i], repeat))
                {
                    std::cerr << "Error: --repeat requires a positive integer.\n";
                    return EXIT_FAILURE;
                }
            }
            else if(std::strcmp(arg, "--help") == 0 || std::strcmp(arg, "-h") == 0)
            {
                std::cout << "Usage: " << argv[0] << " [<matrix size> <repeat>] [--size N] [--repeat N]\n";
                return EXIT_SUCCESS;
            }
            else if(arg[0] == '-' && arg[1] != '\0')
            {
                std::cerr << "Error: unknown option '" << arg << "'.\n";
                return EXIT_FAILURE;
            }
            else
            {
                int value = 0;
                if(!parsePositiveInt(arg, value))
                {
                    std::cerr << "Error: invalid positional argument '" << arg << "'.\n";
                    return EXIT_FAILURE;
                }
                if(positional == 0)
                    size = value;
                else if(positional == 1)
                    repeat = value;
                else
                {
                    std::cerr << "Error: too many positional arguments.\n";
                    return EXIT_FAILURE;
                }
                ++positional;
            }
        }
        return EXIT_SUCCESS;
    }
} // namespace matrixRotate
