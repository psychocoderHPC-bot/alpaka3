/* Copyright 2026 René Widera
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

/** @file Command line parsing helpers for the memcpy benchmark.
 * @attention The license differs from the benchmark implementation because this file contains no parts of the
 * HeCBench code base.
 */

#include "helper.hpp"

#include <catch2/catch_session.hpp>

#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace memcpyBench
{
    //! Memory transfer mode, matching the three directions of the pinned HeCBench reference.
    enum class CopyMode
    {
        hostToDevice,
        deviceToHost,
        deviceToDevice,
        all
    };

    struct Params
    {
        //! Sizes in bytes; empty means the 16 HeCBench default sizes.
        std::vector<std::size_t> sizes;
        int repeat = defaultRepeat;
        int samples = defaultSamples;
        std::size_t maxBytes = defaultMaxBytes;
        CopyMode mode = CopyMode::all;
        //! Restrict the run to validation only (no timing loop).
        bool verifyOnly = false;
    };

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

        inline bool parseSizeList(std::string const& text, std::vector<std::size_t>& out)
        {
            std::size_t start = 0;
            while(start <= text.size())
            {
                std::size_t const comma = text.find(',', start);
                std::string const token
                    = text.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
                if(!allDigits(token.c_str()))
                    return false;
                out.push_back(static_cast<std::size_t>(std::strtoull(token.c_str(), nullptr, 10)));
                if(comma == std::string::npos)
                    break;
                start = comma + 1;
            }
            return !out.empty();
        }
    } // namespace detail

    /** Parse the command line.
     *
     * Accepts the HeCBench positional form `memcpy_alpaka <repeat>` and named options. Named options
     * take precedence.
     */
    inline int parseCmd(int argc, char* argv[], Params& params)
    {
        // HeCBench positional form: `./main <repeat>`.
        if(argc == 2 && detail::allDigits(argv[1]))
        {
            params.repeat = std::atoi(argv[1]);
            if(params.repeat <= 0)
            {
                std::cerr << "Error: repeat must be greater than zero.\n";
                return EXIT_FAILURE;
            }
            return EXIT_SUCCESS;
        }

        std::string sizesCsv;
        std::string mode;

        Catch::Session session;
        using namespace Catch::Clara;

        auto cli = session.cli() | Opt(sizesCsv, "sizes")["--sizes"]("Comma separated byte sizes")
                   | Opt(params.repeat, "repeat")["--repeat"]("Inner copies per sample (HeCBench repeat)")
                   | Opt(params.samples, "samples")["--samples"]("Independent measured samples")
                   | Opt(params.maxBytes, "maxBytes")["--max-bytes"]("Hard cap for a single allocation in bytes")
                   | Opt(mode, "mode")["--mode"]("Transfer mode: h2d, d2h, d2d or all")
                   | Opt(params.verifyOnly)["--verify-only"]("Only run the correctness check, no timing");

        session.cli(cli);

        int const rc = session.applyCommandLine(argc, argv);
        if(rc != 0)
            return rc;

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
        if(params.maxBytes == 0)
        {
            std::cerr << "Error: max-bytes must be greater than zero.\n";
            return EXIT_FAILURE;
        }

        if(!sizesCsv.empty())
        {
            params.sizes.clear();
            if(!detail::parseSizeList(sizesCsv, params.sizes))
            {
                std::cerr << "Error: --sizes must be a comma separated list of positive integers.\n";
                return EXIT_FAILURE;
            }
        }

        if(!mode.empty())
        {
            if(mode == "h2d")
                params.mode = CopyMode::hostToDevice;
            else if(mode == "d2h")
                params.mode = CopyMode::deviceToHost;
            else if(mode == "d2d")
                params.mode = CopyMode::deviceToDevice;
            else if(mode == "all")
                params.mode = CopyMode::all;
            else
            {
                std::cerr << "Error: --mode must be one of h2d, d2h, d2d, all.\n";
                return EXIT_FAILURE;
            }
        }

        return EXIT_SUCCESS;
    }

    inline char const* modeName(CopyMode mode)
    {
        switch(mode)
        {
        case CopyMode::hostToDevice:
            return "H2D";
        case CopyMode::deviceToHost:
            return "D2H";
        case CopyMode::deviceToDevice:
            return "D2D";
        case CopyMode::all:
            break;
        }
        return "all";
    }
} // namespace memcpyBench
