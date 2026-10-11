/* Copyright 2026 René Widera
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

/** @file Host reference and verification for the memcpy benchmark.
 *
 * The HeCBench `memcpy` reference copies an opaque byte payload between host and device memory and
 * only measures bandwidth; it does not verify the transfer. This file supplies a deterministic byte
 * pattern (the host reference) and a byte-exact comparison.
 *
 * The payload is raw bytes, not floating point values, so a NaN/Inf check on the payload would be
 * meaningless (an arbitrary byte sequence interpreted as `float` is frequently NaN). A full
 * byte-exact comparison is strictly stronger than any NaN/Inf test and is what `verifyBytes()` does:
 * every byte must match, otherwise the process returns a nonzero exit code.
 */

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace memcpyBench
{
    //! Number of sizes in the pinned HeCBench reference (`#define NUM_SIZE 16`).
    constexpr std::size_t numSizes = 16u;
    //! Inner copies per measured sample, matching the HeCBench `repeat` argument (`./main 8192`).
    constexpr int defaultRepeat = 8192;
    //! Number of independent measured samples per (mode, size).
    constexpr int defaultSamples = 5;
    //! Default cap for a single allocation, so a user requested size cannot exhaust device memory.
    constexpr std::size_t defaultMaxBytes = 4ull << 30; // 4 GiB

    /** The 16 sizes of the pinned HeCBench reference: `size[i] = 1 << (i + 6)`, i.e. 64 B .. 2 MiB.
     *
     * The task brief mentioned larger sizes; the pinned source at
     * `0a17ee47ca6072192101036d4bf292e6612f22e6` caps at `1 << 21` = 2 MiB. Larger sizes can be
     * requested explicitly with `--size` / `--sizes` and are bounded by `--max-bytes`.
     */
    inline std::vector<std::size_t> defaultSizes()
    {
        std::vector<std::size_t> sizes;
        sizes.reserve(numSizes);
        for(std::size_t i = 0; i < numSizes; ++i)
            sizes.push_back(std::size_t{1} << (i + 6u));
        return sizes;
    }

    /** Deterministic position dependent payload byte.
     *
     * The value depends on the byte index, so a copy that is truncated, shifted, or copied at the
     * wrong alignment produces a mismatch at a well defined position. A constant fill (like the
     * HeCBench `valSet(A, 1, ...)`) would hide e.g. an off-by-one that copies a shifted region.
     */
    inline std::uint8_t patternByte(std::size_t index)
    {
        std::uint64_t const mixed
            = (static_cast<std::uint64_t>(index) + 0x9E3779B97F4A7C15ull) * 0xBF58476D1CE4E5B9ull;
        return static_cast<std::uint8_t>((mixed >> 56) & 0xFFull);
    }

    /** Fill @p bytes bytes at @p dst with the deterministic reference pattern. */
    inline void fillPattern(std::uint8_t* dst, std::size_t bytes)
    {
        for(std::size_t i = 0; i < bytes; ++i)
            dst[i] = patternByte(i);
    }

    /** Byte-exact comparison of @p got against the reference pattern.
     *
     * @return `EXIT_SUCCESS` if all @p bytes bytes match, `EXIT_FAILURE` otherwise. The first
     *         mismatching byte is reported. No tolerance is applied: any difference is a real bug.
     */
    inline int verifyBytes(
        std::uint8_t const* got,
        std::size_t bytes,
        char const* modeLabel,
        std::size_t size,
        char const* backend)
    {
        for(std::size_t i = 0; i < bytes; ++i)
        {
            std::uint8_t const expected = patternByte(i);
            if(got[i] != expected)
            {
                std::printf(
                    "error: %s copy, backend=%s size=%zu byte=%zu expected=%u got=%u\n",
                    modeLabel,
                    backend,
                    size,
                    i,
                    static_cast<unsigned>(expected),
                    static_cast<unsigned>(got[i]));
                std::printf("Verification FAILED\n");
                return EXIT_FAILURE;
            }
        }
        return EXIT_SUCCESS;
    }

} // namespace memcpyBench
