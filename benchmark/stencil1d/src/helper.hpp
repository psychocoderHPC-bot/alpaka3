/* Copyright 2026 René Widera
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

/** @file Host reference, input generation and verification for the stencil1d benchmark.
 *
 * The benchmark is a radius-@c RADIUS sliding sum over a padded input array of
 * @c length + @c RADIUS elements. The padded tail is part of the input, so the window of an output
 * element near the end of the domain reads it; positions at or beyond the padded buffer are zero.
 *
 * Two properties make verification trustworthy:
 *
 *  - The comparison reads the **input array** element by element. It does not use the generation rule
 *    and does not reuse the kernel's shared-memory window or its indexing, so a wrong halo staging
 *    scheme cannot pass by returning a value derived from the same wrong window.
 *  - The default input is a deterministic non-trivial pseudo-random pattern
 *    (@ref makeInput), not the monotone ramp `a[i] = i`. A ramp would hide aliasing: for a symmetric
 *    sum, a duplicated or shifted halo can reproduce the same sum of consecutive integers. A random
 *    pattern has no such shift invariance.
 *
 * @ref oracleSanityCheck feeds a deliberately broken halo scheme through the same comparison and
 * fails if the comparison accepts it. That guards the guard: the validation is shown to have the
 * power to reject a real index error on the same input.
 *
 * The arithmetic is integer addition, so the reference is exact and the comparison is bit-exact
 * (tolerance 0). The tolerance is not widened: a mismatch is a real bug.
 */

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdio.h>
#include <vector>

namespace stencil1d
{
    //! Stencil radius, identical to the pinned HeCBench source (`#define RADIUS 7`).
    constexpr int radius = 7;
    //! Threads per block / tile size, identical to the pinned HeCBench source (`#define BLOCK_SIZE 256`).
    constexpr uint32_t tileSize = 256u;

    /** Fill a padded input buffer with a deterministic non-trivial pattern.
     *
     * A xorshift sequence truncated to a small symmetric range. The range is chosen so that the sum
     * of `2*radius + 1` terms cannot overflow `int32_t` (`15 * 4096` is far below `2^31`), which keeps
     * the device integer arithmetic and the double host reference bit-comparable.
     *
     * A ramp `a[i] = i` is deliberately avoided: for a symmetric sliding sum a shifted or duplicated
     * halo reproduces the same sum of consecutive integers, so a ramp can mask an aliasing error. The
     * random pattern is not shift invariant.
     */
    template<typename T_In>
    inline void makeInput(int padded, T_In& in)
    {
        uint32_t s = 0x9e3779b9u;
        for(int i = 0; i < padded; ++i)
        {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            in[i] = static_cast<int32_t>(s & 0x1fffu) - 0x1000; // [-4096, 4095]
        }
    }

    /** Independent double-precision host reference for output element @p i.
     *
     * The window `[i - radius, i + radius]` is summed by reading @p in directly. Indices outside
     * `[0, length + radius)` are treated as zero, matching the reference boundary handling at both
     * ends of the padded input. The result is computed from the input data, so any wrong halo load
     * or wrong boundary handling in the kernel is detected.
     *
     * @param i      output index
     * @param length number of output elements
     * @param in     padded input, `length + radius` elements
     */
    template<typename T_In>
    inline double referenceValue(int64_t i, int64_t length, T_In const& in)
    {
        int64_t const padded = length + radius;
        double sum = 0.0;
        for(int offset = -radius; offset <= radius; ++offset)
        {
            int64_t const j = i + offset;
            if(j < 0 || j >= padded)
                continue;
            sum += static_cast<double>(in[j]);
        }
        return sum;
    }

    /** Exact comparison of @p out against the independent reference.
     *
     * @return @c EXIT_SUCCESS on exact equality for every element, @c EXIT_FAILURE otherwise. If
     *         @p firstBadOut is not null it receives the index of the first mismatching element.
     */
    template<typename T_Out, typename T_In>
    inline int compareExact(int length, T_Out const& out, T_In const& in, int* firstBadOut = nullptr)
    {
        for(int i = 0; i < length; ++i)
        {
            double const got = static_cast<double>(out[i]);
            double const expected = referenceValue(i, length, in);
            if(!std::isfinite(got) || !std::isfinite(expected) || got != expected)
            {
                if(firstBadOut != nullptr)
                    *firstBadOut = i;
                return EXIT_FAILURE;
            }
        }
        return EXIT_SUCCESS;
    }

    /** Verify the device output against the independent host reference.
     *
     * The comparison is exact. NaN/Inf are rejected explicitly even though integer arithmetic cannot
     * produce them; the finiteness test guards against a corrupted or reinterpreted result buffer.
     */
    template<typename T_Out, typename T_In>
    inline int verify(int length, T_Out const& out, T_In const& in)
    {
        int bad = -1;
        int const error = compareExact(length, out, in, &bad);
        if(error != EXIT_SUCCESS)
            printf("error at out[%d]: got %f expected %f\n", bad, static_cast<double>(out[bad]),
                   referenceValue(bad, length, in));
        printf("Stencil1d verification %s\n", error ? "FAILED" : "PASSED");
        return error;
    }

    /** Recompute the output from a shared-memory window staged with the classic halo scheme.
     *
     * This mirrors the reference kernel's window layout (`temp[0..radius)` left halo,
     * `temp[radius..radius+tile)` centre, `temp[radius+tile..tile+2*radius)` right halo) and is used
     * only by @ref oracleSanityCheck to build both a correct and a deliberately broken output.
     * @p corruptRightHalo selects the broken scheme: the right-halo threads write into the centre
     * region instead of the right halo, which is exactly the aliasing error the benchmark must catch.
     */
    template<typename T_In>
    inline std::vector<int32_t> stageWindow(int length, T_In const& in, bool corruptRightHalo)
    {
        int const padded = length + radius;
        std::vector<int32_t> out(static_cast<size_t>(length), int32_t{0});
        int const blocks = (length + static_cast<int>(tileSize) - 1) / static_cast<int>(tileSize);
        for(int block = 0; block < blocks; ++block)
        {
            int const start = block * static_cast<int>(tileSize);
            std::vector<int32_t> temp(tileSize + 2u * radius, int32_t{0});
            for(int lid = 0; lid < static_cast<int>(tileSize); ++lid)
            {
                int const gindex = start + lid;
                temp[lid + radius] = (gindex < padded) ? in[gindex] : int32_t{0};
                if(lid < radius)
                {
                    int const left = gindex - radius;
                    temp[lid] = (left >= 0 && left < padded) ? in[left] : int32_t{0};

                    int const right = gindex + static_cast<int>(tileSize);
                    int32_t const rightVal = (right < padded) ? in[right] : int32_t{0};
                    if(corruptRightHalo)
                        temp[lid + radius] = rightVal; // BUG: overwrites centre region
                    else
                        temp[lid + radius + static_cast<int>(tileSize)] = rightVal;
                }
            }
            for(int lid = 0; lid < static_cast<int>(tileSize); ++lid)
            {
                int const gindex = start + lid;
                if(gindex < length)
                {
                    int32_t sum = 0;
                    for(int offset = -radius; offset <= radius; ++offset)
                        sum += temp[lid + radius + offset];
                    out[static_cast<size_t>(gindex)] = sum;
                }
            }
        }
        return out;
    }

    /** Prove that @ref compareExact rejects a deliberately broken halo scheme on this input.
     *
     * Builds the correct output and the broken output from the same input, and requires the
     * comparison to accept the first and reject the second. Without this check a bug in the
     * comparison itself could make every verification pass vacuously.
     *
     * @return @c EXIT_SUCCESS if the comparison separates the two outputs, @c EXIT_FAILURE otherwise.
     */
    template<typename T_In>
    inline int oracleSanityCheck(int length, T_In const& in)
    {
        std::vector<int32_t> const correct = stageWindow(length, in, /*corruptRightHalo=*/false);
        std::vector<int32_t> const broken = stageWindow(length, in, /*corruptRightHalo=*/true);

        int const rcCorrect = compareExact(length, correct, in);
        int const rcBroken = compareExact(length, broken, in);

        bool const ok = rcCorrect == EXIT_SUCCESS && rcBroken != EXIT_SUCCESS;
        printf(
            "Stencil1d oracle sanity: correct output accepted=%d, broken scheme rejected=%d -> %s\n",
            rcCorrect == EXIT_SUCCESS,
            rcBroken != EXIT_SUCCESS,
            ok ? "OK" : "FAILED");
        return ok ? EXIT_SUCCESS : EXIT_FAILURE;
    }

} // namespace stencil1d
