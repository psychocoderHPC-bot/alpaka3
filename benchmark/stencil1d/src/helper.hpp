/* Copyright 2026 René Widera
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

/** @file Host reference and verification for the stencil1d benchmark.
 *
 * The HeCBench reference builds an input `a[i] = i` for `i` in `[0, length + RADIUS)` and computes
 * the radius-RADIUS sliding sum. Values left of the domain are zero (`a[j] = 0` for `j < 0`);
 * values right of `length` use the padded sequence `a[j] = j` up to `length + RADIUS - 1`; positions
 * at or beyond the padded buffer are zero.
 *
 * To make the check an *independent* oracle, `referenceValue()` reads the **original padded input
 * array** element by element and sums the window directly. It does not use the generation rule
 * `a[i] = i`, and it does not reuse the shared-memory window or its indexing. A kernel whose halo
 * staging scheme is wrong therefore cannot pass by construction: it would have to return the value
 * computed from the correct input independently.
 *
 * The arithmetic is integer addition, so the reference is exact and the comparison is bit-exact
 * (tolerance 0). The tolerance is not widened: a mismatch is a real bug.
 */

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <stdio.h>

namespace stencil1d
{
    //! Stencil radius, identical to the pinned HeCBench source (`#define RADIUS 7`).
    constexpr int radius = 7;
    //! Threads per block / tile size, identical to the pinned HeCBench source (`#define BLOCK_SIZE 256`).
    constexpr uint32_t tileSize = 256u;

    /** Independent double-precision host reference for output element @p i.
     *
     * The window `[i - radius, i + radius]` is summed by reading @p in directly. Indices outside
     * `[0, length + radius)` are treated as zero, matching the reference boundary handling at both
     * ends of the padded input. The result is computed from the *input data*, so any wrong halo load
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

    /** Verify the device output against the independent host reference.
     *
     * @param length number of output elements
     * @param out    device result, `length` elements
     * @param in     padded input, `length + radius` elements; the reference is computed from it
     *
     * The comparison is exact. NaN/Inf are rejected explicitly even though integer arithmetic cannot
     * produce them; the finiteness test guards against a corrupted or reinterpreted result buffer.
     */
    template<typename T_Out, typename T_In>
    int verify(int length, T_Out const& out, T_In const& in)
    {
        int error = EXIT_SUCCESS;
        for(int i = 0; i < length; ++i)
        {
            double const got = static_cast<double>(out[i]);
            double const expected = referenceValue(i, length, in);
            if(!std::isfinite(got) || !std::isfinite(expected))
            {
                printf("error at out[%d]: non-finite value got %f expected %f\n", i, got, expected);
                error = EXIT_FAILURE;
                break;
            }
            if(got != expected)
            {
                printf("error at out[%d]: got %f expected %f\n", i, got, expected);
                error = EXIT_FAILURE;
                break;
            }
        }
        printf("Stencil1d verification %s\n", error ? "FAILED" : "PASSED");
        return error;
    }

} // namespace stencil1d
