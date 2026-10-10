/* Copyright 2026 René Widera
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

/** @file Host reference and verification for the stencil1d benchmark.
 *
 * The HeCBench reference builds an input `a[i] = i` for `i` in `[0, length + RADIUS)` and computes
 * the radius-RADIUS sliding sum. Values left of the domain are treated as zero (`a[j] = 0` for `j < 0`);
 * values right of `length` use the padded sequence `a[j] = j` up to `length + RADIUS - 1`. This file
 * reproduces exactly that result in double precision and compares the device result against it.
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

    /** Value of the logical input at position @p j.
     *
     * The HeCBench reference stores `a[i] = i`. Positions left of the domain are zero; positions in
     * `[length, length + RADIUS)` continue the sequence; positions at or beyond `length + RADIUS` are
     * outside the padded buffer and are treated as zero. This matches the reference for every index it
     * reads and extends it to arbitrary (non multiple of the tile size) lengths.
     */
    inline double logicalInput(int64_t j, int64_t length)
    {
        if(j < 0 || j >= length + radius)
            return 0.0;
        return static_cast<double>(j);
    }

    /** Double precision host reference for output element @p i. */
    inline double referenceValue(int64_t i, int64_t length)
    {
        double sum = 0.0;
        for(int offset = -radius; offset <= radius; ++offset)
            sum += logicalInput(i + offset, length);
        return sum;
    }

    /** Verify the device output against the host reference.
     *
     * @param length number of output elements
     * @param out    device result, `length` elements
     * @param in     padded input, `length + radius` elements (used only as an extra consistency check)
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
            double const expected = referenceValue(i, length);
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
            // Cross-check that the device input matches the generation rule a[i] = i.
            if(i < length + radius && in[i] != static_cast<int>(i))
            {
                printf("error at in[%d]: got %d expected %d\n", i, static_cast<int>(in[i]), i);
                error = EXIT_FAILURE;
                break;
            }
        }
        printf("Stencil1d verification %s\n", error ? "FAILED" : "PASSED");
        return error;
    }

} // namespace stencil1d
