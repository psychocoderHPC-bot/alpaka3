/* Copyright 2026 René Widera
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

/** @file Host reference and verification for the popcount benchmark.
 *
 * The reference implements the scalar `popcount_ref` of the HeCBench reference exactly: Kernighan's
 * `x &= x - 1` loop, returning the number of set bits. The result is an exact integer count, so the
 * comparison is bit-exact and no floating point tolerance is required.
 */

#include <stdint.h>
#include <stdio.h>

#include <cstdlib>

namespace popcount
{
    /** Count the set bits with the Kernighan loop.
     *
     * This is a direct copy of the HeCBench reference `popcount_ref`.
     */
    ALPAKA_FN_HOST_ACC inline int popcountReference(uint64_t x)
    {
        int count;
        for(count = 0; x; count++)
            x &= x - 1;
        return count;
    }

    /** Verify the device result against the host reference.
     *
     * The operation produces exact integer counts in `[0, 64]`, so the comparison is bit-exact.
     * Non-finite values cannot occur; the range check also rejects any out of range value.
     */
    template<typename T_Out, typename T_In>
    int verify(int count, T_Out const& out, T_In const& in)
    {
        int error = EXIT_SUCCESS;
        for(int i = 0; i < count; ++i)
        {
            int const expected = popcountReference(static_cast<uint64_t>(in[i]));
            int const got = static_cast<int>(out[i]);
            if(got != expected || got < 0 || got > 64)
            {
                printf("error at result[%d]: got %d expected %d\n", i, got, expected);
                error = EXIT_FAILURE;
                break;
            }
        }
        printf("Popcount verification %s\n", error ? "FAILED" : "PASSED");
        return error;
    }

} // namespace popcount
