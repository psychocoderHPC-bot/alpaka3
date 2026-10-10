/* Copyright 2026 René Widera
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

/** @file Host reference and verification for the reverse benchmark.
 *
 * The HeCBench `reverse` reference reverses a length-256 `int` array in place on the device. It
 * verifies by comparing the device buffer with `gold_even[i] = i` after an even number of reverse
 * operations and with `gold_odd[i] = len - i - 1` after an odd number, using `memcmp` over the whole
 * array. The device writes are exact integer copies, so the comparison is exact.
 *
 * This file reproduces the expected sequence in a host array with the same integer type and compares
 * it element by element against the device result. `memcmp` is reproduced exactly by comparing the
 * raw bytes of a plain `int` array, so the semantics are identical while the diagnostics are better.
 *
 * The comparison is exact (tolerance 0). Non-finite checks are performed even though the data is
 * integral, to reject a corrupted or reinterpreted result buffer.
 */

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdio.h>

namespace reverse
{
    /** Build the HeCBench expected buffer for a given parity.
     *
     * @param len    number of elements (the reference uses 256)
     * @param odd    true for an odd number of applied reverses, false for an even number
     * @param gold   output array with @p len elements
     *
     * The reference fills `gold_odd[i] = len - i - 1` and `gold_even[i] = i` once, before the timed
     * loop. This function produces byte-identical arrays.
     */
    inline void makeGold(int len, bool odd, int* gold)
    {
        for(int i = 0; i < len; ++i)
            gold[i] = odd ? (len - i - 1) : i;
    }

    /** Verify the device result against the expected sequence.
     *
     * @param len     number of elements
     * @param result  device result copied back to the host, @p len elements
     * @param odd     parity of the number of reverse operations used to produce @p result
     *
     * The comparison is exact and mirrors the reference `memcmp(result, gold, len * sizeof(int))`.
     * On the first mismatch the function prints the offending element and index and returns failure.
     */
    inline int verify(int len, int const* result, bool odd)
    {
        int error = EXIT_SUCCESS;
        for(int i = 0; i < len; ++i)
        {
            double const got = static_cast<double>(result[i]);
            double const expected = static_cast<double>(odd ? (len - i - 1) : i);
            if(!std::isfinite(got) || !std::isfinite(expected))
            {
                printf("error at result[%d]: non-finite value got %f expected %f\n", i, got, expected);
                error = EXIT_FAILURE;
                break;
            }
            if(result[i] != static_cast<int>(expected))
            {
                printf("error at result[%d]: got %d expected %d\n", i, result[i], static_cast<int>(expected));
                error = EXIT_FAILURE;
                break;
            }
        }
        // Reproduce the reference memcmp over the raw bytes as an independent cross-check.
        int* gold = static_cast<int*>(std::malloc(sizeof(int) * static_cast<size_t>(len)));
        makeGold(len, odd, gold);
        if(error == EXIT_SUCCESS && std::memcmp(result, gold, sizeof(int) * static_cast<size_t>(len)) != 0)
        {
            printf("error: memcmp against the expected buffer failed\n");
            error = EXIT_FAILURE;
        }
        std::free(gold);
        if(error != EXIT_SUCCESS)
            printf("Reverse verification FAILED\n");
        return error;
    }

} // namespace reverse
