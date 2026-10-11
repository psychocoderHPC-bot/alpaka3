/* Copyright 2026 René Widera
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

/** @file Host reference, deterministic input generation and verification for the `scan` benchmark.
 *
 * The pinned HeCBench reference (`src/scan-cuda/main.cu`, and its OpenMP twin `src/scan-omp/main.cpp`)
 * splits an input array of `nelems = ceil(n / N) * N` elements into independent chunks of `N` elements
 * and computes an *exclusive* prefix sum inside every chunk. Chunks do not carry into each other; the
 * last element of each chunk is discarded (the reference writes `t_out[0] = 0` and then
 * `t_out[i] = t_out[i-1] + t_in[i-1]` per chunk). The device result is compared with an exact,
 * element-wise equality.
 *
 * This file reproduces the input generator (`srand(123)`, `rand() % 5 + 1`) and an independent
 * sequential host exclusive scan with the exact same chunk semantics, so the same input and the same
 * trusted result are used for validation. Nothing in this file is device code.
 */

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace scanBench
{
    //! Seed used by the pinned reference's `srand(123)`.
    constexpr unsigned int rngSeed = 123u;

    /** Generate the reference input: `nelems` values in [1, 5] from the C library RNG with seed 123.
     *
     * Reproduces `srand(123); for (n = 0; n < nelems; n++) in[n] = rand() % 5 + 1;` exactly, including
     * the accumulation in the element type `T` (so `char`/`short` results wrap exactly like the
     * reference).
     */
    template<typename T>
    inline void generateInput(T* in, std::uint64_t nelems)
    {
        std::srand(rngSeed);
        for(std::uint64_t i = 0; i < nelems; ++i)
            in[i] = static_cast<T>(std::rand() % 5 + 1);
    }

    /** Independent sequential host exclusive scan, chunk by chunk.
     *
     * For every chunk `c` of @p chunk elements at offset `c * chunk`, element `i` receives the sum of
     * the chunk's elements `[0, i)`. The scan resets at every chunk boundary, matching the reference.
     * All arithmetic is performed in @p T, so wrap-around for narrow integer types matches the device.
     *
     * @param in        input array of @p nelems elements
     * @param out       output array of @p nelems elements (may alias no other buffer)
     * @param nelems    total number of elements (`numChunks * chunk`)
     * @param chunk     number of elements per independent chunk (`N` in the reference)
     * @param numChunks number of chunks (`nelems / chunk`)
     */
    template<typename T>
    inline void scanGold(
        T const* in,
        T* out,
        std::uint64_t nelems,
        std::uint32_t chunk,
        std::uint64_t numChunks)
    {
        for(std::uint64_t c = 0; c < numChunks; ++c)
        {
            std::uint64_t const base = c * static_cast<std::uint64_t>(chunk);
            T acc = static_cast<T>(0);
            for(std::uint32_t i = 0; i < chunk; ++i)
            {
                T const value = in[base + i];
                out[base + i] = acc;
                acc = static_cast<T>(acc + value);
            }
        }
        (void) nelems;
    }

    //! Finite predicate for integral element types; the reference only scans integral types.
    template<typename T>
    inline bool isFinite(T)
    {
        return true;
    }

    /** Check that every element is finite; returns EXIT_SUCCESS unless a non-finite value is found. */
    template<typename T>
    inline int checkFinite(T const* data, std::uint64_t nelems, char const* label)
    {
        for(std::uint64_t i = 0; i < nelems; ++i)
        {
            if(!isFinite(data[i]))
            {
                std::printf("error: non-finite value at index %llu (%s)\n", static_cast<unsigned long long>(i), label);
                return EXIT_FAILURE;
            }
        }
        return EXIT_SUCCESS;
    }

    /** Exact element-wise comparison of a computed scan against the host reference.
     *
     * A chunked prefix sum over integers is an exact reduction; the reference itself compares with
     * `memcmp`/`!=` (tolerance 0). Any difference is a real bug, so no tolerance is applied. The first
     * mismatch is printed as `index, expected, got` and `EXIT_FAILURE` is returned.
     */
    template<typename T>
    inline int verify(
        T const* computed,
        T const* reference,
        std::uint64_t nelems,
        char const* label)
    {
        for(std::uint64_t i = 0; i < nelems; ++i)
        {
            if(computed[i] != reference[i])
            {
                std::printf(
                    "error: %s mismatch at index %llu expected=%lld got=%lld\n",
                    label,
                    static_cast<unsigned long long>(i),
                    static_cast<long long>(reference[i]),
                    static_cast<long long>(computed[i]));
                return EXIT_FAILURE;
            }
        }
        return EXIT_SUCCESS;
    }

} // namespace scanBench
