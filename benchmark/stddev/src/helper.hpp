/* Copyright (c) 2018-2020, NVIDIA CORPORATION.
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

/** @file Host reference, input generation and verification for the stddev benchmark.
 *
 * The HeCBench reference (`src/stddev-cuda/main.cu`) is derived from NVIDIA RAPIDS `raft` code and is
 * distributed under Apache-2.0. This file contains no HeCBench code; it provides an independent
 * double-precision host evaluation of the same mathematical quantity and a comparator.
 *
 * The pinned HeCBench kernel computes, per column `c` of an `N x D` row-major matrix,
 *
 *     out[c] = sqrt( sum_{r=0}^{N-1} data[r*D + c]^2 / sampleSize )
 *
 * with `sampleSize = N-1` for the sample variant used by the benchmark (`sample = true`). Despite its
 * name the routine is therefore a root-mean-square (sum of squares) reduction, not a mean centred
 * standard deviation: a mean centred value would not reproduce the kernel. This file evaluates the
 * exact pinned formula in `double` precision with a different memory order and a distinct second
 * pass, which makes it an independent oracle rather than a copy of the device reduction.
 */

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace stddevBench
{
    /** Fill the input exactly like the pinned HeCBench reference.
     *
     * The reference does `srand(123)` and then fills the row-major matrix with
     * `rand() / (float)RAND_MAX` in increasing flat order. Reproducing the same libc generator and
     * call order keeps the benchmark input identical to the reference on this platform.
     */
    inline void generateInput(float* data, size_t flatSize)
    {
        std::srand(123);
        for(size_t i = 0; i < flatSize; ++i)
            data[i] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
    }

    /** Independent double-precision evaluation of the pinned stddev formula.
     *
     * Pass 1 accumulates the sum of squares of every column in `double` while streaming the input in
     * linear order (cache friendly, unlike the device's strided reduction). Pass 2 normalises with
     * `sampleSize` and takes the square root in `double`.
     *
     * @param out  output vector of @p D doubles
     * @param data row-major `N x D` input in `float`
     * @param D    number of columns
     * @param N    number of rows
     * @param sample true -> divide by `N-1`, false -> divide by `N`
     * @return `EXIT_SUCCESS` on success, `EXIT_FAILURE` on an invalid size
     */
    inline int hostReference(double* out, float const* data, uint32_t D, uint32_t N, bool sample)
    {
        uint32_t const sampleSize = sample ? (N - 1u) : N;
        if(sampleSize == 0u)
        {
            std::printf("error: sample stddev requires at least two rows, got N=%u\n", N);
            return EXIT_FAILURE;
        }

        std::vector<double> sumOfSquares(static_cast<size_t>(D), 0.0);
        for(uint32_t r = 0; r < N; ++r)
        {
            float const* row = data + static_cast<size_t>(r) * D;
            for(uint32_t c = 0; c < D; ++c)
            {
                double const v = static_cast<double>(row[c]);
                sumOfSquares[c] += v * v;
            }
        }

        for(uint32_t c = 0; c < D; ++c)
            out[c] = std::sqrt(sumOfSquares[c] / static_cast<double>(sampleSize));

        return EXIT_SUCCESS;
    }

    /** Compare a device result against the double host reference.
     *
     * Explicitly rejects NaN/Inf values in either buffer. The tolerance combines an absolute and a
     * relative term: the pinned HeCBench reference itself accepts `fabsf(ref - got) <= 1e-3` on the
     * same data, and the device result uses `float` accumulation with an atomics-dependent summation
     * order, which is the only source of the deviation. The absolute term is therefore the reference's
     * own tolerance; it is not widened beyond it.
     */
    inline int verify(uint32_t D, float const* got, double const* ref, double atol = 1e-3, double rtol = 1e-3)
    {
        int error = EXIT_SUCCESS;
        unsigned reported = 0;
        for(uint32_t c = 0; c < D; ++c)
        {
            double const g = static_cast<double>(got[c]);
            double const r = ref[c];
            if(!std::isfinite(g) || !std::isfinite(r))
            {
                std::printf("error at std[%u]: non-finite value got %f reference %f\n", c, g, r);
                return EXIT_FAILURE;
            }
            double const tol = atol + rtol * std::fabs(r);
            if(std::fabs(g - r) > tol)
            {
                if(reported < 10u)
                    std::printf(
                        "error at std[%u]: got %.8g reference %.8g (|diff|=%.3g tol=%.3g)\n",
                        c,
                        g,
                        r,
                        std::fabs(g - r),
                        tol);
                ++reported;
                error = EXIT_FAILURE;
            }
        }
        if(error != EXIT_SUCCESS)
            std::printf("stddev verification FAILED (%u column(s) outside tolerance)\n", reported);
        return error;
    }
} // namespace stddevBench
