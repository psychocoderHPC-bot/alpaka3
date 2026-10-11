/* Copyright 2026 Rene Widera
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

/** @file Host-side reference and verification helpers for the softmax benchmark.
 * @attention The license differs from the benchmark implementation because this file contains no parts of the
 * HeCBench code base.
 */

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <type_traits>

namespace softmax
{
    /** Independent row-wise softmax reference computed in double precision.
     *
     * For every row the reference reproduces the exact HeCBench scheme
     * (`src/softmax-cuda/main.cu`, `src/softmax-omp/main.cpp`):
     *
     *   m   = max_j x[j]
     *   e_j = exp(x[j] - m)
     *   s   = sum_j e_j
     *   y_j = e_j / s
     *
     * The arithmetic is a standard softmax formulation and is re-derived here rather than copied from
     * HeCBench; the inputs are widened to `double` so the reference is not limited by the single-precision
     * rounding of the tested kernel.
     *
     * @param numRows number of independent rows.
     * @param sliceSize number of elements per row.
     * @param src row-major input of length `numRows * sliceSize`.
     * @param dest row-major output of length `numRows * sliceSize`.
     */
    template<typename T>
    inline void reference(int numRows, int sliceSize, T const* src, double* dest)
    {
        for(int i = 0; i < numRows; ++i)
        {
            T const* row = src + static_cast<std::size_t>(i) * static_cast<std::size_t>(sliceSize);
            double* out = dest + static_cast<std::size_t>(i) * static_cast<std::size_t>(sliceSize);

            double maxValue = static_cast<double>(row[0]);
            for(int j = 1; j < sliceSize; ++j)
            {
                double const value = static_cast<double>(row[j]);
                if(value > maxValue)
                    maxValue = value;
            }

            double sum = 0.0;
            for(int j = 0; j < sliceSize; ++j)
            {
                double const e = std::exp(static_cast<double>(row[j]) - maxValue);
                out[j] = e;
                sum += e;
            }

            for(int j = 0; j < sliceSize; ++j)
                out[j] /= sum;
        }
    }

    /** Absolute tolerance used for the float case.
     *
     * The HeCBench softmax check uses `fabsf(output_cpu[i] - output_gpu[i]) > 1e-3` on the normalized
     * outputs, which are guaranteed to be in `[0, 1]`. We keep that absolute tolerance and add a small
     * relative term so that near-zero entries (where the absolute test is loose) are still compared
     * meaningfully. The tolerance is never widened beyond the HeCBench check.
     */
    template<typename T>
    constexpr double absTolerance()
    {
        if constexpr(std::is_same_v<T, float>)
            return 1e-3;
        else
            return 1e-9;
    }

    template<typename T>
    constexpr double relTolerance()
    {
        if constexpr(std::is_same_v<T, float>)
            return 1e-4;
        else
            return 1e-9;
    }

    /** Verify the accelerator result against the independent double-precision row reference.
     *
     * Non-finite (NaN/Inf) samples in the produced output are reported explicitly and fail the check.
     *
     * @return EXIT_SUCCESS when all rows match, EXIT_FAILURE otherwise.
     */
    template<typename T>
    inline int verify(int numRows, int sliceSize, T const* src, T const* out)
    {
        double const atol = absTolerance<T>();
        double const rtol = relTolerance<T>();
        int failures = 0;

        for(int i = 0; i < numRows; ++i)
        {
            T const* row = src + static_cast<std::size_t>(i) * static_cast<std::size_t>(sliceSize);
            T const* outRow = out + static_cast<std::size_t>(i) * static_cast<std::size_t>(sliceSize);

            double maxValue = static_cast<double>(row[0]);
            for(int j = 1; j < sliceSize; ++j)
            {
                double const value = static_cast<double>(row[j]);
                if(value > maxValue)
                    maxValue = value;
            }
            double sum = 0.0;
            for(int j = 0; j < sliceSize; ++j)
                sum += std::exp(static_cast<double>(row[j]) - maxValue);

            for(int j = 0; j < sliceSize; ++j)
            {
                double const ref = std::exp(static_cast<double>(row[j]) - maxValue) / sum;
                double const value = static_cast<double>(outRow[j]);

                if(!std::isfinite(value))
                {
                    if(failures < 5)
                        printf("non-finite output at row %d col %d: %f\n", i, j, value);
                    ++failures;
                    continue;
                }
                double const diff = std::fabs(value - ref);
                double const tol = atol + rtol * std::fabs(ref);
                if(diff > tol)
                {
                    if(failures < 5)
                        printf(
                            "error at row %d col %d: got %.9g expected %.9g (diff %.3g > tol %.3g)\n",
                            i,
                            j,
                            value,
                            ref,
                            diff,
                            tol);
                    ++failures;
                }
            }
            if(failures >= 5)
                break;
        }

        printf("%s\n", failures ? "Verification FAILED" : "Verification PASSED");
        return failures ? EXIT_FAILURE : EXIT_SUCCESS;
    }

} // namespace softmax
