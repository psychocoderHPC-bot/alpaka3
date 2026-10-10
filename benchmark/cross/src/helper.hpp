/* Copyright 2026 Rene Widera
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

/** @file Host-side reference and verification helpers for the cross benchmark.
 * @attention The license differs from the benchmark implementation because this file contains no parts of the
 * HeCBench code base.
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>

namespace cross
{
    /** Reference 3D cross product, computed in double precision from the two row vectors `x1` and `x2`.
     *
     * The low-precision inputs are widened to double before the products and differences are formed, so the
     * reference is not affected by the rounding of the tested single-precision kernel.
     */
    template<typename T>
    inline void crossReference(T const* x1, T const* x2, double* out)
    {
        double const a0 = static_cast<double>(x1[0]);
        double const a1 = static_cast<double>(x1[1]);
        double const a2 = static_cast<double>(x1[2]);
        double const b0 = static_cast<double>(x2[0]);
        double const b1 = static_cast<double>(x2[1]);
        double const b2 = static_cast<double>(x2[2]);

        out[0] = a1 * b2 - a2 * b1;
        out[1] = a2 * b0 - a0 * b2;
        out[2] = a0 * b1 - a1 * b0;
    }

    /** Absolute tolerance used for the float case, matching the HeCBench reference check.
     *
     * Relative tolerance used in addition to catch results whose magnitude exceeds the [-2, 2] member range; the
     * HeCBench reference only uses an absolute tolerance.
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

    /** Verify the accelerator result against the double-precision reference.
     *
     * Non-finite samples in the produced output are reported explicitly and fail the check.
     *
     * @return EXIT_SUCCESS when all rows match, EXIT_FAILURE otherwise.
     */
    template<typename T>
    inline int verify(int nrows, T const* out, T const* x1, T const* x2)
    {
        int error = EXIT_SUCCESS;
        double const atol = absTolerance<T>();
        double const rtol = relTolerance<T>();
        int failures = 0;

        for(int i = 0; i < nrows; ++i)
        {
            double ref[3];
            crossReference(&x1[3 * i], &x2[3 * i], ref);

            for(int k = 0; k < 3; ++k)
            {
                double const got = static_cast<double>(out[3 * i + k]);
                if(!std::isfinite(got))
                {
                    if(failures < 5)
                        printf("non-finite output at row %d component %d: %f\n", i, k, got);
                    error = EXIT_FAILURE;
                    ++failures;
                    continue;
                }
                double const diff = std::fabs(got - ref[k]);
                double const tol = atol + rtol * std::fabs(ref[k]);
                if(diff > tol)
                {
                    if(failures < 5)
                        printf(
                            "error at row %d component %d: got %.9g expected %.9g (diff %.3g > tol %.3g)\n",
                            i,
                            k,
                            got,
                            ref[k],
                            diff,
                            tol);
                    error = EXIT_FAILURE;
                    ++failures;
                }
            }
            if(failures >= 5)
                break;
        }

        printf("%s\n", error ? "Verification FAILED" : "Verification PASSED");
        return error;
    }

} // namespace cross
