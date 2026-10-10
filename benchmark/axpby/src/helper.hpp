/* Copyright 2026 Rene Widera
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

/** @file Host-side input generation, reference and verification helpers for the axpby benchmark.
 * @attention The license differs from the benchmark implementation because this file contains no parts of the
 * HeCBench code base.
 */

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace axpby
{
    /** Generate the per-tensor lengths and the two input tensors exactly like the HeCBench reference.
     *
     * The HeCBench `src/axpby-cuda/main.cu` and `src/axpby-omp/main.cpp` use a single `srand(123)` stream:
     * per tensor first a length `rand() % (1024*1024) + 1024` is drawn, then `length` values for `x` and `length`
     * values for `y` (the output list draws nothing). The generated `x`/`y` are stored in flat vectors whose
     * per-tensor base offset is returned in `offsets`.
     *
     * @param fixedLength if greater than zero every tensor has exactly this length instead of the drawn value
     *                    (the random value stream is then replayed with the forced lengths).
     */
    inline void generateInputs(
        int numTensors,
        int fixedLength,
        std::vector<int64_t>& lengths,
        std::vector<int64_t>& offsets,
        std::vector<float>& x,
        std::vector<float>& y)
    {
        lengths.assign(static_cast<std::size_t>(numTensors), 0);
        offsets.assign(static_cast<std::size_t>(numTensors), 0);
        x.clear();
        y.clear();

        // rough reserve to avoid many reallocations: average drawn length is ~0.5M
        std::size_t const reserveElems = static_cast<std::size_t>(numTensors) * 600'000u;
        x.reserve(reserveElems);
        y.reserve(reserveElems);

        srand(123);
        for(int n = 0; n < numTensors; ++n)
        {
            int64_t const length
                = fixedLength > 0 ? static_cast<int64_t>(fixedLength)
                                  : static_cast<int64_t>(rand() % (1024 * 1024) + 1024);
            lengths[static_cast<std::size_t>(n)] = length;
            offsets[static_cast<std::size_t>(n)] = static_cast<int64_t>(x.size());

            for(int64_t i = 0; i < length; ++i)
                x.push_back(static_cast<float>(rand() % length));
            for(int64_t i = 0; i < length; ++i)
                y.push_back(static_cast<float>(rand() % length));
        }
    }

    /** Independent double-precision reference for the fused `out = a*x + b*y`, checked per tensor.
     *
     * The tested kernel computes `a * (float)x + b * (float)y` in single precision. The inputs are non-negative
     * integers below 2^20 and the default `a = b = 1`, so every sum is below 2^24 and is exactly representable as
     * `float`. The double reference is therefore exact for the default configuration; the tolerance below only
     * guards against compiler contraction differences.
     *
     * Non-finite samples in the produced output are reported and fail the check.
     *
     * @return EXIT_SUCCESS when all elements match, EXIT_FAILURE otherwise.
     */
    inline int verify(
        std::vector<int64_t> const& lengths,
        std::vector<int64_t> const& offsets,
        float const* x,
        float const* y,
        float const* out,
        float a,
        float b)
    {
        int error = EXIT_SUCCESS;
        constexpr double atol = 1e-3; // matches the HeCBench `fabsf(...) > 1e-3` check
        constexpr double rtol = 1e-6; // a few ULPs of fp32 for the single fused rounding
        int failures = 0;
        int64_t checked = 0;

        for(std::size_t t = 0; t < lengths.size(); ++t)
        {
            int64_t const off = offsets[t];
            int64_t const len = lengths[t];
            for(int64_t i = 0; i < len; ++i)
            {
                int64_t const idx = off + i;
                double const ref = static_cast<double>(a) * static_cast<double>(x[idx])
                                   + static_cast<double>(b) * static_cast<double>(y[idx]);
                double const got = static_cast<double>(out[idx]);
                ++checked;
                if(!std::isfinite(got))
                {
                    if(failures < 5)
                        printf("non-finite output at tensor %zu element %lld: %f\n", t, (long long)i, got);
                    error = EXIT_FAILURE;
                    ++failures;
                    continue;
                }
                double const diff = std::fabs(got - ref);
                double const tol = atol + rtol * std::fabs(ref);
                if(diff > tol)
                {
                    if(failures < 5)
                        printf(
                            "error at tensor %zu element %lld: got %.9g expected %.9g (diff %.3g > tol %.3g)\n",
                            t,
                            (long long)i,
                            got,
                            ref,
                            diff,
                            tol);
                    error = EXIT_FAILURE;
                    ++failures;
                }
                if(failures >= 5)
                    break;
            }
            if(failures >= 5)
                break;
        }

        printf(
            "checked %lld elements across %zu tensors: %s\n",
            (long long)checked,
            lengths.size(),
            error ? "Verification FAILED" : "Verification PASSED");
        return error;
    }
} // namespace axpby
