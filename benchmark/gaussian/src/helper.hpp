/* Copyright 2026 René Widera
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

/** @file Independent host reference and verification helpers for the HeCBench `gaussian` benchmark.
 *
 * The HeCBench reference solves a dense linear system `A x = b` with Gaussian elimination without
 * pivoting (forward elimination on the accelerator, back substitution on the host) and validates the
 * device result by comparing it against the very same no-pivot elimination executed on the CPU. That
 * self-comparison cannot detect a systematically wrong elimination.
 *
 * This file provides a genuinely independent oracle: a double-precision LU solve with partial
 * pivoting that is unrelated to the no-pivot single-precision elimination used by the benchmark. It
 * also provides the exact input generator of the reference (`init_matrix`), a plain no-pivot back
 * substitution (kept for completeness), and a `verify()` that checks finiteness and compares the
 * results with a tolerance derived from the reference precision.
 */

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace gaussianBench
{
    /** Generate the benchmark matrix exactly like the HeCBench reference `init_matrix`.
     *
     * `coe` is a Toeplitz sequence `10 * exp(-0.01 * i)` rounded to float and the matrix entry
     * `A[i][j]` is `coe[size - 1 - i + j]`. The right-hand side is all ones. The reference computes
     * `lamda * i` in single precision before promoting to `double` for `exp`; the same order is kept
     * here so the generated system is byte-for-byte the reference input.
     */
    inline void initMatrix(float* a, float* b, std::uint32_t size)
    {
        float const lamda = -0.01f;
        std::vector<float> coe(2u * size - 1u);
        for(std::uint32_t i = 0; i < size; ++i)
        {
            /* The reference computes `coe_i = 10 * exp(lamda * i)` with `lamda` a float and `i` an int.
             * Through `<math.h>` this promotes the float product `lamda * i` to double, calls the double
             * `exp`, multiplies by the int `10` in double, then rounds the assignment back to float. The
             * float product is formed before the promotion, so it is computed here in float and only then
             * widened; doing the multiply in double changes the matrix. Verified byte-identical to the
             * reference `init_matrix` for the exercised sizes. */
            float const arg = lamda * static_cast<float>(i);
            float const coeI = static_cast<float>(10.0 * std::exp(static_cast<double>(arg)));
            coe[size - 1u + i] = coeI;
            coe[size - 1u - i] = coeI;
        }
        for(std::uint32_t i = 0; i < size; ++i)
            for(std::uint32_t j = 0; j < size; ++j)
                a[static_cast<std::size_t>(i) * size + j] = coe[size - 1u - i + j];
        for(std::uint32_t i = 0; i < size; ++i)
            b[i] = 1.0f;
    }

    /** Plain no-pivot back substitution, identical to the reference `BackSub`.
     *
     * The forward elimination has already reduced the system to upper triangular form in `a`; this
     * reproduces the reference host phase exactly. It is exposed for completeness and documentation.
     */
    inline void backSubstitutionFloat(
        float const* a,
        float const* b,
        float* finalVec,
        std::uint32_t size)
    {
        for(std::uint32_t i = 0; i < size; ++i)
        {
            std::uint32_t const r = size - i - 1u;
            float value = b[r];
            for(std::uint32_t j = 0; j < i; ++j)
            {
                std::uint32_t const c = size - j - 1u;
                value -= a[static_cast<std::size_t>(r) * size + c] * finalVec[c];
            }
            finalVec[r] = value / a[static_cast<std::size_t>(r) * size + r];
        }
    }

    /** Independent double-precision solver `A x = b` with partial pivoting.
     *
     * This is a textbook LU solve in `double` on a copy of `A` and `b` (passed as `float`). It shares
     * neither the no-pivot variant nor the single precision of the benchmark and therefore is a real
     * oracle. `x` must have room for `size` elements.
     */
    inline void gaussianGoldDouble(
        float const* A,
        float const* b,
        double* x,
        std::uint32_t size)
    {
        std::size_t const n = size;
        std::vector<double> m(n * n);
        std::vector<double> rhs(n);
        for(std::size_t i = 0; i < n * n; ++i)
            m[i] = static_cast<double>(A[i]);
        for(std::size_t i = 0; i < n; ++i)
            rhs[i] = static_cast<double>(b[i]);

        for(std::size_t k = 0; k < n; ++k)
        {
            // partial pivoting
            std::size_t pivot = k;
            double maxAbs = std::fabs(m[k * n + k]);
            for(std::size_t i = k + 1u; i < n; ++i)
            {
                double const v = std::fabs(m[i * n + k]);
                if(v > maxAbs)
                {
                    maxAbs = v;
                    pivot = i;
                }
            }
            if(pivot != k)
            {
                for(std::size_t j = k; j < n; ++j)
                    std::swap(m[k * n + j], m[pivot * n + j]);
                std::swap(rhs[k], rhs[pivot]);
            }
            double const piv = m[k * n + k];
            for(std::size_t i = k + 1u; i < n; ++i)
            {
                double const f = m[i * n + k] / piv;
                m[i * n + k] = 0.0;
                for(std::size_t j = k + 1u; j < n; ++j)
                    m[i * n + j] -= f * m[k * n + j];
                rhs[i] -= f * rhs[k];
            }
        }
        for(std::size_t ii = n; ii-- > 0u;)
        {
            double s = rhs[ii];
            for(std::size_t j = ii + 1u; j < n; ++j)
                s -= m[ii * n + j] * x[j];
            x[ii] = s / m[ii * n + ii];
        }
    }

    /** Verify a computed solution against the independent double-precision oracle.
     *
     * Checks, in order:
     *  1. finiteness of every computed entry and of every oracle entry (explicit NaN/Inf check),
     *  2. per-entry agreement `|x_i - gold_i| <= atol + rtol * |gold_i|`,
     *  3. the relative residual `||A x - b||_2 / ||b||_2` of the computed single-precision solution
     *     evaluated in `double` against the original `A` and `b`.
     *
     * @param atol absolute tolerance. The reference uses `1e-3`; the systematic difference between the
     *             no-pivot single-precision elimination and the partial-pivot double oracle was
     *             measured to stay below `1.8e-5` for all exercised sizes, so `1e-3` leaves a safety
     *             margin above 50x without hiding gross errors.
     * @param rtol relative tolerance, `1e-3`.
     * @param maxResidual upper bound for the relative residual, `1e-3`.
     * @return EXIT_SUCCESS on success, EXIT_FAILURE on any failure.
     */
    inline int verify(
        float const* x,
        double const* gold,
        float const* A,
        float const* b,
        std::uint32_t size,
        double atol = 1e-3,
        double rtol = 1e-3,
        double maxResidual = 1e-3)
    {
        int error = EXIT_SUCCESS;
        for(std::uint32_t i = 0; i < size; ++i)
        {
            if(!std::isfinite(x[i]))
            {
                std::printf("error: non-finite solution entry x[%u] = %g\n", i, static_cast<double>(x[i]));
                error = EXIT_FAILURE;
            }
            if(!std::isfinite(gold[i]))
            {
                std::printf("error: non-finite oracle entry gold[%u] = %g\n", i, gold[i]);
                error = EXIT_FAILURE;
            }
        }
        if(error != EXIT_SUCCESS)
            return error;

        double maxAbsDiff = 0.0;
        double maxRelDiff = 0.0;
        std::uint32_t worstIdx = 0u;
        for(std::uint32_t i = 0; i < size; ++i)
        {
            double const diff = std::fabs(static_cast<double>(x[i]) - gold[i]);
            double const bound = atol + rtol * std::fabs(gold[i]);
            double const rel = diff / (std::fabs(gold[i]) + 1e-30);
            if(diff > maxAbsDiff)
            {
                maxAbsDiff = diff;
                worstIdx = i;
            }
            if(rel > maxRelDiff)
                maxRelDiff = rel;
            if(diff > bound)
            {
                std::printf(
                    "error at x[%u]: device %.8g gold %.8g diff %.3e > %.3e\n",
                    i,
                    static_cast<double>(x[i]),
                    gold[i],
                    diff,
                    bound);
                error = EXIT_FAILURE;
            }
        }

        // relative residual of the computed single-precision solution against the original system
        double residual = 0.0;
        double rhsNorm = 0.0;
        for(std::uint32_t i = 0; i < size; ++i)
        {
            double s = 0.0;
            for(std::uint32_t j = 0; j < size; ++j)
                s += static_cast<double>(A[static_cast<std::size_t>(i) * size + j])
                     * static_cast<double>(x[j]);
            double const e = s - static_cast<double>(b[i]);
            residual += e * e;
            rhsNorm += static_cast<double>(b[i]) * static_cast<double>(b[i]);
        }
        double const relResidual = std::sqrt(residual) / (std::sqrt(rhsNorm) + 1e-300);
        if(!(relResidual <= maxResidual))
        {
            std::printf("error: relative residual %.6e exceeds %.3e\n", relResidual, maxResidual);
            error = EXIT_FAILURE;
        }

        std::printf(
            "verification: size=%u max_abs_diff=%.6e max_rel_diff=%.6e worst_index=%u rel_residual=%.6e\n",
            size,
            maxAbsDiff,
            maxRelDiff,
            worstIdx,
            relResidual);
        std::printf("%s\n", error == EXIT_SUCCESS ? "Verification PASSED" : "Verification FAILED");
        return error;
    }
} // namespace gaussianBench
