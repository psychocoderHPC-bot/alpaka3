/* Copyright 2026 René Widera
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

/** @file Independent host reference and verification for the jacobi benchmark.
 *
 * The HeCBench `jacobi` benchmark solves the 2D Laplace equation on an `N x N` grid with a
 * five-point Jacobi stencil. The four grid edges hold a fixed sinusoidal boundary condition, the
 * interior starts at zero, and the solver iterates until the normalised L2 change per step drops
 * below `tol`. This file re-implements that computation in **double precision** on the host and
 * compares it to the (single precision) device result.
 *
 * Independence: this reference shares no indexing, tiling or reduction helper with the device
 * kernel. It re-derives the boundary values from `sin(...)`, applies the plain five-point average
 * over the full domain and computes the L2 change directly. A kernel with a wrong halo, a wrong
 * boundary update or a wrong reduction therefore cannot pass by construction.
 *
 * The device result is single precision (`float`) while the reference is `double`, so the
 * comparison is not bit-exact. The tolerance follows from the single-precision spacing of the
 * solution: the update is a convex average of the previous field and the boundary stays in
 * `[-1, 1]`, so `|u| <= 1` and one `float` ulp near 1 is `2^-24 = 5.96e-8`. The measured
 * worst-case difference over the validated sizes is a few ulps; `verifyTolerance` keeps a documented
 * safety margin (see README) rather than a widened bound. NaN/Inf are rejected explicitly.
 */

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

namespace jacobi
{
    //! Single-precision residual bound: ~1600x one float ulp, covering the few-ulp measured spread.
    constexpr float verifyTolerance = 1e-4f;

    /** HeCBench initial condition, evaluated in double precision.
     *
     * `f[i + j*n] = sin(j * 2pi / (n-1))` on the left/right edge, `sin(i * 2pi / (n-1))` on the
     * bottom/top edge, `0` in the interior. This mirrors `initialize_data` in the pinned reference.
     */
    inline void initializeData(std::vector<double>& f, int n)
    {
        double const denom = static_cast<double>(n - 1);
        for(int j = 0; j < n; ++j)
        {
            for(int i = 0; i < n; ++i)
            {
                std::size_t const idx = static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * n;
                if(i == 0 || i == n - 1)
                    f[idx] = std::sin(j * 2.0 * M_PI / denom);
                else if(j == 0 || j == n - 1)
                    f[idx] = std::sin(i * 2.0 * M_PI / denom);
                else
                    f[idx] = 0.0;
            }
        }
    }

    /** Advance one five-point Jacobi step from @p old_ into @p cur.
     *
     * Only interior points `1..n-2` are updated; the boundary stays fixed. The L2 change
     * `sqrt(sum(df^2) / (n*n))` is returned as the convergence measure. The field update is
     * deterministic; the reduction order of the scalar error does not affect the field.
     */
    inline double step(std::vector<double> const& old_, std::vector<double>& cur, int n)
    {
        double err = 0.0;
#ifdef _OPENMP
#    pragma omp parallel for collapse(2) reduction(+ : err) schedule(static)
#endif
        for(int j = 1; j <= n - 2; ++j)
        {
            for(int i = 1; i <= n - 2; ++i)
            {
                std::size_t const idx = static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * n;
                double const t = 0.25
                                 * (old_[idx - 1] + old_[idx + 1] + old_[idx - static_cast<std::size_t>(n)]
                                    + old_[idx + static_cast<std::size_t>(n)]);
                double const df = t - old_[idx];
                cur[idx] = t;
                err += df * df;
            }
        }
        return std::sqrt(err / (static_cast<double>(n) * static_cast<double>(n)));
    }

    /** Run the double-precision reference for exactly @p iters steps and return the final field. */
    inline std::vector<double> referenceField(int n, int iters)
    {
        std::vector<double> f(static_cast<std::size_t>(n) * n);
        std::vector<double> fOld(static_cast<std::size_t>(n) * n);
        initializeData(f, n);
        initializeData(fOld, n);
        for(int it = 0; it < iters; ++it)
        {
            step(fOld, f, n);
            std::swap(f, fOld);
        }
        // after the last swap the freshly written field is in fOld
        return fOld;
    }

    /** Verify the device result against the independent double-precision reference.
     *
     * Checks:
     *  - the reported iteration count is positive;
     *  - every device value is finite (NaN/Inf rejected explicitly);
     *  - the final field matches the double reference computed to the same iteration within
     *    @c jacobi::verifyTolerance.
     *
     * @param n     grid side length
     * @param iters iteration count reported by the device
     * @param result device result, `n*n` elements in row-major order
     * @return EXIT_SUCCESS on pass, EXIT_FAILURE on failure
     */
    template<typename T_Result>
    int verify(int n, int iters, T_Result const& result)
    {
        std::size_t const count = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);

        if(iters <= 0)
        {
            printf("Jacobi verification FAILED: iteration count %d is not positive\n", iters);
            return EXIT_FAILURE;
        }

        for(std::size_t k = 0; k < count; ++k)
        {
            float const v = static_cast<float>(result[k]);
            if(!std::isfinite(v))
            {
                printf("Jacobi verification FAILED: non-finite value at index %zu\n", k);
                return EXIT_FAILURE;
            }
        }

        std::vector<double> const reference = referenceField(n, iters);
        double maxDiff = 0.0;
        std::size_t maxIdx = 0;
        for(std::size_t k = 0; k < count; ++k)
        {
            double const diff = std::fabs(static_cast<double>(static_cast<float>(result[k])) - reference[k]);
            if(diff > maxDiff)
            {
                maxDiff = diff;
                maxIdx = k;
            }
        }

        if(maxDiff > static_cast<double>(verifyTolerance))
        {
            printf(
                "Jacobi verification FAILED: max |alpaka - double ref| = %.6e at index %zu exceeds %.6e\n",
                maxDiff,
                maxIdx,
                static_cast<double>(verifyTolerance));
            return EXIT_FAILURE;
        }

        printf(
            "Jacobi verification PASSED: N=%d iters=%d max |alpaka - double ref| = %.6e (tolerance %.6e)\n",
            n,
            iters,
            maxDiff,
            static_cast<double>(verifyTolerance));
        return EXIT_SUCCESS;
    }
} // namespace jacobi
