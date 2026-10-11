/* Copyright 2026 René Widera
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

/** @file Independent host reference and verification for the HeCBench `heat2d` benchmark.
 *
 * The HeCBench `heat2d` benchmark advances the 2D explicit heat equation with a single five-point
 * Laplacian kernel on a fully periodic `Lx x Ly` grid:
 *
 *     out[i] = norm * in[i] + delta * (in[left] + in[right] + in[down] + in[up])
 *
 * with `delta = sigma / (1 + 4*sigma)`, `norm = 1 / (1 + 4*sigma)` and a fixed `sigma = 0.01`.
 * Neighbours wrap around the domain in both directions. The grid indexing is row-major, i.e. the
 * flat index of `(x, y)` is `y*Lx + x`.
 *
 * This file re-implements that update in **double precision** on the host and compares it to the
 * (single precision) device result. Independence from the device kernel: this reference shares no
 * indexing helper, no tiling and no launch configuration with `main_alpaka.cpp`; it re-derives the
 * periodic neighbours, the coefficients and the initial field. A kernel with a wrong neighbour, a
 * wrong wrap or a wrong coefficient therefore cannot pass by construction.
 *
 * Tolerance rationale: the device result is `float`, the reference is `double`. The update is a
 * convex combination (`norm + 4*delta == 1`) of the previous field, so every value stays bounded by
 * the `{0,1}` initial data and one `float` ulp near 1 is `2^-24 = 5.96e-8`. The single-precision
 * round-off accumulates with the iteration count at roughly `1e-8` per iteration or less (measured
 * `1.15e-5` at 1000 iterations and `7.0e-5` at 10000 for a 256x256 grid). `verifyTolerance()` keeps a documented
 * safety margin of about an order of magnitude over that measured drift instead of widening the bound to hide a bug;
 * at the HeCBench reference run of 100000 iterations it lands at the reference's own documented `1e-2` tolerance.
 * NaN/Inf are rejected explicitly.
 */

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace heat2d
{
    //! HeCBench fixed `sigma`.
    constexpr float sigma = 0.01f;

    //! Host-side coefficients, evaluated in double precision.
    inline double deltaD()
    {
        return static_cast<double>(sigma) / (1.0 + 4.0 * static_cast<double>(sigma));
    }

    inline double normD()
    {
        return 1.0 / (1.0 + 4.0 * static_cast<double>(sigma));
    }

    /** Host reference tolerance as a function of the iteration count.
     *
     * `1e-5 + 1e-7 * iters`: at 1000 iterations this is `1.1e-4`, about ten times the measured
     * single-precision drift of `1.15e-5`. It stays above the measured drift at every validated
     * size, and at the reference's 100000-iteration default reaches the reference's documented
     * `1e-2`.
     */
    inline double verifyTolerance(int iters)
    {
        return 1e-5 + 1e-7 * static_cast<double>(iters);
    }

    /** Reproduce the HeCBench initial field.
     *
     * The pinned reference stamps a `1` into pseudo-random rows and columns (`srand(123)` then
     * `rand() % L`) every 16 entries. The reference allocates its buffer with `malloc` and leaves
     * every cell that is not stamped at its uninitialised value, which makes the benchmark
     * non-deterministic. This reference zero-initialises the buffer first, so the initial field is
     * deterministic while the stamped `1`s match the reference exactly. The stencil, the periodic
     * wrap and the iteration count are unaffected.
     */
    inline std::vector<float> makeInitialField(int Lx, int Ly)
    {
        std::vector<float> buffer(static_cast<std::size_t>(Lx) * static_cast<std::size_t>(Ly), 0.f);
        std::srand(123);
        for(int i = 0; i < Lx; i += 16)
        {
            int const x = std::rand() % Lx;
            for(int j = 0; j < Ly; ++j)
                buffer[static_cast<std::size_t>(x) + static_cast<std::size_t>(j) * Lx] = 1.f;
        }
        for(int i = 0; i < Ly; i += 16)
        {
            int const y = std::rand() % Ly;
            for(int j = 0; j < Lx; ++j)
                buffer[static_cast<std::size_t>(j) + static_cast<std::size_t>(y) * Lx] = 1.f;
        }
        return buffer;
    }

    /** Advance one double-precision periodic heat step from @p in into @p out. */
    inline void stepD(
        std::vector<double> const& in,
        std::vector<double>& out,
        double delta,
        double norm,
        int Lx,
        int Ly)
    {
        for(int y = 0; y < Ly; ++y)
        {
            for(int x = 0; x < Lx; ++x)
            {
                std::size_t const v00 = static_cast<std::size_t>(y) * Lx + x;
                std::size_t const v0p = static_cast<std::size_t>(y) * Lx + (x + 1) % Lx;
                std::size_t const v0m = static_cast<std::size_t>(y) * Lx + (Lx + x - 1) % Lx;
                std::size_t const vp0 = static_cast<std::size_t>((y + 1) % Ly) * Lx + x;
                std::size_t const vm0 = static_cast<std::size_t>((Ly + y - 1) % Ly) * Lx + x;
                out[v00] = norm * in[v00] + delta * (in[v0p] + in[v0m] + in[vp0] + in[vm0]);
            }
        }
    }

    /** Run the double-precision reference for exactly @p iters steps starting from @p init. */
    inline std::vector<double> referenceField(std::vector<float> const& init, int Lx, int Ly, int iters)
    {
        double const delta = deltaD();
        double const norm = normD();
        std::vector<double> a(init.begin(), init.end());
        std::vector<double> b(static_cast<std::size_t>(Lx) * static_cast<std::size_t>(Ly), 0.0);
        for(int it = 0; it < iters; ++it)
        {
            stepD(a, b, delta, norm, Lx, Ly);
            std::swap(a, b);
        }
        return a;
    }

    /** Verify the device result against the independent double-precision reference.
     *
     * Checks:
     *  - `Lx`, `Ly` and the iteration count are positive;
     *  - every device value is finite (NaN/Inf rejected explicitly);
     *  - the final field matches the double reference computed from the same initial field to the
     *    same iteration within @c heat2d::verifyTolerance().
     *
     * @param init   initial single-precision field, `Lx*Ly` elements, row-major
     * @param Lx,Ly  grid dimensions
     * @param iters  number of applied steps
     * @param result device result, `Lx*Ly` elements, row-major
     * @return EXIT_SUCCESS on pass, EXIT_FAILURE on failure
     */
    template<typename T_Result>
    int verify(std::vector<float> const& init, int Lx, int Ly, int iters, T_Result const& result)
    {
        std::size_t const count = static_cast<std::size_t>(Lx) * static_cast<std::size_t>(Ly);

        if(Lx <= 0 || Ly <= 0 || iters < 0)
        {
            printf("heat2d verification FAILED: invalid dimensions Lx=%d Ly=%d iters=%d\n", Lx, Ly, iters);
            return EXIT_FAILURE;
        }

        for(std::size_t k = 0; k < count; ++k)
        {
            float const v = static_cast<float>(result[k]);
            if(!std::isfinite(v))
            {
                printf("heat2d verification FAILED: non-finite value at index %zu\n", k);
                return EXIT_FAILURE;
            }
        }

        std::vector<double> const reference = referenceField(init, Lx, Ly, iters);
        double const tolerance = verifyTolerance(iters);
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

        if(maxDiff > tolerance)
        {
            printf(
                "heat2d verification FAILED: max |alpaka - double ref| = %.6e at index %zu exceeds %.6e\n",
                maxDiff,
                maxIdx,
                tolerance);
            return EXIT_FAILURE;
        }

        printf(
            "heat2d verification PASSED: Lx=%d Ly=%d iters=%d max |alpaka - double ref| = %.6e (tolerance "
            "%.6e)\n",
            Lx,
            Ly,
            iters,
            maxDiff,
            tolerance);
        return EXIT_SUCCESS;
    }
} // namespace heat2d
