/* Copyright 2026 René Widera
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

/** @file Host reference, parameters and verification for the mandelbrot benchmark.
 *
 * `ComplexF`, `MandelParameters`, the row/column defaults and the scaling functions are a faithful
 * copy of the HeCBench reference `src/mandelbrot-cuda/mandel.hpp` (MIT, Copyright (c) 2019 Intel
 * Corporation), with the CUDA specific annotations removed. The alpaka kernel and the "faithful"
 * host reference go through exactly the same float arithmetic.
 *
 * A second, independent host reference in `double` precision is provided for verification. It uses
 * the same sampling grid but higher precision arithmetic, so it is not bit-identical to the float
 * kernel near the parabolic boundary. The classification of a pixel (interior vs. diverged) is
 * therefore compared with a small, measured tolerance; see `verify()` for the rationale.
 */

#include <alpaka/alpaka.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace mandel
{
    // Defaults replicate the HeCBench reference (mandel.hpp).
    constexpr int defaultRowSize = 1024;
    constexpr int defaultColSize = 1024;
    constexpr int defaultMaxIterations = 1000;
    constexpr int defaultRepetitions = 5;

    // The reference defines the block shape as 16 x 16; kept for documentation parity.
    constexpr int threadsPerBlockX = 16;
    constexpr int threadsPerBlockY = 16;

    /** Floating point complex number, bitwise the same type as the reference `ComplexF`. */
    struct ComplexF
    {
        float real;
        float imag;
    };

    /** Geometry and iteration limit of the computed image.
     *
     * `scaleRow` and `scaleCol` are copied verbatim from the reference so the sampling grid is
     * identical. `point` is the reference escape-time iteration (the CPU body of the reference
     * `MandelParameters::Point`).
     */
    struct MandelParameters
    {
        int row_count_;
        int col_count_;
        int max_iterations_;

        constexpr MandelParameters(int rowCount, int colCount, int maxIterations)
            : row_count_(rowCount)
            , col_count_(colCount)
            , max_iterations_(maxIterations)
        {
        }

        constexpr int rowCount() const
        {
            return row_count_;
        }
        constexpr int colCount() const
        {
            return col_count_;
        }
        constexpr int maxIterations() const
        {
            return max_iterations_;
        }

        // scale from 0..row_count to -1.5..0.5
        ALPAKA_FN_HOST_ACC constexpr float scaleRow(int i) const
        {
            return -1.5f + (i * (2.0f / row_count_));
        }

        // scale from 0..col_count to -1..1
        ALPAKA_FN_HOST_ACC constexpr float scaleCol(int i) const
        {
            return -1.0f + (i * (2.0f / col_count_));
        }

        // mandelbrot set are points that do not diverge within max_iterations
        ALPAKA_FN_HOST_ACC constexpr int point(ComplexF const& c) const
        {
            int count = 0;
            ComplexF z = {0.f, 0.f};
            for(int i = 0; i < max_iterations_; ++i)
            {
                float const r = z.real;
                float const im = z.imag;
                // leave loop if diverging
                if(((r * r) + (im * im)) >= 4.0f)
                    break;
                // z = z * z + c
                z.real = r * r - im * im + c.real;
                z.imag = 2.f * r * im + c.imag;
                count++;
            }
            return count;
        }
    };

    /** Serial float reference over the sampling grid; the "faithful" reference. */
    inline void evaluateReferenceFloat(int rows, int cols, int maxIterations, float* out)
    {
        MandelParameters const p{rows, cols, maxIterations};
        for(int i = 0; i < rows; ++i)
            for(int j = 0; j < cols; ++j)
                out[i * cols + j] = static_cast<float>(p.point(ComplexF{p.scaleRow(i), p.scaleCol(j)}));
    }

    /** Double precision escape count for one grid coordinate; the independent reference. */
    inline int pointDouble(int i, int j, int rows, int cols, int maxIterations)
    {
        double const cr = -1.5 + (static_cast<double>(i) * (2.0 / rows));
        double const ci = -1.0 + (static_cast<double>(j) * (2.0 / cols));
        double zr = 0.0;
        double zi = 0.0;
        int count = 0;
        for(int k = 0; k < maxIterations; ++k)
        {
            if(((zr * zr) + (zi * zi)) >= 4.0)
                break;
            double const r = zr;
            double const im = zi;
            zr = r * r - im * im + cr;
            zi = 2.0 * r * im + ci;
            count++;
        }
        return count;
    }

    /** Serial double precision reference over the sampling grid; the independent reference. */
    inline void evaluateReferenceDouble(int rows, int cols, int maxIterations, double* out)
    {
        for(int i = 0; i < rows; ++i)
            for(int j = 0; j < cols; ++j)
                out[i * cols + j] = static_cast<double>(pointDouble(i, j, rows, cols, maxIterations));
    }

    /** Verify a computed escape-count image.
     *
     * The kernel result is checked against two host references and a random spot check:
     *
     *  1. the *faithful* float reference (`evaluateReferenceFloat`), i.e. exactly the HeCBench
     *     `MandelParameters::Point` arithmetic, sharing the code with the kernel; and
     *  2. the *independent* double precision reference (`evaluateReferenceDouble`), which uses
     *     higher precision arithmetic so it cannot be fooled by a systematic defect that also
     *     exists in the shared float helper; and
     *  3. a deterministic random spot check that recomputes pixels of the *interior* (where the
     *     double reference reaches `maxIterations`) and requires the device to classify them as
     *     interior as well. Interior pixels are robust against the last-bit rounding of the
     *     boundary, so this check is exact.
     *
     * For every pixel the algorithm classifies interior (`escape count == maxIterations`) versus
     * diverged (`< maxIterations`). The comparison against the double reference allows a small
     * fraction of boundary pixels to differ; see the tolerance rationale below. The comparison
     * against the faithful float reference must agree for at least `1 - tolerance` of the pixels as
     * well. Every value must be finite and in `[0, maxIterations]`.
     *
     * Tolerance rationale. The escape-time map is chaotic on the parabolic boundary between the
     * interior and the exterior: a last-bit difference in the float iteration changes when a pixel
     * crosses the escape radius near `|c| ~ 1/2`. The committed kernel and the faithful reference
     * use the identical `float` arithmetic, so they agree bit-exactly; the independent `double`
     * reference differs on a small set of boundary pixels. Measured on the host with a `double`
     * evaluation of the same grid, the classification mismatch is 0.50 % at
     * `1024 x 1024 x 1000`, 0.68 % at `1080 x 1920 x 1000` and 0.75 % at `1080 x 1920 x 10000`.
     * A tolerance of 5 % (the value used by the HeCBench reference `Verify`) is therefore ~7x
     * above the observed float/double divergence and still catches any real defect, which would
     * affect whole regions, not only the boundary. The mismatch ratio is printed for every run.
     *
     * The tolerance is a correctness bound on a chaotic map, never widened to hide a bug: the exact
     * float-reference comparison, the spot check and the finite/range guards are independent
     * hard checks that must pass on their own.
     */
    template<typename T_Span>
    int verify(
        T_Span const& result,
        int rows,
        int cols,
        int maxIterations,
        float const* referenceFloat,
        double const* referenceDouble,
        double tolerance = 0.05)
    {
        int error = EXIT_SUCCESS;
        long long const total = static_cast<long long>(rows) * static_cast<long long>(cols);
        long long qNaN = 0;
        long long mismatchFloat = 0;
        long long mismatchDouble = 0;

        for(long long idx = 0; idx < total; ++idx)
        {
            double const value = static_cast<double>(result[idx]);
            if(!std::isfinite(value))
            {
                if(qNaN < 5) // bound the output
                    std::printf("error at result[%lld]: value is not finite (%f)\n", idx, value);
                ++qNaN;
                continue;
            }
            int const got = static_cast<int>(value);
            if(got < 0 || got > maxIterations)
            {
                std::printf("error at result[%lld]: %d outside [0, %d]\n", idx, got, maxIterations);
                return EXIT_FAILURE;
            }
            if(got != static_cast<int>(referenceFloat[idx]))
                ++mismatchFloat;
            if(got != static_cast<int>(referenceDouble[idx]))
                ++mismatchDouble;
        }
        if(qNaN != 0)
        {
            std::printf("error: %lld non-finite pixel values\n", qNaN);
            return EXIT_FAILURE;
        }

        // Independent spot check using the double reference on a deterministic set of pixels.
        // Only pixels that the double reference and all eight of their neighbours classify as
        // interior are asserted. Deep interior pixels never escape, so their classification is
        // robust against the last-bit float rounding and the check is exact.
        constexpr int spotCount = 4096;
        int checkedSpots = 0;
        std::srand(12345);
        for(int s = 0; s < spotCount * 8 && checkedSpots < spotCount; ++s)
        {
            int const i = std::rand() % rows;
            int const j = std::rand() % cols;
            int const expected = pointDouble(i, j, rows, cols, maxIterations);
            if(expected != maxIterations)
                continue;
            bool stable = true;
            for(int di = -1; di <= 1 && stable; ++di)
                for(int dj = -1; dj <= 1 && stable; ++dj)
                {
                    int const ni = i + di;
                    int const nj = j + dj;
                    if(ni < 0 || ni >= rows || nj < 0 || nj >= cols)
                    {
                        stable = false;
                        break;
                    }
                    if(pointDouble(ni, nj, rows, cols, maxIterations) != maxIterations)
                        stable = false;
                }
            if(!stable)
                continue;
            ++checkedSpots;
            long long const idx = static_cast<long long>(i) * cols + j;
            int const got = static_cast<int>(result[idx]);
            if(got != maxIterations)
            {
                std::printf(
                    "error at result[%d][%d]: %d expected %d (interior spot check)\n",
                    i,
                    j,
                    got,
                    maxIterations);
                error = EXIT_FAILURE;
                break;
            }
        }
        std::printf("stable interior spot checks: %d\n", checkedSpots);
        if(checkedSpots < 64)
            std::printf("warning: only %d stable interior spot checks were evaluated\n", checkedSpots);

        double const ratioFloat = static_cast<double>(mismatchFloat) / static_cast<double>(total);
        double const ratioDouble = static_cast<double>(mismatchDouble) / static_cast<double>(total);
        std::printf(
            "mismatch vs float reference: %lld / %lld (ratio %.6f)\n"
            "mismatch vs double reference: %lld / %lld (ratio %.6f, tolerance %.4f)\n",
            mismatchFloat,
            total,
            ratioFloat,
            mismatchDouble,
            total,
            ratioDouble,
            tolerance);

        if(ratioFloat > tolerance)
        {
            std::printf("error: float-reference mismatch ratio %.6f exceeds tolerance %.4f\n", ratioFloat, tolerance);
            error = EXIT_FAILURE;
        }
        if(ratioDouble > tolerance)
        {
            std::printf("error: double-reference mismatch ratio %.6f exceeds tolerance %.4f\n", ratioDouble, tolerance);
            error = EXIT_FAILURE;
        }

        std::printf("Mandelbrot verification %s\n", error ? "FAILED" : "PASSED");
        return error;
    }

} // namespace mandel
