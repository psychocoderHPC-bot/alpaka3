/* Copyright 2026 René Widera
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

/** @file Host reference, deterministic input generation and verification for the HeCBench `affine` benchmark.
 *
 * The pinned HeCBench reference (`src/affine-cuda/kernel.h`, shared through `reference.h` by the CUDA,
 * HIP, OpenMP and SYCL variants) applies a fixed 2D affine image transform to a grayscale 16-bit
 * image. The forward transform is a rotation of 30 degrees, an expansion of 0.5 in both axes and a
 * translation of zero:
 *
 *   affine = [[0.5*cos(30deg), 0.5*sin(0deg)],
 *             [0.5*sin(30deg), 0.5*cos(0deg)]]   beta = [0, 0]
 *
 * Each output pixel is produced by the inverse transform and a bilinear sample of the source:
 *  - `x_new`/`y_new` are the source coordinates of output pixel `(x, y)`,
 *  - `m = floor(x_new)`, `n = floor(y_new)` and `x_frac`/`y_frac` are the fractional parts,
 *  - when `m` and `n` are strictly interior the result is the bilinear interpolation of the four
 *    source samples, truncated to `unsigned short`,
 *  - when the sample falls exactly on the last row or column the nearest edge sample is used,
 *  - otherwise the pixel is filled with `WHITE = 1`.
 *
 * The reference computes the coefficients and everything else in single precision (`float`), and the
 * harness compares the device result against the same host reference, printing the maximum absolute
 * error over the image. The pinned `reference.h` hard-codes `X_SIZE`/`Y_SIZE`; this port generalises
 * them to runtime `width`/`height` without changing the arithmetic.
 *
 * This file provides:
 *  - `affineReference`: a host translation of the pinned reference algorithm in single precision,
 *    the trusted result the accelerator output is compared against.
 *  - `affineReferenceDouble`: the mathematically equivalent transform evaluated in double precision,
 *    used to justify the tolerance and to cross-check the single-precision reference.
 *  - `verify`: a tolerance comparison that reports the number of differing pixels, the maximum
 *    absolute difference and the location of the maximum; it fails when the maximum exceeds the
 *    justified tolerance (default 1 grey level).
 *  - `checkFinite`: an explicit NaN/Inf and range check of the input samples and the double-precision
 *    transform intermediates.
 *
 * Nothing in this file is device code; it only produces the input and the trusted result.
 */

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace affineBench
{
    //! Reference constant `PI` from the pinned `reference.h`.
    constexpr float pi = 3.14159265359f;
    //! Reference fill value for out-of-domain pixels.
    constexpr unsigned short white = 1;
    //! Rotation angles in degrees (pinned reference values). The reference uses `lx_rot` for the
    //! first column of the forward matrix and `ly_rot` for the second.
    constexpr float xRotationDeg = 30.0f;
    constexpr float yRotationDeg = 0.0f;
    //! Expansion factors (pinned reference values).
    constexpr float xExpand = 0.5f;
    constexpr float yExpand = 0.5f;

    /** Deterministic grayscale 16-bit image generator (xorshift64*).
     *
     * The reference reads an external RAW image (`CT-MONO2-16-brain.raw`, 512x512 unsigned short).
     * To keep the benchmark self-contained the port generates a deterministic image in the same
     * format. Samples are masked to 12 bits (`[0, 4095]`), the dynamic range of the pinned CT input.
     */
    class Rng
    {
    public:
        explicit Rng(std::uint64_t seed) : m_state(seed | 1u)
        {
        }

        std::uint32_t nextU32()
        {
            std::uint64_t x = m_state;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            m_state = x;
            return static_cast<std::uint32_t>((x * 0x2545F4914F6CDD1Du) >> 32);
        }

    private:
        std::uint64_t m_state;
    };

    //! Generate a deterministic 16-bit grayscale image with samples in `[0, 4095]`.
    inline void generateImage(unsigned short* image, std::size_t numPixels, std::uint64_t seed)
    {
        Rng rng{seed};
        for(std::size_t i = 0; i < numPixels; ++i)
            image[i] = static_cast<unsigned short>(rng.nextU32() & 0xfffu);
    }

    /** Single-precision host reference, a direct translation of the pinned `reference.h`.
     *
     * The arithmetic, branch conditions and truncation match the pinned source. `width`/`height`
     * replace its compile-time `X_SIZE`/`Y_SIZE`.
     */
    inline void affineReference(
        unsigned short const* src,
        unsigned short* dst,
        int width,
        int height)
    {
        for(int y = 0; y < height; ++y)
        {
            for(int x = 0; x < width; ++x)
            {
                const float lxRot = xRotationDeg;
                const float lyRot = yRotationDeg;
                const float lxExpan = xExpand;
                const float lyExpan = yExpand;
                int const lxMove = 0;
                int const lyMove = 0;
                float affine[2][2];
                float iAffine[2][2];
                float beta[2];
                float iBeta[2];
                float det;
                float xNew, yNew;
                float xFrac, yFrac;
                float grayNew;
                int m, n;

                affine[0][0] = lxExpan * std::cos((float)(lxRot * pi / 180.0f));
                affine[0][1] = lyExpan * std::sin((float)(lyRot * pi / 180.0f));
                affine[1][0] = lxExpan * std::sin((float)(lxRot * pi / 180.0f));
                affine[1][1] = lyExpan * std::cos((float)(lyRot * pi / 180.0f));
                beta[0] = static_cast<float>(lxMove);
                beta[1] = static_cast<float>(lyMove);

                det = (affine[0][0] * affine[1][1]) - (affine[0][1] * affine[1][0]);
                if(det == 0.0f)
                {
                    iAffine[0][0] = 1.0f;
                    iAffine[0][1] = 0.0f;
                    iAffine[1][0] = 0.0f;
                    iAffine[1][1] = 1.0f;
                    iBeta[0] = -beta[0];
                    iBeta[1] = -beta[1];
                }
                else
                {
                    iAffine[0][0] = affine[1][1] / det;
                    iAffine[0][1] = -affine[0][1] / det;
                    iAffine[1][0] = -affine[1][0] / det;
                    iAffine[1][1] = affine[0][0] / det;
                    iBeta[0] = -iAffine[0][0] * beta[0] - iAffine[0][1] * beta[1];
                    iBeta[1] = -iAffine[1][0] * beta[0] - iAffine[1][1] * beta[1];
                }

                xNew = iBeta[0] + iAffine[0][0] * (static_cast<float>(x) - width / 2.0f)
                       + iAffine[0][1] * (static_cast<float>(y) - height / 2.0f) + width / 2.0f;
                yNew = iBeta[1] + iAffine[1][0] * (static_cast<float>(x) - width / 2.0f)
                       + iAffine[1][1] * (static_cast<float>(y) - height / 2.0f) + height / 2.0f;

                m = (int)std::floor(xNew);
                n = (int)std::floor(yNew);

                xFrac = xNew - m;
                yFrac = yNew - n;

                unsigned short outputBuffer;
                if((m >= 0) && (m + 1 < width) && (n >= 0) && (n + 1 < height))
                {
                    grayNew = (1.0f - yFrac)
                                  * ((1.0f - xFrac) * (src[(n * width) + m])
                                     + xFrac * (src[(n * width) + m + 1]))
                              + yFrac
                                    * ((1.0f - xFrac) * (src[((n + 1) * width) + m])
                                       + xFrac * (src[((n + 1) * width) + m + 1]));
                    outputBuffer = (unsigned short)grayNew;
                }
                else if(
                    ((m + 1 == width) && (n >= 0) && (n < height))
                    || ((n + 1 == height) && (m >= 0) && (m < width)))
                {
                    outputBuffer = src[(n * width) + m];
                }
                else
                {
                    outputBuffer = white;
                }

                dst[(y * width) + x] = outputBuffer;
            }
        }
    }

    /** Double-precision independent oracle of the same transform.
     *
     * Same geometry, branch conditions and truncation as `affineReference`, but every intermediate is
     * computed in double precision. It differs from the single-precision reference only by the
     * rounding of a truncated bilinear sample; that difference is what justifies a tolerance of one
     * grey level (see `verify` and the README).
     */
    inline void affineReferenceDouble(
        unsigned short const* src,
        unsigned short* dst,
        int width,
        int height)
    {
        double const piD = static_cast<double>(pi);
        double const xExpandD = static_cast<double>(xExpand);
        double const yExpandD = static_cast<double>(yExpand);
        double const xRotD = static_cast<double>(xRotationDeg);
        double const yRotD = static_cast<double>(yRotationDeg);

        double const cx = std::cos(xRotD * piD / 180.0);
        double const sx = std::sin(xRotD * piD / 180.0);
        double const cy = std::cos(yRotD * piD / 180.0);
        double const sy = std::sin(yRotD * piD / 180.0);
        double const affine[2][2] = {{xExpandD * cx, yExpandD * sy}, {xExpandD * sx, yExpandD * cy}};
        double const beta[2] = {0.0, 0.0};
        double const det = affine[0][0] * affine[1][1] - affine[0][1] * affine[1][0];
        double iAffine[2][2];
        double iBeta[2];
        if(det == 0.0)
        {
            iAffine[0][0] = 1.0;
            iAffine[0][1] = 0.0;
            iAffine[1][0] = 0.0;
            iAffine[1][1] = 1.0;
            iBeta[0] = -beta[0];
            iBeta[1] = -beta[1];
        }
        else
        {
            iAffine[0][0] = affine[1][1] / det;
            iAffine[0][1] = -affine[0][1] / det;
            iAffine[1][0] = -affine[1][0] / det;
            iAffine[1][1] = affine[0][0] / det;
            iBeta[0] = -iAffine[0][0] * beta[0] - iAffine[0][1] * beta[1];
            iBeta[1] = -iAffine[1][0] * beta[0] - iAffine[1][1] * beta[1];
        }

        for(int y = 0; y < height; ++y)
        {
            for(int x = 0; x < width; ++x)
            {
                double const xNew = iBeta[0] + iAffine[0][0] * (x - width / 2.0)
                                    + iAffine[0][1] * (y - height / 2.0) + width / 2.0;
                double const yNew = iBeta[1] + iAffine[1][0] * (x - width / 2.0)
                                    + iAffine[1][1] * (y - height / 2.0) + height / 2.0;

                int const m = (int)std::floor(xNew);
                int const n = (int)std::floor(yNew);
                double const xFrac = xNew - m;
                double const yFrac = yNew - n;

                unsigned short outputBuffer;
                if((m >= 0) && (m + 1 < width) && (n >= 0) && (n + 1 < height))
                {
                    double const grayNew = (1.0 - yFrac)
                                               * ((1.0 - xFrac) * src[(n * width) + m]
                                                  + xFrac * src[(n * width) + m + 1])
                                           + yFrac
                                                 * ((1.0 - xFrac) * src[((n + 1) * width) + m]
                                                    + xFrac * src[((n + 1) * width) + m + 1]);
                    outputBuffer = (unsigned short)grayNew;
                }
                else if(
                    ((m + 1 == width) && (n >= 0) && (n < height))
                    || ((n + 1 == height) && (m >= 0) && (m < width)))
                {
                    outputBuffer = src[(n * width) + m];
                }
                else
                {
                    outputBuffer = white;
                }

                dst[(y * width) + x] = outputBuffer;
            }
        }
    }

    /** Explicit NaN/Inf and range guard.
     *
     * The reference datapath is a bilinear interpolation of 16-bit samples followed by a truncation to
     * `unsigned short`. Finite inputs cannot produce NaN or Inf, and the output is inherently in
     * `[0, 65535]`. This function makes that explicit: it re-derives every source sample and the
     * double-precision transform intermediates, asserts they are finite, asserts the samples are
     * inside `[0, 65535]`, and asserts the output values are inside `[0, 65535]`.
     *
     * @return EXIT_SUCCESS when everything is valid, EXIT_FAILURE otherwise.
     */
    inline int checkFinite(
        unsigned short const* src,
        unsigned short const* output,
        int width,
        int height,
        char const* label)
    {
        for(std::size_t i = 0; i < static_cast<std::size_t>(width) * static_cast<std::size_t>(height); ++i)
        {
            double const v = static_cast<double>(src[i]);
            if(!std::isfinite(v) || v < 0.0 || v > 65535.0)
            {
                std::printf("error: invalid source sample at %zu (%s)\n", i, label);
                return EXIT_FAILURE;
            }
        }

        // Recompute the double-precision transform for every pixel and assert the intermediates are
        // finite. The transform of finite input is finite; this is an explicit guard, not a heuristic.
        double const piD = static_cast<double>(pi);
        double const xRotD = static_cast<double>(xRotationDeg);
        double const yRotD = static_cast<double>(yRotationDeg);
        double const cx = std::cos(xRotD * piD / 180.0);
        double const sx = std::sin(xRotD * piD / 180.0);
        double const cy = std::cos(yRotD * piD / 180.0);
        double const sy = std::sin(yRotD * piD / 180.0);
        double const a00 = xExpand * cx, a01 = yExpand * sy, a10 = xExpand * sx, a11 = yExpand * cy;
        double const det = a00 * a11 - a01 * a10;
        double i00, i01, i10, i11;
        if(det == 0.0)
        {
            i00 = 1.0;
            i01 = 0.0;
            i10 = 0.0;
            i11 = 1.0;
        }
        else
        {
            i00 = a11 / det;
            i01 = -a01 / det;
            i10 = -a10 / det;
            i11 = a00 / det;
        }

        for(int y = 0; y < height; ++y)
        {
            for(int x = 0; x < width; ++x)
            {
                double const xNew = i00 * (x - width / 2.0) + i01 * (y - height / 2.0) + width / 2.0;
                double const yNew = i10 * (x - width / 2.0) + i11 * (y - height / 2.0) + height / 2.0;
                if(!std::isfinite(xNew) || !std::isfinite(yNew))
                {
                    std::printf("error: non-finite transform at (y=%d,x=%d) (%s)\n", y, x, label);
                    return EXIT_FAILURE;
                }
                double const out = static_cast<double>(output[static_cast<std::size_t>(y) * width + x]);
                if(!std::isfinite(out) || out < 0.0 || out > 65535.0)
                {
                    std::printf("error: invalid output at (y=%d,x=%d) (%s)\n", y, x, label);
                    return EXIT_FAILURE;
                }
            }
        }
        return EXIT_SUCCESS;
    }

    /** Tolerance comparison of the accelerator output against the single-precision host reference.
     *
     * The result is a truncated integer bilinear sample of a single-precision transform.
     * A device and a host evaluation of the same formula can round the transform (in particular the
     * last digit of `cos`/`sin` and the interpolation) slightly differently; that only changes the
     * truncated integer when the interpolated value is within the rounding error of an integer. The
     * justified tolerance is therefore one grey level (`tolerance = 1`), never more: a correctly
     * implemented port can differ from the host reference by at most one unit, and only for samples
     * whose exact value is essentially integer. Any larger difference is a real defect.
     *
     * Reports the number of differing pixels, the maximum absolute difference and the first location
     * of that maximum. Returns EXIT_FAILURE when the maximum exceeds `tolerance`.
     */
    inline int verify(
        unsigned short const* computed,
        unsigned short const* reference,
        int width,
        int height,
        char const* label,
        int tolerance = 1)
    {
        std::size_t numDiff = 0;
        int maxDelta = 0;
        int maxX = -1;
        int maxY = -1;
        for(int y = 0; y < height; ++y)
        {
            for(int x = 0; x < width; ++x)
            {
                std::size_t const idx = static_cast<std::size_t>(y) * width + x;
                int const a = static_cast<int>(computed[idx]);
                int const b = static_cast<int>(reference[idx]);
                int const delta = (a > b) ? a - b : b - a;
                if(delta != 0)
                    ++numDiff;
                if(delta > maxDelta)
                {
                    maxDelta = delta;
                    maxX = x;
                    maxY = y;
                }
            }
        }

        std::printf(
            "%s: %zu/%d pixels differ, max delta %d at (y=%d,x=%d), tolerance %d\n",
            label,
            numDiff,
            width * height,
            maxDelta,
            maxY,
            maxX,
            tolerance);

        if(maxDelta > tolerance)
        {
            std::printf(
                "error: %s max delta %d exceeds tolerance %d\n",
                label,
                maxDelta,
                tolerance);
            return EXIT_FAILURE;
        }
        return EXIT_SUCCESS;
    }

    /** Cross-check the single-precision reference against the double-precision oracle.
     *
     * The two references must agree within one grey level (the same rounding argument as `verify`).
     * Any larger difference means one of the references is wrong. Reports the differing count and the
     * maximum delta and returns EXIT_FAILURE when the bound is exceeded.
     */
    inline int verifyReferenceAgainstDouble(
        unsigned short const* singleReference,
        unsigned short const* doubleReference,
        int width,
        int height,
        char const* label,
        int tolerance = 1)
    {
        return verify(singleReference, doubleReference, width, height, label, tolerance);
    }

} // namespace affineBench
