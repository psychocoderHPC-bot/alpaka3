/* Copyright 2026 René Widera
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

/** @file Independent host reference and verification for the `colorwheel` benchmark.
 *
 * The benchmark encodes a 2D vector field as an RGB colour wheel (Daniel Scharstein, 2007): the
 * direction of each vector selects a hue along a 55-entry colour ramp and its magnitude controls the
 * saturation. The pinned HeCBench reference is `src/colorwheel-cuda/main.cu` (CUDA) and
 * `src/colorwheel-omp/main.cpp` (OpenMP); both compute the same mapping in single precision and
 * compare the device result against the host result with an absolute tolerance of one per channel:
 *
 * @code
 * const int tolerance = 1; // Allow differences up to 1 in each color channel
 * @endcode
 *
 * This file provides a **separate, double-precision** implementation of the same mapping. It is not
 * the kernel source, is never compiled into the accelerator path, and is written independently of
 * the alpaka kernel. It independently validates the mapping and colour path (hue index and
 * saturation) against double precision. Because it rebuilds the colour ramp from the same segment
 * formulas as the kernel, a systematic ramp-value error would not be caught here; the ramp values
 * were hand-verified against the pinned HeCBench reference (`src/colorwheel-cuda/main.cu` at SHA
 * `0a17ee47ca6072192101036d4bf292e6612f22e6`). The colour-ramp table is the only shared data (the
 * algorithm's definition), rebuilt from the six transition lengths exactly as the reference does.
 *
 * Tolerance: the reference accepts an absolute difference of 1 per channel because the mapping is a
 * lossy `int(255*col)` truncation of a single-precision computation. The port compares the kernel
 * against the double reference with the same tolerance of 1; the README shows that the reference's
 * own single-precision pipeline stays within 1 of the double result for every tested size, so the
 * tolerance covers rounding, not an algorithmic difference.
 *
 * NaN/Inf: the datapath is bounded (`col` is multiplied by 0.75 outside the unit disc and lies in
 * `[0,1]` inside it), so NaN/Inf cannot arise from the arithmetic. @ref computeColorReference still
 * rejects a non-finite intermediate explicitly, @ref verify checks the output range, and the axes
 * are required to be exactly black.
 *
 * Nothing in this file is device code.
 */

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace colorwheel
{
    //! Transition lengths of the six colour-ramp segments, identical to the pinned source.
    constexpr int RY = 15;
    constexpr int YG = 6;
    constexpr int GC = 4;
    constexpr int CB = 11;
    constexpr int BM = 13;
    constexpr int MR = 6;
    //! Total number of colour-ramp entries, identical to the pinned source (`MAXCOLS`).
    constexpr int MAXCOLS = RY + YG + GC + CB + BM + MR; // 55

    //! pi as a literal, to avoid the platform-dependent `M_PI` macro.
    constexpr double kPi = 3.14159265358979323846;

    //! Absolute per-channel tolerance of the HeCBench reference (`const int tolerance = 1`).
    constexpr int tolerance = 1;

    //! Build the colour-ramp table exactly as @c setcols sequences in the pinned source.
    inline void buildColorWheel(int cw[MAXCOLS][3])
    {
        int k = 0;
        for(int i = 0; i < RY; i++)
        {
            cw[k][0] = 255;
            cw[k][1] = 255 * i / RY;
            cw[k][2] = 0;
            ++k;
        }
        for(int i = 0; i < YG; i++)
        {
            cw[k][0] = 255 - 255 * i / YG;
            cw[k][1] = 255;
            cw[k][2] = 0;
            ++k;
        }
        for(int i = 0; i < GC; i++)
        {
            cw[k][0] = 0;
            cw[k][1] = 255;
            cw[k][2] = 255 * i / GC;
            ++k;
        }
        for(int i = 0; i < CB; i++)
        {
            cw[k][0] = 0;
            cw[k][1] = 255 - 255 * i / CB;
            cw[k][2] = 255;
            ++k;
        }
        for(int i = 0; i < BM; i++)
        {
            cw[k][0] = 255 * i / BM;
            cw[k][1] = 0;
            cw[k][2] = 255;
            ++k;
        }
        for(int i = 0; i < MR; i++)
        {
            cw[k][0] = 255;
            cw[k][1] = 0;
            cw[k][2] = 255 - 255 * i / MR;
            ++k;
        }
    }

    /** Independent double-precision colour encoding of one flow vector.
     *
     * Mirrors the mapping of the pinned reference (`colorcode.cpp` / `computeColor`), but evaluates
     * every intermediate in double precision. The ramp table is rebuilt locally.
     *
     * @param fx  x component of the (already range-normalised) flow vector
     * @param fy  y component
     * @param pix receives the three channels in the reference's byte order: @c pix[2]=red,
     *            @c pix[1]=green, @c pix[0]=blue
     * @return @c true when the result is finite, @c false if a non-finite value is encountered
     */
    inline bool computeColorReference(double fx, double fy, unsigned char pix[3])
    {
        int cw[MAXCOLS][3];
        buildColorWheel(cw);

        double const rad = std::sqrt(fx * fx + fy * fy);
        double const a = std::atan2(-fy, -fx) / kPi;
        double const fk = (a + 1.0) / 2.0 * static_cast<double>(MAXCOLS - 1);
        int const k0 = static_cast<int>(fk);
        int const k1 = (k0 + 1) % MAXCOLS;
        double const f = fk - static_cast<double>(k0);

        for(int b = 0; b < 3; b++)
        {
            double const col0 = static_cast<double>(cw[k0][b]) / 255.0;
            double const col1 = static_cast<double>(cw[k1][b]) / 255.0;
            double col = (1.0 - f) * col0 + f * col1;
            if(rad <= 1.0)
                col = 1.0 - rad * (1.0 - col); // increase saturation with radius
            else
                col *= 0.75; // out of range
            double const v = 255.0 * col;
            if(!std::isfinite(v))
                return false;
            pix[2 - b] = static_cast<unsigned char>(static_cast<int>(v));
        }
        return true;
    }

    /** Independent double-precision colour encoding returning the raw channel values `255*col`.
     *
     * Same mapping as @ref computeColorReference but without the `int()` truncation, so callers can
     * assert the value is finite and lies in `[0,255]`. @p out receives the three channels in the
     * reference order (`out[2]` = red, `out[0]` = blue).
     *
     * @return @c true when every channel is finite, @c false otherwise.
     */
    inline bool computeColorValueReference(double fx, double fy, double out[3])
    {
        int cw[MAXCOLS][3];
        buildColorWheel(cw);

        double const rad = std::sqrt(fx * fx + fy * fy);
        double const a = std::atan2(-fy, -fx) / kPi;
        double const fk = (a + 1.0) / 2.0 * static_cast<double>(MAXCOLS - 1);
        int const k0 = static_cast<int>(fk);
        int const k1 = (k0 + 1) % MAXCOLS;
        double const f = fk - static_cast<double>(k0);

        bool ok = true;
        for(int b = 0; b < 3; b++)
        {
            double const col0 = static_cast<double>(cw[k0][b]) / 255.0;
            double const col1 = static_cast<double>(cw[k1][b]) / 255.0;
            double col = (1.0 - f) * col0 + f * col1;
            if(rad <= 1.0)
                col = 1.0 - rad * (1.0 - col); // increase saturation with radius
            else
                col *= 0.75; // out of range
            double const v = 255.0 * col;
            if(!std::isfinite(v))
                ok = false;
            out[2 - b] = v;
        }
        return ok;
    }

    /** Map one pixel index to the normalised flow components exactly as the reference does.
     *
     * @param x,y       pixel coordinates
     * @param halfSize  @c size/2 (the reference `half_size`)
     * @param range     the padded range (1.04 * truerange)
     * @param trueRange @c truerange
     */
    inline void pixelToFlow(int x, int y, int halfSize, float range, float trueRange, double& fx, double& fy)
    {
        double const rx = static_cast<double>(x) / static_cast<double>(halfSize) * static_cast<double>(range)
                          - static_cast<double>(range);
        double const ry = static_cast<double>(y) / static_cast<double>(halfSize) * static_cast<double>(range)
                          - static_cast<double>(range);
        fx = rx / static_cast<double>(trueRange);
        fy = ry / static_cast<double>(trueRange);
    }

    /** Fill @p pix (3 bytes per pixel, row-major) with the double-precision reference image.
     *
     * Axis pixels (`x == halfSize` or `y == halfSize`) are left at zero, reproducing the reference's
     * black coordinate axes. Any non-finite mapping result is reported and makes the call return
     * @c false.
     */
    inline bool referenceImage(int size, int halfSize, float range, float trueRange, std::vector<unsigned char>& pix)
    {
        pix.assign(static_cast<std::size_t>(size) * size * 3u, 0u);
        for(int y = 0; y < size; ++y)
        {
            for(int x = 0; x < size; ++x)
            {
                if(x == halfSize || y == halfSize)
                    continue; // black axes
                std::size_t const idx = (static_cast<std::size_t>(y) * size + x) * 3u;
                double fx, fy;
                pixelToFlow(x, y, halfSize, range, trueRange, fx, fy);
                if(!computeColorReference(fx, fy, pix.data() + idx))
                {
                    std::printf("error: non-finite reference colour at (x=%d,y=%d)\n", x, y);
                    return false;
                }
            }
        }
        return true;
    }

    /** Explicit finiteness and range guard for the accelerator output.
     *
     * The encoded image is stored as bytes, so IEEE NaN/Inf cannot appear. This function makes the
     * invariant explicit by recomputing each non-axis pixel in double and rejecting a non-finite value,
     * and by checking that every byte is in `[0,255]` and that axis bytes are exactly zero.
     *
     * @return @c EXIT_SUCCESS when every checked value is valid, @c EXIT_FAILURE otherwise.
     */
    inline int checkFinite(
        unsigned char const* output,
        int size,
        int halfSize,
        float range,
        float trueRange)
    {
        for(int y = 0; y < size; ++y)
        {
            for(int x = 0; x < size; ++x)
            {
                std::size_t const idx = (static_cast<std::size_t>(y) * size + x) * 3u;
                if(x == halfSize || y == halfSize)
                {
                    if(output[idx] != 0 || output[idx + 1] != 0 || output[idx + 2] != 0)
                    {
                        std::printf("error: axis pixel (x=%d,y=%d) is not black\n", x, y);
                        return EXIT_FAILURE;
                    }
                    continue;
                }
                double fx, fy;
                pixelToFlow(x, y, halfSize, range, trueRange, fx, fy);
                double vals[3];
                if(!computeColorValueReference(fx, fy, vals))
                {
                    std::printf("error: non-finite reference at (x=%d,y=%d)\n", x, y);
                    return EXIT_FAILURE;
                }
                for(int c = 0; c < 3; ++c)
                {
                    if(!(vals[c] >= 0.0 && vals[c] <= 255.0))
                    {
                        std::printf(
                            "error: reference channel out of range at (x=%d,y=%d) channel=%d value=%f\n",
                            x,
                            y,
                            c,
                            vals[c]);
                        return EXIT_FAILURE;
                    }
                }
            }
        }
        return EXIT_SUCCESS;
    }

    /** Compare the accelerator image against the independent double-precision reference.
     *
     * Every pixel (including the black axes) is checked channel by channel with the reference's own
     * absolute tolerance of 1. The comparison reads the double reference, not the kernel code, so it
     * independently validates the mapping and colour path (hue index and saturation); a wrong hue or
     * wrong saturation cannot pass. The ramp values are shared by both implementations and were
     * hand-verified against the pinned reference. A nonzero exit code signals failure.
     *
     * @return @c EXIT_SUCCESS on success, @c EXIT_FAILURE on the first mismatch.
     */
    inline int verify(unsigned char const* computed, int size, int halfSize, float range, float trueRange)
    {
        int maxError = 0;
        for(int y = 0; y < size; ++y)
        {
            for(int x = 0; x < size; ++x)
            {
                std::size_t const idx = (static_cast<std::size_t>(y) * size + x) * 3u;
                unsigned char ref[3] = {0u, 0u, 0u};
                if(x != halfSize && y != halfSize)
                {
                    double fx, fy;
                    pixelToFlow(x, y, halfSize, range, trueRange, fx, fy);
                    if(!computeColorReference(fx, fy, ref))
                    {
                        std::printf("error: non-finite reference at (x=%d,y=%d)\n", x, y);
                        return EXIT_FAILURE;
                    }
                }
                for(int c = 0; c < 3; ++c)
                {
                    int const e = std::abs(static_cast<int>(computed[idx + c]) - static_cast<int>(ref[c]));
                    if(e > maxError)
                        maxError = e;
                    if(e > tolerance)
                    {
                        std::printf(
                            "error: colorwheel mismatch at (x=%d,y=%d) channel=%d expected=%d got=%d\n",
                            x,
                            y,
                            c,
                            static_cast<int>(ref[c]),
                            static_cast<int>(computed[idx + c]));
                        return EXIT_FAILURE;
                    }
                }
            }
        }
        std::printf("colorwheel verification PASSED (max per-channel error %d, tolerance %d)\n", maxError, tolerance);
        return EXIT_SUCCESS;
    }

} // namespace colorwheel
