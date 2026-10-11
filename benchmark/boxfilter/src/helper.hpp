/* Copyright 2026 René Widera
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

/** @file Host reference, deterministic input generation and verification for the `boxfilter` benchmark.
 *
 * The pinned HeCBench reference filters a PPM image with a separable 2D box blur of radius `RADIUS = 10`.
 * The horizontal pass writes a packed RGBA intermediate image, where every channel is the truncated
 * average of the horizontal window; the vertical pass then averages the intermediate image.
 *
 * Boundary semantics reproduced from the reference:
 *  - Horizontal: pixels outside `[0, width)` contribute zero (`uc4LocalData = {0,0,0,0}`).
 *  - Vertical: the reference's register sliding window is algebraically identical to clamping the
 *    window to `[0, height-1]` (replicate boundary) whenever `height >= 2*RADIUS + 1`; this is the
 *    regime the pinned benchmark uses (`height = 1024`). The port uses the clamp formulation, and
 *    verification compares only rows `[RADIUS, height-RADIUS)`, exactly the region the HeCBench
 *    harness checks (`start = RADIUS * width`, `end = size - start`).
 *
 * Quantization: the reference computes `(unsigned int)(sum * (1.0f/(2r+1)))` in single precision. For
 * the pinned radius the single-precision scale agrees with exact integer division `sum / (2r+1)` for
 * every achievable sum (`0..255*(2r+1)`), so this port uses the exact integer truncation, which is
 * deterministic and portable. See the README for the radius range where the two coincide.
 *
 * Nothing in this file is device code; it only produces the input and the trusted result.
 */

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace boxfilterBench
{
    //! 8-bit RGBA pixel, matching the reference `uchar4`.
    struct Uchar4
    {
        std::uint8_t x;
        std::uint8_t y;
        std::uint8_t z;
        std::uint8_t w;
    };

    /** Deterministic image generator (xorshift64*).
     *
     * The reference reads an external PPM image. To keep the benchmark self-contained the port
     * generates a deterministic 8-bit RGBA image; alpha is set to zero because `shrLoadPPM4ub` pads
     * the 4th channel with zero.
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

    //! Generate a deterministic 8-bit RGBA input image with zero alpha.
    inline void generateImage(Uchar4* image, std::size_t numPixels, std::uint64_t seed)
    {
        Rng rng{seed};
        for(std::size_t i = 0; i < numPixels; ++i)
        {
            std::uint32_t const word = rng.nextU32();
            image[i].x = static_cast<std::uint8_t>(word & 0xffu);
            image[i].y = static_cast<std::uint8_t>((word >> 8) & 0xffu);
            image[i].z = static_cast<std::uint8_t>((word >> 16) & 0xffu);
            image[i].w = 0u;
        }
    }

    /** Independent host reference of the reference pipeline.
     *
     * Horizontal pass: every row, exact integer channel sums over the zero-padded window
     * `[max(0,x-r), min(width-1,x+r)]`, written as `floor(sum / (2r+1))` per channel into `tmp`.
     * Vertical pass: every column, exact integer channel sums over the clamped window
     * `[max(0,y-r), min(height-1,y+r)]` of `tmp`, written as `floor(sum / (2r+1))` into `out`.
     */
    inline void boxFilterGold(
        Uchar4 const* input,
        std::size_t width,
        std::size_t height,
        int radius,
        std::uint32_t* tmp,
        std::uint32_t* out)
    {
        std::int64_t const divisor = 2 * static_cast<std::int64_t>(radius) + 1;

        for(std::size_t y = 0; y < height; ++y)
        {
            for(std::size_t x = 0; x < width; ++x)
            {
                std::int64_t s[4] = {0, 0, 0, 0};
                std::size_t const lo = (x > static_cast<std::size_t>(radius)) ? x - radius : 0u;
                std::size_t hi = x + static_cast<std::size_t>(radius);
                if(hi >= width)
                    hi = width - 1u;
                for(std::size_t xx = lo; xx <= hi; ++xx)
                {
                    Uchar4 const c = input[y * width + xx];
                    s[0] += c.x;
                    s[1] += c.y;
                    s[2] += c.z;
                    s[3] += c.w;
                }
                std::uint32_t packed = 0u;
                for(int ch = 0; ch < 4; ++ch)
                    packed |= (static_cast<std::uint32_t>(s[ch] / divisor) & 0xffu) << (8 * ch);
                tmp[y * width + x] = packed;
            }
        }

        for(std::size_t y = 0; y < height; ++y)
        {
            std::size_t const lo = (y > static_cast<std::size_t>(radius)) ? y - radius : 0u;
            std::size_t hi = y + static_cast<std::size_t>(radius);
            if(hi >= height)
                hi = height - 1u;
            for(std::size_t x = 0; x < width; ++x)
            {
                std::int64_t s[4] = {0, 0, 0, 0};
                for(std::size_t yy = lo; yy <= hi; ++yy)
                {
                    std::uint32_t const c = tmp[yy * width + x];
                    for(int ch = 0; ch < 4; ++ch)
                        s[ch] += (c >> (8 * ch)) & 0xffu;
                }
                std::uint32_t packed = 0u;
                for(int ch = 0; ch < 4; ++ch)
                    packed |= (static_cast<std::uint32_t>(s[ch] / divisor) & 0xffu) << (8 * ch);
                out[y * width + x] = packed;
            }
        }
    }

    /** Guard the integer datapath against non-finite and out-of-range intermediate values.
     *
     * The filter datapath is integer, so IEEE NaN/Inf cannot arise. This function makes that explicit:
     * it re-derives the horizontal channel sums in double and asserts each is finite, and it checks
     * that every packed channel fits in 8 bits (which is guaranteed by the `& 0xff` packing and the
     * accumulator width). It also validates the alpha invariant (generated input alpha is zero, so the
     * filtered alpha is zero).
     *
     * @return EXIT_SUCCESS when all sampled values are valid, EXIT_FAILURE otherwise.
     */
    inline int checkFinite(
        Uchar4 const* input,
        std::uint32_t const* output,
        std::size_t width,
        std::size_t height,
        int radius,
        char const* label)
    {
        std::size_t const r = static_cast<std::size_t>(radius);
        for(std::size_t y = 0; y < height; ++y)
        {
            for(std::size_t x = 0; x < width; ++x)
            {
                double s[4] = {0.0, 0.0, 0.0, 0.0};
                std::size_t const lo = (x > r) ? x - r : 0u;
                std::size_t hi = x + r;
                if(hi >= width)
                    hi = width - 1u;
                for(std::size_t xx = lo; xx <= hi; ++xx)
                {
                    Uchar4 const c = input[y * width + xx];
                    s[0] += c.x;
                    s[1] += c.y;
                    s[2] += c.z;
                    s[3] += c.w;
                }
                for(int ch = 0; ch < 4; ++ch)
                {
                    if(!std::isfinite(s[ch]))
                    {
                        std::printf("error: non-finite row sum at (y=%zu,x=%zu) ch=%d (%s)\n", y, x, ch, label);
                        return EXIT_FAILURE;
                    }
                }

                std::uint32_t const packed = output[y * width + x];
                if(((packed >> 24) & 0xffu) != 0u)
                {
                    std::printf("error: non-zero alpha at (y=%zu,x=%zu) (%s)\n", y, x, label);
                    return EXIT_FAILURE;
                }
            }
        }
        return EXIT_SUCCESS;
    }

    /** Compare the accelerator result against the host reference over the same region the HeCBench
     * harness checks: rows `[radius, height - radius)`, every column.
     *
     * Both the reference and the port truncate exact integer sums, so the comparison is exact
     * (tolerance 0). On the first mismatch the pixel coordinates and both packed values are printed and
     * `EXIT_FAILURE` is returned. A nonzero exit code signals failure.
     */
    inline int verify(
        std::uint32_t const* computed,
        std::uint32_t const* reference,
        std::size_t width,
        std::size_t height,
        int radius,
        char const* label)
    {
        std::size_t const r = static_cast<std::size_t>(radius);
        if(height <= 2u * r)
        {
            std::printf("error: %s height %zu too small for radius %d (empty ROI)\n", label, height, radius);
            return EXIT_FAILURE;
        }

        for(std::size_t y = r; y < height - r; ++y)
        {
            for(std::size_t x = 0; x < width; ++x)
            {
                std::size_t const idx = y * width + x;
                if(computed[idx] != reference[idx])
                {
                    std::printf(
                        "error: %s mismatch at (y=%zu, x=%zu) expected=%08x got=%08x\n",
                        label,
                        y,
                        x,
                        reference[idx],
                        computed[idx]);
                    return EXIT_FAILURE;
                }
            }
        }
        return EXIT_SUCCESS;
    }

} // namespace boxfilterBench
