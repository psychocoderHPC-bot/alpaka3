/* Copyright 2026 René Widera
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/** @file Independent host reference, deterministic input generation and verification for the `resize`
 * benchmark.
 *
 * The pinned HeCBench reference (`src/resize-cuda/main.cu`) resizes a stack of `num_channels`
 * single-channel images with either a nearest-neighbour (`resize`) or a bilinear (`resize_bilinear`)
 * kernel. Both kernels are reproduced here exactly, including behaviours that a "textbook" resize
 * would get wrong and that the port must preserve bit for bit:
 *
 *  1. The scaling ratios are computed with **integer division** before being converted to `float`:
 *     `const float fx = in_width / out_width;`. E.g. `1920/256 == 7` (not `7.5`). The coordinate
 *     mapping therefore samples the source on an integer-step grid, not on the mathematically exact
 *     ratio.
 *  2. `resize_image` passes the ratios to the kernels in the order `(fx, fy)` for the parameters
 *     `(o2i_fy, o2i_fx)`, i.e. the x and y ratios are **swapped**: the kernel's `o2i_fy` receives
 *     `in_width / out_width` and its `o2i_fx` receives `in_height / out_height`.
 *  3. The bilinear weights `T(in_y - in_y0)` and `T(in_x - in_x0)` are cast to the pixel type `T`.
 *     For the integer pixel types the benchmark instantiates (`unsigned char`, `unsigned short`,
 *     `unsigned int`) their fractional values truncate to `0`, so the bilinear kernel degenerates to
 *     `v_00` for the standard half-pixel-centers call. This host reference evaluates the identical
 *     expression in `T`, so it tracks the reference for integer and floating point pixel types alike.
 *  4. The kernels process `CHANNELS_PER_ITER = 8` channels per work item and compute
 *     `num_channel_iters_per_xy = num_channels / 8` with **integer division**. Channels
 *     `[ (num_channels/8)*8, num_channels )` are therefore never written and keep their initial
 *     (zero-initialised) value. This port reproduces that truncation.
 *
 * This file is host-only; it produces the deterministic input and the trusted result.
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <type_traits>
#include <vector>

namespace resizeBench
{
    //! Channels processed per work item by the HeCBench reference (`CHANNELS_PER_ITER`).
    constexpr std::uint32_t channelsPerIter = 8u;

    /** Number of channels the reference actually writes: `floor(numChannels / channelsPerIter)*channelsPerIter`. */
    inline std::uint32_t processedChannels(std::uint32_t numChannels)
    {
        return (numChannels / channelsPerIter) * channelsPerIter;
    }

    /** Deterministic source-image generator.
     *
     * The reference fills its input with `static_cast<T>((i + 1) % 13)` over the linear index; this
     * port reproduces that pattern exactly so the input is byte-identical to the reference.
     */
    template<typename T>
    inline void generateInput(T* data, std::size_t count)
    {
        for(std::size_t i = 0; i < count; ++i)
            data[i] = static_cast<T>((i + 1u) % 13u);
    }

    /** Per-pixel-type arithmetic mirror of the device bilinear expression.
     *
     * The update is evaluated in the pixel type `T` (with unsigned wrap on overflow for the integer
     * types). The interpolation weights are the float fractional parts `in_x - in_x0` and
     * `in_y - in_y0`, cast to `T` exactly as the reference does; for the integer pixel types they
     * truncate to `0`. For floating point `T` the cast is a no-op.
     */
    template<typename T>
    inline T bilinearValue(T v00, T v01, T v10, T v11, float fracX, float fracY)
    {
        T const wx = static_cast<T>(fracX);
        T const wy = static_cast<T>(fracY);
        return static_cast<T>(
            v00 + wy * static_cast<T>(v10 - v00) + wx * static_cast<T>(v01 - v00)
            + wy * wx * static_cast<T>(v11 - v01 - v10 + v00));
    }

    /** Nearest-neighbour coordinate: `lroundf((outIdx + 0.5f) * ratio)`, matching the reference. */
    inline int nearestCoord(int outIdx, float ratio)
    {
        return static_cast<int>(std::lroundf((static_cast<float>(outIdx) + 0.5f) * ratio));
    }

    /** Bilinear source coordinate `fmaxf((outIdx + 0.5f) * ratio - 0.5f, 0)`. */
    inline float bilinearCoord(int outIdx, float ratio)
    {
        return std::fmax((static_cast<float>(outIdx) + 0.5f) * ratio - 0.5f, 0.0f);
    }

    /** Bilinear lower coordinate (floor of the source coordinate). */
    inline int bilinearFloorCoord(int outIdx, float ratio)
    {
        return static_cast<int>(bilinearCoord(outIdx, ratio));
    }

    /** Independent host reference of the reference nearest-neighbour pipeline, evaluated in `T`.
     *
     * `o2i_fy == in_width/out_width` and `o2i_fx == in_height/out_height` (the reference's swapped
     * argument order); sample coordinates are clamped to `[0, size-1]`. Only the first
     * `processedChannels(numChannels)` channels are written; the caller must zero-initialise
     * `output`.
     */
    template<typename T>
    inline void resizeNearestGold(
        std::vector<T> const& input,
        std::vector<T>& output,
        std::uint32_t inWidth,
        std::uint32_t inHeight,
        std::uint32_t outWidth,
        std::uint32_t outHeight,
        std::uint32_t numChannels)
    {
        std::size_t const inImageSize = static_cast<std::size_t>(inWidth) * inHeight;
        std::size_t const outImageSize = static_cast<std::size_t>(outWidth) * outHeight;
        float const o2i_fy = static_cast<float>(inWidth / outWidth);
        float const o2i_fx = static_cast<float>(inHeight / outHeight);
        std::uint32_t const active = processedChannels(numChannels);

        for(std::uint32_t c = 0; c < active; ++c)
        {
            for(std::uint32_t y = 0; y < outHeight; ++y)
            {
                int inY = nearestCoord(static_cast<int>(y), o2i_fy);
                inY = std::clamp(inY, 0, static_cast<int>(inHeight) - 1);
                for(std::uint32_t x = 0; x < outWidth; ++x)
                {
                    int inX = nearestCoord(static_cast<int>(x), o2i_fx);
                    inX = std::clamp(inX, 0, static_cast<int>(inWidth) - 1);
                    output[c * outImageSize + static_cast<std::size_t>(y) * outWidth + x]
                        = input[c * inImageSize + static_cast<std::size_t>(inY) * inWidth + inX];
                }
            }
        }
    }

    /** Independent host reference of the reference bilinear pipeline, evaluated in `T`.
     *
     * Only the first `processedChannels(numChannels)` channels are written; the caller must
     * zero-initialise `output`.
     */
    template<typename T>
    inline void resizeBilinearGold(
        std::vector<T> const& input,
        std::vector<T>& output,
        std::uint32_t inWidth,
        std::uint32_t inHeight,
        std::uint32_t outWidth,
        std::uint32_t outHeight,
        std::uint32_t numChannels)
    {
        std::size_t const inImageSize = static_cast<std::size_t>(inWidth) * inHeight;
        std::size_t const outImageSize = static_cast<std::size_t>(outWidth) * outHeight;
        float const o2i_fy = static_cast<float>(inWidth / outWidth);
        float const o2i_fx = static_cast<float>(inHeight / outHeight);
        std::uint32_t const active = processedChannels(numChannels);

        for(std::uint32_t c = 0; c < active; ++c)
        {
            for(std::uint32_t y = 0; y < outHeight; ++y)
            {
                float const inYf = bilinearCoord(static_cast<int>(y), o2i_fy);
                int const inY0 = static_cast<int>(inYf);
                int const inY1 = std::clamp(inY0, 0, static_cast<int>(inHeight) - 1);
                int const inY2 = std::clamp(inY0 + 1, 0, static_cast<int>(inHeight) - 1);
                for(std::uint32_t x = 0; x < outWidth; ++x)
                {
                    float const inXf = bilinearCoord(static_cast<int>(x), o2i_fx);
                    int const inX0 = static_cast<int>(inXf);
                    int const inX1 = std::clamp(inX0 + 1, 0, static_cast<int>(inWidth) - 1);

                    std::size_t const base = c * inImageSize;
                    T const v00 = input[base + static_cast<std::size_t>(inY1) * inWidth + inX0];
                    T const v01 = input[base + static_cast<std::size_t>(inY1) * inWidth + inX1];
                    T const v10 = input[base + static_cast<std::size_t>(inY2) * inWidth + inX0];
                    T const v11 = input[base + static_cast<std::size_t>(inY2) * inWidth + inX1];

                    output[c * outImageSize + static_cast<std::size_t>(y) * outWidth + x]
                        = bilinearValue(v00, v01, v10, v11, inXf - static_cast<float>(inX0), inYf - static_cast<float>(inY0));
                }
            }
        }
    }

    /** Guard the datapath against non-finite and out-of-range results.
     *
     * The benchmark uses integer pixel types, so IEEE NaN/Inf cannot arise; this function makes that
     * explicit. For floating point pixel types it rejects non-finite input or output values.
     *
     * @return EXIT_SUCCESS when all checks pass, EXIT_FAILURE otherwise.
     */
    template<typename T>
    inline int checkSane(std::vector<T> const& input, std::vector<T> const& output, char const* label)
    {
        if constexpr(std::is_floating_point_v<T>)
        {
            for(std::size_t i = 0; i < input.size(); ++i)
            {
                if(!std::isfinite(static_cast<double>(input[i])))
                {
                    std::printf("error: non-finite input value at index %zu (%s)\n", i, label);
                    return EXIT_FAILURE;
                }
            }
            for(std::size_t i = 0; i < output.size(); ++i)
            {
                if(!std::isfinite(static_cast<double>(output[i])))
                {
                    std::printf("error: non-finite output value at index %zu (%s)\n", i, label);
                    return EXIT_FAILURE;
                }
            }
        }
        return EXIT_SUCCESS;
    }

    /** Compare the accelerator result against the host reference element by element (exact).
     *
     * For the integer pixel types the reference arithmetic is integer and the host reference is
     * computed in the same type, so the comparison is exact (tolerance 0). On the first mismatch the
     * linear index and both values are printed and `EXIT_FAILURE` is returned.
     */
    template<typename T>
    inline int verify(std::vector<T> const& computed, std::vector<T> const& reference, char const* label)
    {
        if(computed.size() != reference.size())
        {
            std::printf(
                "error: %s size mismatch computed=%zu reference=%zu\n",
                label,
                computed.size(),
                reference.size());
            return EXIT_FAILURE;
        }
        for(std::size_t i = 0; i < computed.size(); ++i)
        {
            if(computed[i] != reference[i])
            {
                std::printf(
                    "error: %s mismatch at index %zu expected=%lld got=%lld\n",
                    label,
                    i,
                    static_cast<long long>(reference[i]),
                    static_cast<long long>(computed[i]));
                return EXIT_FAILURE;
            }
        }
        return EXIT_SUCCESS;
    }

} // namespace resizeBench
