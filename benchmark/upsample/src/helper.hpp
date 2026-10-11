/* Copyright 2026 René Widera
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

/** @file Deterministic input, independent host references and verification for the `upsample`
 * benchmark.
 *
 * The pinned HeCBench reference (`src/upsample-cuda/main.cu`, OpenMP twin `src/upsample-omp/main.cpp`)
 * implements the 2x nearest-neighbour upsampling layer of a small U-Net. It contains two passes with
 * two kernel formulations each, but all four kernels compute the same two mathematical operations on
 * an `N, C, H, W` tensor (row-major):
 *
 *  - forward: for every input element `(b, c, i, j)` the value is replicated into the 2x2 output tile
 *    at rows `2i, 2i+1` and columns `2j, 2j+1` of the `2H x 2W` output image;
 *  - backward: for every element `(b, c, i, j)` the gradient is the sum of the four `dout` values of
 *    that 2x2 tile.
 *
 * The reference's CPU comparison (`src/upsample-cuda/reference.h`) is single precision and the
 * reference's own GPU-vs-CPU check is an implicit-tolerance float compare. Because both operations
 * are a fixed-order sum of at most four values, a correct single-precision port reproduces the
 * reference single-precision arithmetic exactly, and the reference's per-tile relative-L2 metric
 * (`< 1e-8`) is a meaningful acceptance test.
 *
 * This file provides two trusted references:
 *  - `forwardReference` / `backwardReference`: single-precision references written independently from
 *    the definition above in the reference's exact accumulation order. These are the CPU references
 *    the accelerator result is compared against (`verifyForward`, `verifyBackward`); a correct port
 *    matches them bit-for-bit, so the per-tile relative-L2 is exactly zero and the `1e-8` threshold
 *    rejects any wrong tile mapping, border rule or offset.
 *  - `forwardOracleDouble` / `backwardOracleDouble`: independent double-precision oracles computed
 *    from the same definition. They do not share the single-precision accumulation order, so they pin
 *    the single-precision references to within one fp32 rounding unit
 *    (`verifyReferenceAgreement`). This guards against a shared bug in two single-precision
 *    reimplementations.
 *
 * Nothing in this file is device code; it only produces the input and the trusted results.
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

namespace upsampleBench
{
    /** Deterministic input generator (xorshift64*).
     *
     * The pinned reference fills its buffers with `make_random_float`, i.e. `rand()/RAND_MAX*2-1` in
     * `[-1, 1)`, after `srand(0)`. That generator is not thread-safe and its output depends on the
     * OpenMP thread count. This generator produces the same value distribution and range but is fully
     * deterministic and independent of the threading configuration.
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

        //! Uniform float in [-1, 1).
        float nextUnit()
        {
            return static_cast<float>(nextU32() * (1.0 / 4294967296.0)) * 2.0f - 1.0f;
        }

    private:
        std::uint64_t m_state;
    };

    //! Fill `count` floats with the deterministic range [-1, 1).
    inline void makeRandomFloat(float* data, std::size_t count, std::uint64_t seed)
    {
        Rng rng{seed};
        for(std::size_t i = 0; i < count; ++i)
            data[i] = rng.nextUnit();
    }

    //! Flat output offset of the top-left corner of the 2x2 tile belonging to element `(b, c, i, j)`.
    inline std::uint64_t tileBase(
        std::uint64_t b,
        std::uint64_t c,
        std::uint64_t i,
        std::uint64_t j,
        std::uint64_t C,
        std::uint64_t H,
        std::uint64_t W)
    {
        std::uint64_t const imgOut = (2u * H) * (2u * W);
        std::uint64_t const wOut = 2u * W;
        return b * C * imgOut + c * imgOut + (2u * i) * wOut + 2u * j;
    }

    //! Decompose the flat tile index of an `N, C, H, W` tensor into `(b, c, i, j)`.
    struct Decomposed
    {
        std::uint64_t b;
        std::uint64_t c;
        std::uint64_t i;
        std::uint64_t j;
    };

    inline Decomposed decompose(
        std::uint64_t p,
        std::uint64_t C,
        std::uint64_t H,
        std::uint64_t W)
    {
        std::uint64_t const img = H * W;
        Decomposed d;
        d.b = p / (C * img);
        d.c = (p / img) % C;
        d.i = (p / W) % H;
        d.j = p % W;
        return d;
    }

    // -------------------------------------------------------------------------------------------
    // Single-precision CPU references (the comparison target for the accelerator result).
    // -------------------------------------------------------------------------------------------

    /** Independent single-precision forward reference.
     *
     * For every input element the four elements of its 2x2 output tile are the input value exactly.
     * `out` is resized to `B*C*(2H)*(2W)`.
     */
    inline void forwardReference(
        float const* x,
        std::vector<float>& out,
        std::uint64_t B,
        std::uint64_t C,
        std::uint64_t H,
        std::uint64_t W)
    {
        std::uint64_t const img = H * W;
        std::uint64_t const numTiles = B * C * img;
        std::uint64_t const wOut = 2u * W;
        out.assign(static_cast<std::size_t>(B * C * (2u * H) * (2u * W)), 0.0f);

        for(std::uint64_t p = 0; p < numTiles; ++p)
        {
            Decomposed const d = decompose(p, C, H, W);
            float const value = x[p];
            std::uint64_t const base = tileBase(d.b, d.c, d.i, d.j, C, H, W);
            out[static_cast<std::size_t>(base)] = value;
            out[static_cast<std::size_t>(base + 1u)] = value;
            out[static_cast<std::size_t>(base + wOut)] = value;
            out[static_cast<std::size_t>(base + wOut + 1u)] = value;
        }
    }

    /** Independent single-precision backward reference.
     *
     * For every element the gradient is the single-precision sum of the four `dout` values of its 2x2
     * tile, accumulated in the reference order `((v00 + v01) + v10) + v11`. `dx` is resized to
     * `B*C*H*W`.
     */
    inline void backwardReference(
        float const* dout,
        std::vector<float>& dx,
        std::uint64_t B,
        std::uint64_t C,
        std::uint64_t H,
        std::uint64_t W)
    {
        std::uint64_t const img = H * W;
        std::uint64_t const numTiles = B * C * img;
        std::uint64_t const wOut = 2u * W;
        dx.assign(static_cast<std::size_t>(numTiles), 0.0f);

        for(std::uint64_t p = 0; p < numTiles; ++p)
        {
            Decomposed const d = decompose(p, C, H, W);
            std::uint64_t const base = tileBase(d.b, d.c, d.i, d.j, C, H, W);
            float sum = 0.0f;
            sum += dout[static_cast<std::size_t>(base)];
            sum += dout[static_cast<std::size_t>(base + 1u)];
            sum += dout[static_cast<std::size_t>(base + wOut)];
            sum += dout[static_cast<std::size_t>(base + wOut + 1u)];
            dx[static_cast<std::size_t>(p)] = sum;
        }
    }

    // -------------------------------------------------------------------------------------------
    // Double-precision oracles (independent, pin the single-precision references).
    // -------------------------------------------------------------------------------------------

    //! Independent double-precision forward oracle.
    inline void forwardOracleDouble(
        float const* x,
        std::vector<double>& out,
        std::uint64_t B,
        std::uint64_t C,
        std::uint64_t H,
        std::uint64_t W)
    {
        std::uint64_t const img = H * W;
        std::uint64_t const numTiles = B * C * img;
        std::uint64_t const wOut = 2u * W;
        out.assign(static_cast<std::size_t>(B * C * (2u * H) * (2u * W)), 0.0);

        for(std::uint64_t p = 0; p < numTiles; ++p)
        {
            Decomposed const d = decompose(p, C, H, W);
            double const value = static_cast<double>(x[p]);
            std::uint64_t const base = tileBase(d.b, d.c, d.i, d.j, C, H, W);
            out[static_cast<std::size_t>(base)] = value;
            out[static_cast<std::size_t>(base + 1u)] = value;
            out[static_cast<std::size_t>(base + wOut)] = value;
            out[static_cast<std::size_t>(base + wOut + 1u)] = value;
        }
    }

    //! Independent double-precision backward oracle.
    inline void backwardOracleDouble(
        float const* dout,
        std::vector<double>& dx,
        std::uint64_t B,
        std::uint64_t C,
        std::uint64_t H,
        std::uint64_t W)
    {
        std::uint64_t const img = H * W;
        std::uint64_t const numTiles = B * C * img;
        std::uint64_t const wOut = 2u * W;
        dx.assign(static_cast<std::size_t>(numTiles), 0.0);

        for(std::uint64_t p = 0; p < numTiles; ++p)
        {
            Decomposed const d = decompose(p, C, H, W);
            std::uint64_t const base = tileBase(d.b, d.c, d.i, d.j, C, H, W);
            double const row0 = static_cast<double>(dout[static_cast<std::size_t>(base)])
                                + static_cast<double>(dout[static_cast<std::size_t>(base + 1u)]);
            double const row1 = static_cast<double>(dout[static_cast<std::size_t>(base + wOut)])
                                + static_cast<double>(dout[static_cast<std::size_t>(base + wOut + 1u)]);
            dx[static_cast<std::size_t>(p)] = row0 + row1;
        }
    }

    // -------------------------------------------------------------------------------------------
    // Verification.
    // -------------------------------------------------------------------------------------------

    //! Tolerance used when the single-precision references are pinned against the double oracles.
    constexpr double referenceAgreementTolerance = 1e-6;

    /** Compare the single-precision references against the double-precision oracles.
     *
     * Both are computed from the same mathematical definition, so their difference is bounded by fp32
     * rounding. This proves the two single-precision references are not systematically wrong before
     * they are used as the accelerator comparison target.
     *
     * @return `EXIT_SUCCESS` when every element agrees within `tolerance`, else `EXIT_FAILURE`.
     */
    inline int verifyReferenceAgreement(
        float const* floatRef,
        std::vector<double> const& doubleRef,
        std::size_t count,
        double tolerance,
        char const* label)
    {
        double delta2 = 0.0;
        double ref2 = 0.0;
        double maxAbs = 0.0;
        std::size_t firstNonFinite = std::numeric_limits<std::size_t>::max();

        for(std::size_t i = 0; i < count; ++i)
        {
            if(!std::isfinite(floatRef[i]))
            {
                firstNonFinite = i;
                break;
            }
            double const e = doubleRef[i];
            double const d = static_cast<double>(floatRef[i]) - e;
            delta2 += d * d;
            ref2 += e * e;
            maxAbs = std::max(maxAbs, std::fabs(d));
        }

        if(firstNonFinite != std::numeric_limits<std::size_t>::max())
        {
            std::printf(
                "  %s: non-finite reference value (NaN/Inf) at index %llu\n",
                label,
                static_cast<unsigned long long>(firstNonFinite));
            std::printf("%s FAIL\n", label);
            return EXIT_FAILURE;
        }

        double const relL2 = (ref2 > 0.0) ? std::sqrt(delta2 / ref2) : 0.0;
        std::printf(
            "  %s: fp32-vs-fp64 global rel-L2 = %.3e, max abs = %.3e, tolerance = %.1e\n",
            label,
            relL2,
            maxAbs,
            tolerance);
        if(relL2 > tolerance)
        {
            std::printf("%s FAIL\n", label);
            return EXIT_FAILURE;
        }
        std::printf("%s PASS\n", label);
        return EXIT_SUCCESS;
    }

    /** Compare the forward pass tile-by-tile with a per-tile relative-L2 metric.
     *
     * For every 2x2 tile the squared error and the squared single-precision reference norm are
     * accumulated over the four output positions; a tile fails when
     * `sqrt(delta2 / ref2) > tolerance`. This cannot hide one wrong tile inside a large tensor the way
     * a single global norm can. NaN/Inf anywhere fails.
     *
     * @return `EXIT_SUCCESS` when every tile is finite and within `tolerance`, else `EXIT_FAILURE`.
     */
    inline int verifyForward(
        float const* got,
        std::vector<float> const& expected,
        std::uint64_t B,
        std::uint64_t C,
        std::uint64_t H,
        std::uint64_t W,
        double tolerance,
        char const* label)
    {
        std::uint64_t const img = H * W;
        std::uint64_t const numTiles = B * C * img;
        std::uint64_t const wOut = 2u * W;

        double globalDelta2 = 0.0;
        double globalRef2 = 0.0;
        double maxTileRelL2 = 0.0;
        double maxAbs = 0.0;
        std::uint64_t worstTile = 0u;
        std::size_t firstNonFinite = std::numeric_limits<std::size_t>::max();

        for(std::uint64_t p = 0; p < numTiles; ++p)
        {
            Decomposed const d = decompose(p, C, H, W);
            std::uint64_t const base = tileBase(d.b, d.c, d.i, d.j, C, H, W);
            std::uint64_t const idx[4] = {base, base + 1u, base + wOut, base + wOut + 1u};

            double tileDelta2 = 0.0;
            double tileRef2 = 0.0;
            for(int k = 0; k < 4; ++k)
            {
                float const g = got[static_cast<std::size_t>(idx[k])];
                if(!std::isfinite(g))
                {
                    if(firstNonFinite == std::numeric_limits<std::size_t>::max())
                        firstNonFinite = static_cast<std::size_t>(idx[k]);
                    tileDelta2 += std::numeric_limits<double>::infinity();
                    continue;
                }
                double const e = static_cast<double>(expected[static_cast<std::size_t>(idx[k])]);
                double const diff = static_cast<double>(g) - e;
                tileDelta2 += diff * diff;
                tileRef2 += e * e;
                maxAbs = std::max(maxAbs, std::fabs(diff));
            }
            globalDelta2 += tileDelta2;
            globalRef2 += tileRef2;

            double tileRelL2 = 0.0;
            if(tileRef2 > 0.0)
                tileRelL2 = std::sqrt(tileDelta2 / tileRef2);
            else if(tileDelta2 > 0.0)
                tileRelL2 = std::numeric_limits<double>::infinity();
            if(tileRelL2 > maxTileRelL2)
            {
                maxTileRelL2 = tileRelL2;
                worstTile = p;
            }
        }

        double const globalRelL2 = (globalRef2 > 0.0) ? std::sqrt(globalDelta2 / globalRef2) : 0.0;
        std::printf(
            "  %s: max tile rel-L2 = %.3e (tile %llu), global rel-L2 = %.3e, max abs = %.3e, tolerance = %.1e\n",
            label,
            maxTileRelL2,
            static_cast<unsigned long long>(worstTile),
            globalRelL2,
            maxAbs,
            tolerance);

        if(firstNonFinite != std::numeric_limits<std::size_t>::max())
        {
            std::printf(
                "  %s: non-finite value (NaN/Inf) at index %llu\n",
                label,
                static_cast<unsigned long long>(firstNonFinite));
            std::printf("%s FAIL\n", label);
            return EXIT_FAILURE;
        }
        if(maxTileRelL2 > tolerance)
        {
            std::printf("%s FAIL\n", label);
            return EXIT_FAILURE;
        }
        std::printf("%s PASS\n", label);
        return EXIT_SUCCESS;
    }

    /** Compare the backward pass element-by-element with a per-tile relative-L2 metric.
     *
     * The gradient tile has one result value; the metric is `|got - ref| / |ref|`. This is the same
     * per-tile relative measure as the forward pass. NaN/Inf anywhere fails.
     *
     * @return `EXIT_SUCCESS` when every element is finite and within `tolerance`, else `EXIT_FAILURE`.
     */
    inline int verifyBackward(
        float const* got,
        std::vector<float> const& expected,
        std::uint64_t B,
        std::uint64_t C,
        std::uint64_t H,
        std::uint64_t W,
        double tolerance,
        char const* label)
    {
        std::uint64_t const numTiles = B * C * H * W;
        double globalDelta2 = 0.0;
        double globalRef2 = 0.0;
        double maxTileRelL2 = 0.0;
        double maxAbs = 0.0;
        std::uint64_t worstTile = 0u;
        std::size_t firstNonFinite = std::numeric_limits<std::size_t>::max();

        for(std::uint64_t p = 0; p < numTiles; ++p)
        {
            float const g = got[static_cast<std::size_t>(p)];
            if(!std::isfinite(g))
            {
                if(firstNonFinite == std::numeric_limits<std::size_t>::max())
                    firstNonFinite = static_cast<std::size_t>(p);
                globalDelta2 += std::numeric_limits<double>::infinity();
                continue;
            }
            double const e = static_cast<double>(expected[static_cast<std::size_t>(p)]);
            double const diff = static_cast<double>(g) - e;
            globalDelta2 += diff * diff;
            globalRef2 += e * e;
            maxAbs = std::max(maxAbs, std::fabs(diff));

            double tileRelL2 = 0.0;
            if(e * e > 0.0)
                tileRelL2 = std::fabs(diff / e);
            else if(diff * diff > 0.0)
                tileRelL2 = std::numeric_limits<double>::infinity();
            if(tileRelL2 > maxTileRelL2)
            {
                maxTileRelL2 = tileRelL2;
                worstTile = p;
            }
        }

        double const globalRelL2 = (globalRef2 > 0.0) ? std::sqrt(globalDelta2 / globalRef2) : 0.0;
        std::printf(
            "  %s: max element rel-L2 = %.3e (tile %llu), global rel-L2 = %.3e, max abs = %.3e, tolerance = %.1e\n",
            label,
            maxTileRelL2,
            static_cast<unsigned long long>(worstTile),
            globalRelL2,
            maxAbs,
            tolerance);

        if(firstNonFinite != std::numeric_limits<std::size_t>::max())
        {
            std::printf(
                "  %s: non-finite value (NaN/Inf) at index %llu\n",
                label,
                static_cast<unsigned long long>(firstNonFinite));
            std::printf("%s FAIL\n", label);
            return EXIT_FAILURE;
        }
        if(maxTileRelL2 > tolerance)
        {
            std::printf("%s FAIL\n", label);
            return EXIT_FAILURE;
        }
        std::printf("%s PASS\n", label);
        return EXIT_SUCCESS;
    }

} // namespace upsampleBench
