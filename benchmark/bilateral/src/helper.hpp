/* Copyright 2026 René Widera
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

/** @file Host reference, deterministic input generation and verification for the `bilateral` benchmark.
 *
 * The pinned HeCBench reference (`src/bilateral-cuda/main.cu`, `src/bilateral-omp/main.cpp`,
 * `src/bilateral-cuda/reference.h`) filters a single-channel float image with a window filter of radius
 * `R` in {3, 6, 9}. Per output pixel `(idx, idy)` it accumulates
 *
 *   range      = -(I - I_w)^2 / (2 * variance_I)
 *   spatial    = -((idk-idx)^2 + (idl-idy)^2) / (2 * variance_spatial)
 *   weight     = a_square * expf(spatial + range)
 *   out        = sum(I_w * weight) / sum(weight)
 *
 * with `a_square = 0.5f / (variance_I * pi)`. The window uses the reference's "mirror edges" index
 * folding:
 *
 *   if (idk < 0)          idk = -idk;
 *   if (idl < 0)          idl = -idl;
 *   if (idk > w - 1)      idk = w - 1 - i;
 *   if (idl > h - 1)      idl = h - 1 - j;
 *
 * This port keeps that folding verbatim. It is only well defined when the image is at least as large
 * as the window (`width, height >= 2*R + 1`); the benchmark enforces that precondition, exactly the
 * regime the pinned reference is written for.
 *
 * Input generation reproduces the reference `srand(123); rand() % 256` bit for bit. glibc's `rand()`
 * is a `TYPE_3` additive-feedback generator with a degree-31 state, seed 123 and separation 3; the
 * `GlibcRandom` class below reimplements that recurrence so the input is identical on any host libc.
 * It was cross-checked against the system `rand()` for the first 4096 outputs (see README).
 *
 * Verification uses a double-precision host reference. Single-precision accumulation of up to 100
 * terms with magnitudes of order 100 has a worst-case rounding of a few 1e-4; the tolerance is the
 * pinned harness's own 1e-3 (see README), applied after explicit NaN/Inf and zero-normalisation
 * checks.
 */

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

// The index folding is shared between the host reference and the device kernel. Annotate it with
// `__host__ __device__` only when compiling with a device compiler; plain host builds need no
// annotation. This avoids clashing with alpaka's own ALPAKA_FN_HOST_ACC macro, whose include order in
// the generated device translation unit is not under our control.
#if defined(__CUDACC__) || defined(__HIPCC__)
#    define BILATERAL_FN_HOST_ACC __host__ __device__
#else
#    define BILATERAL_FN_HOST_ACC
#endif

namespace bilateralBench
{
    //! Reference window radii, in the order the pinned `main` runs them.
    constexpr int radii[3] = {3, 6, 9};

    /** glibc-compatible `random()` (`TYPE_3`: degree 31, separation 3).
     *
     * Reimplements `__srandom_r` / `__random_r` from glibc's `random_r.c` (LGPL-2.1), restricted to
     * the single state size glibc uses for `rand()`. This is an independent reimplementation, not a
     * copy; it exists only to reproduce the reference input sequence on any host libc.
     */
    class GlibcRandom
    {
    public:
        static constexpr int degree = 31;
        static constexpr int separation = 3;

        explicit GlibcRandom(std::uint32_t seed)
        {
            seedInternal(seed);
        }

        //! glibc `random()`: returns a non-negative signed 31-bit value.
        std::int32_t next()
        {
            std::uint32_t const val
                = static_cast<std::uint32_t>(m_state[m_fptr]) + static_cast<std::uint32_t>(m_state[m_rptr]);
            m_state[m_fptr] = static_cast<std::int32_t>(val);
            std::int32_t const result = static_cast<std::int32_t>(val >> 1);
            ++m_fptr;
            if(m_fptr >= degree)
            {
                m_fptr = 0;
                ++m_rptr;
            }
            else
            {
                ++m_rptr;
                if(m_rptr >= degree)
                    m_rptr = 0;
            }
            return result;
        }

    private:
        void seedInternal(std::uint32_t seed)
        {
            if(seed == 0u)
                seed = 1u;
            std::int32_t word = static_cast<std::int32_t>(seed);
            m_state[0] = word;
            for(int i = 1; i < degree; ++i)
            {
                long const hi = word / 127773;
                long const lo = word % 127773;
                word = static_cast<std::int32_t>(16807 * lo - 2836 * hi);
                if(word < 0)
                    word += 2'147'483'647;
                m_state[i] = word;
            }
            // glibc discards degree*10 values after seeding.
            m_fptr = separation;
            m_rptr = 0;
            int kc = degree * 10;
            while(kc-- > 0)
                (void) next();
        }

        std::int32_t m_state[degree]{};
        int m_fptr = 0;
        int m_rptr = 0;
    };

    /** Reproduce the reference input: `srand(123); src[i] = rand() % 256` as floats. */
    inline void generateInput(float* src, std::size_t numPixels)
    {
        GlibcRandom rng{123u};
        for(std::size_t i = 0; i < numPixels; ++i)
            src[i] = static_cast<float>(static_cast<unsigned int>(rng.next()) % 256u);
    }

    //! Reference index folding ("mirror edges"), verbatim from `reference.h`.
    BILATERAL_FN_HOST_ACC inline void mirrorIndex(int& idk, int& idl, int idx, int idy, int i, int j, int w, int h)
    {
        idk = idx + i;
        idl = idy + j;
        if(idk < 0)
            idk = -idk;
        if(idl < 0)
            idl = -idl;
        if(idk > w - 1)
            idk = w - 1 - i;
        if(idl > h - 1)
            idl = h - 1 - j;
    }

    /** Double-precision host reference for a single radius.
     *
     * @return EXIT_SUCCESS, or EXIT_FAILURE if a weight normalisation is zero (non-finite result).
     */
    template<int R>
    inline int bilateralReference(
        float const* in,
        float* out,
        int w,
        int h,
        double a_square,
        double variance_I,
        double variance_spatial)
    {
        int status = EXIT_SUCCESS;
        for(int idx = 0; idx < w; ++idx)
        {
            for(int idy = 0; idy < h; ++idy)
            {
                int const id = idy * w + idx;
                double const I = static_cast<double>(in[id]);
                double res = 0.0;
                double normalization = 0.0;
                for(int i = -R; i <= R; ++i)
                {
                    for(int j = -R; j <= R; ++j)
                    {
                        int idk = 0;
                        int idl = 0;
                        mirrorIndex(idk, idl, idx, idy, i, j, w, h);
                        if(idk < 0 || idk >= w || idl < 0 || idl >= h)
                        {
                            std::printf(
                                "error: reference index out of range at (%d,%d) window (%d,%d): (%d,%d)\n",
                                idx,
                                idy,
                                i,
                                j,
                                idk,
                                idl);
                            return EXIT_FAILURE;
                        }
                        double const I_w = static_cast<double>(in[idl * w + idk]);
                        double const range = -(I - I_w) * (I - I_w) / (2.0 * variance_I);
                        double const dx = static_cast<double>(idk - idx);
                        double const dy = static_cast<double>(idl - idy);
                        double const spatial = -(dx * dx + dy * dy) / (2.0 * variance_spatial);
                        double const weight = a_square * std::exp(spatial + range);
                        normalization += weight;
                        res += I_w * weight;
                    }
                }
                if(!(normalization != 0.0))
                {
                    std::printf("error: reference zero/NaN normalisation at (%d,%d)\n", idx, idy);
                    return EXIT_FAILURE;
                }
                out[id] = static_cast<float>(res / normalization);
            }
        }
        return status;
    }

    /** Runtime-radius variant of the double host reference, for radii other than the pinned {3,6,9}.
     * The arithmetic and traversal order are identical to `bilateralReference`.
     */
    inline int bilateralReferenceRuntime(
        float const* in,
        float* out,
        int w,
        int h,
        double a_square,
        double variance_I,
        double variance_spatial,
        int radius)
    {
        for(int idx = 0; idx < w; ++idx)
        {
            for(int idy = 0; idy < h; ++idy)
            {
                int const id = idy * w + idx;
                double const I = static_cast<double>(in[id]);
                double res = 0.0;
                double normalization = 0.0;
                for(int i = -radius; i <= radius; ++i)
                {
                    for(int j = -radius; j <= radius; ++j)
                    {
                        int idk = 0;
                        int idl = 0;
                        mirrorIndex(idk, idl, idx, idy, i, j, w, h);
                        if(idk < 0 || idk >= w || idl < 0 || idl >= h)
                        {
                            std::printf(
                                "error: reference index out of range at (%d,%d) window (%d,%d): (%d,%d)\n",
                                idx,
                                idy,
                                i,
                                j,
                                idk,
                                idl);
                            return EXIT_FAILURE;
                        }
                        double const I_w = static_cast<double>(in[idl * w + idk]);
                        double const range = -(I - I_w) * (I - I_w) / (2.0 * variance_I);
                        double const dx = static_cast<double>(idk - idx);
                        double const dy = static_cast<double>(idl - idy);
                        double const spatial = -(dx * dx + dy * dy) / (2.0 * variance_spatial);
                        double const weight = a_square * std::exp(spatial + range);
                        normalization += weight;
                        res += I_w * weight;
                    }
                }
                if(!(normalization != 0.0))
                {
                    std::printf("error: reference zero/NaN normalisation at (%d,%d)\n", idx, idy);
                    return EXIT_FAILURE;
                }
                out[id] = static_cast<float>(res / normalization);
            }
        }
        return EXIT_SUCCESS;
    }

    /** Explicit NaN/Inf check on an accelerator result.
     *
     * @return EXIT_SUCCESS when every value is finite, EXIT_FAILURE and a message otherwise.
     */
    inline int checkFinite(float const* values, std::size_t count, char const* label)
    {
        for(std::size_t i = 0; i < count; ++i)
        {
            if(!std::isfinite(values[i]))
            {
                std::printf("error: non-finite value at %s[%zu] = %g\n", label, i, static_cast<double>(values[i]));
                return EXIT_FAILURE;
            }
        }
        return EXIT_SUCCESS;
    }

    /** Compare an accelerator result against the double host reference.
     *
     * Tolerance `tol = 1e-3`, the pinned harness's own threshold for the identical computation. It is
     * justified by single-precision reduction over up to (2R+1)^2 = 361 terms (R=9) of magnitude up to
     * ~255 with a normalisation of order 0.5; the accumulated rounding is at most a few 1e-4.
     *
     * @return EXIT_SUCCESS on pass, EXIT_FAILURE on the first out-of-tolerance pixel.
     */
    inline int verify(float const* computed, float const* reference, std::size_t count, double tol, char const* label)
    {
        double maxAbs = 0.0;
        for(std::size_t i = 0; i < count; ++i)
        {
            double const d = std::fabs(static_cast<double>(computed[i]) - static_cast<double>(reference[i]));
            if(d > maxAbs)
                maxAbs = d;
            if(d > tol)
            {
                std::printf(
                    "error: %s mismatch at %zu: computed=%g reference=%g |diff|=%g > tol=%g\n",
                    label,
                    i,
                    static_cast<double>(computed[i]),
                    static_cast<double>(reference[i]),
                    d,
                    tol);
                return EXIT_FAILURE;
            }
        }
        std::printf("  %s max_abs_diff=%.3e (tol=%.1e)\n", label, maxAbs, tol);
        return EXIT_SUCCESS;
    }

} // namespace bilateralBench
