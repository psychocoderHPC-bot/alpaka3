/* Copyright 2026 René Widera
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

/** @file Independent host reference and verification helpers for the HeCBench `dct8x8` benchmark.
 *
 * The pinned HeCBench reference (`src/dct8x8-cuda/main.cu` + `kernels.cu` + `DCT8x8_gold.cu`,
 * OpenMP twin `src/dct8x8-omp/`) performs an 8x8 DCT (`DCT_FORWARD` = 666) or its inverse
 * (`DCT_INVERSE` = 777) on every full 8x8 block of a float image.
 *
 * The transform is an orthonormal DCT-II applied separably (rows, then columns); the forward
 * coefficients are `sqrt(1/8)` for the DC term and `sqrt(2/8)` otherwise, and the inverse is the
 * transpose (DCT-III). The pinned reference uses the hardcoded AAN fast transform from the NVIDIA
 * CUDA SDK sample with the constants in `DCT8x8.h`; it was verified numerically to equal this
 * orthonormal DCT-II/DCT-III pair. This file contains a deliberately different, straightforward
 * double-precision matrix implementation of the same transform. It is not used for timing; it is
 * the trusted oracle for `verify()`.
 *
 * Only full 8x8 blocks are transformed, exactly like the pinned reference (`i + 8 - 1 < imageH`,
 * `j + 8 - 1 < imageW`). Every cell outside a full block is left equal to the input. The reference
 * leaves the remainder uninitialized; this port defines it as a copy of the input, which does not
 * change the transform of any full block.
 *
 * Nothing in this file is device code.
 */

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace dct8x8Bench
{
    //! pi with enough digits for double precision.
    constexpr double pi = 3.14159265358979323846264338327950288;

    //! Seed used by the pinned reference `main`: `srand(2009)`.
    constexpr unsigned int referenceSeed = 2009u;

    //! Block edge length of the transform.
    constexpr int blockSize = 8;

    /** Orthonormal DCT-II basis of length 8: `B[k][n] = a(k) * cos(pi * (2n+1) * k / 16)` with
     *  `a(0) = sqrt(1/8)` and `a(k>0) = sqrt(2/8)`. The inverse (DCT-III) uses the transpose.
     *
     * The basis is computed once on first use so the oracle stays O(64) per 1D transform and runs
     * in a few seconds even for the 8192x8192 measured image.
     */
    struct Basis
    {
        double b[blockSize][blockSize];

        Basis()
        {
            for(int k = 0; k < blockSize; ++k)
            {
                double const a = (k == 0) ? std::sqrt(1.0 / blockSize) : std::sqrt(2.0 / blockSize);
                for(int n = 0; n < blockSize; ++n)
                    b[k][n] = a * std::cos(pi * static_cast<double>(2 * n + 1) * static_cast<double>(k) / (2.0 * blockSize));
            }
        }
    };

    inline Basis const& basis()
    {
        static Basis const instance;
        return instance;
    }

    /** Orthonormal 1D DCT of length 8.
     *
     * @param dir +1 selects the forward transform (DCT-II), -1 the inverse (DCT-III, the transpose
     *            of DCT-II).
     */
    inline void dct1d(double* out, const double* in, int dir)
    {
        Basis const& B = basis();
        if(dir > 0)
        {
            for(int k = 0; k < blockSize; ++k)
            {
                double s = 0.0;
                for(int n = 0; n < blockSize; ++n)
                    s += B.b[k][n] * in[n];
                out[k] = s;
            }
        }
        else
        {
            for(int n = 0; n < blockSize; ++n)
            {
                double s = 0.0;
                for(int k = 0; k < blockSize; ++k)
                    s += B.b[k][n] * in[k];
                out[n] = s;
            }
        }
    }

    /** Fill the host input exactly like the pinned reference `main`:
     *  `srand(2009); x = (float)rand() / (float)RAND_MAX;`.
     */
    inline void generateInput(float* data, std::uint32_t width, std::uint32_t height, unsigned int seed)
    {
        std::srand(seed);
        std::size_t const count = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
        for(std::size_t i = 0; i < count; ++i)
            data[i] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
    }

    /** Independent double-precision reference transform.
     *
     * Applies the separable orthonormal DCT-II (`dir > 0`) or DCT-III (`dir < 0`) to every full 8x8
     * block (rows first, then columns), exactly like `DCT8x8CPU` in the pinned reference. All cells
     * outside full blocks copy the input.
     */
    inline void referenceTransform(
        float const* input,
        std::vector<double>& out,
        std::uint32_t width,
        std::uint32_t height,
        int dir)
    {
        std::size_t const count = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
        std::vector<double> buf(count);
        for(std::size_t i = 0; i < count; ++i)
            buf[i] = static_cast<double>(input[i]);

        for(std::uint32_t bi = 0; bi + static_cast<std::uint32_t>(blockSize) - 1u < height;
            bi += static_cast<std::uint32_t>(blockSize))
        {
            for(std::uint32_t bj = 0; bj + static_cast<std::uint32_t>(blockSize) - 1u < width;
                bj += static_cast<std::uint32_t>(blockSize))
            {
                // rows
                for(std::uint32_t k = 0; k < static_cast<std::uint32_t>(blockSize); ++k)
                {
                    double D[blockSize];
                    double O[blockSize];
                    for(int i = 0; i < blockSize; ++i)
                        D[i] = buf[(static_cast<std::size_t>(bi + k)) * width + bj + static_cast<std::uint32_t>(i)];
                    dct1d(O, D, dir);
                    for(int i = 0; i < blockSize; ++i)
                        buf[(static_cast<std::size_t>(bi + k)) * width + bj + static_cast<std::uint32_t>(i)] = O[i];
                }
                // columns
                for(std::uint32_t k = 0; k < static_cast<std::uint32_t>(blockSize); ++k)
                {
                    double D[blockSize];
                    double O[blockSize];
                    for(int i = 0; i < blockSize; ++i)
                        D[i] = buf[(static_cast<std::size_t>(bi + static_cast<std::uint32_t>(i))) * width + bj + k];
                    dct1d(O, D, dir);
                    for(int i = 0; i < blockSize; ++i)
                        buf[(static_cast<std::size_t>(bi + static_cast<std::uint32_t>(i))) * width + bj + k] = O[i];
                }
            }
        }
        out.swap(buf);
    }

    /** Compare a float result against the double reference with the reference's own metric.
     *
     * The pinned `Verify()` prints the relative L2 norm `sqrt(sum((gpu-cpu)^2) / sum(cpu^2))` and
     * requires `< 1e-6`. This function reproduces that metric, additionally rejects any NaN or Inf
     * in the result and reports the maximum absolute difference.
     *
     * @return `EXIT_SUCCESS` on a finite result within `tolerance`, otherwise `EXIT_FAILURE`.
     */
    inline int verify(
        float const* got,
        std::vector<double> const& expected,
        std::size_t count,
        double tolerance,
        char const* label)
    {
        double sum = 0.0;
        double delta = 0.0;
        double maxAbs = 0.0;
        std::size_t firstNonFinite = count;
        for(std::size_t i = 0; i < count; ++i)
        {
            if(!std::isfinite(got[i]))
            {
                firstNonFinite = i;
                break;
            }
            double const e = expected[i];
            sum += e * e;
            double const d = static_cast<double>(got[i]) - e;
            delta += d * d;
            double const ad = std::fabs(d);
            if(ad > maxAbs)
                maxAbs = ad;
        }

        if(firstNonFinite != count)
        {
            std::printf(
                "  %s: non-finite value (NaN/Inf) at index %llu\n",
                label,
                static_cast<unsigned long long>(firstNonFinite));
            std::printf("%s FAIL\n", label);
            return EXIT_FAILURE;
        }

        double const relL2 = (sum > 0.0) ? std::sqrt(delta / sum) : (delta > 0.0 ? 1.0 : 0.0);
        std::printf(
            "  %s: relative L2 = %.3e, max abs = %.3e, tolerance = %.1e\n",
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

} // namespace dct8x8Bench
