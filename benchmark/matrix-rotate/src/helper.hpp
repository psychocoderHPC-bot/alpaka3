/* Copyright 2026 Rene Widera
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

/** @file Independent host reference and verification helpers for the matrix-rotate benchmark.
 * @attention The license differs from the benchmark implementation because this file contains no parts of the
 * HeCBench code base.
 */

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace matrixRotate
{
    /** Fill a square matrix with the HeCBench initialization `m[i][j] = i * n + j`.
     *
     * Every entry is a non-negative integer below `n * n`, so for realistic side lengths the value is exactly
     * representable as `float` and the in-place rotation, which is a pure permutation, is exact.
     */
    inline void initMatrix(std::vector<float>& matrix, int n)
    {
        matrix.resize(static_cast<std::size_t>(n) * static_cast<std::size_t>(n));
        for(int i = 0; i < n; ++i)
            for(int j = 0; j < n; ++j)
                matrix[static_cast<std::size_t>(i) * n + j] = static_cast<float>(i * n + j);
    }

    /** Independent, double-precision in-place 90-degree clockwise rotation.
     *
     * This reproduces the HeCBench ring-by-ring corner swap scheme (`first = layer`, `last = n - 1 - layer`),
     * but runs in `double` in a plain host loop with no parallelism. It is the golden reference: because the
     * operation only permutes entries, the alpaka result must match the double reference entry for entry after
     * converting back to `float`.
     */
    inline void hostRotate(std::vector<double>& matrix, int n)
    {
        for(int layer = 0; layer < n / 2; ++layer)
        {
            int const first = layer;
            int const last = n - 1 - layer;
            for(int i = first; i < last; ++i)
            {
                int const offset = i - first;
                double const top = matrix[static_cast<std::size_t>(first) * n + i];
                // left -> top
                matrix[static_cast<std::size_t>(first) * n + i]
                    = matrix[static_cast<std::size_t>(last - offset) * n + first];
                // bottom -> left
                matrix[static_cast<std::size_t>(last - offset) * n + first]
                    = matrix[static_cast<std::size_t>(last) * n + (last - offset)];
                // right -> bottom
                matrix[static_cast<std::size_t>(last) * n + (last - offset)]
                    = matrix[static_cast<std::size_t>(i) * n + last];
                // top -> right
                matrix[static_cast<std::size_t>(i) * n + last] = top;
            }
        }
    }

    /** Apply the reference rotation `repeat` times, matching the HeCBench reference loop. */
    inline void referenceRotate(std::vector<double>& matrix, int n, int repeat)
    {
        for(int r = 0; r < repeat; ++r)
            hostRotate(matrix, n);
    }

    /** Exact comparison of the produced float matrix against the double golden matrix.
     *
     * The reference and the benchmark perform only permutations of exactly representable integer values, so the
     * comparison is bit-exact: the tolerance is zero. A nonzero finite tolerance would be unjustified and could
     * hide a wrong addressing scheme. Non-finite values are always reported and fail the check.
     *
     * @return EXIT_SUCCESS when the matrices match exactly, EXIT_FAILURE otherwise.
     */
    inline int verify(float const* got, std::vector<double> const& gold, int n)
    {
        std::size_t const count = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
        if(gold.size() != count)
        {
            printf("Verification FAILED: reference size mismatch (reference %zu, expected %zu)\n", gold.size(), count);
            return EXIT_FAILURE;
        }

        int failures = 0;
        for(int i = 0; i < n; ++i)
        {
            for(int j = 0; j < n; ++j)
            {
                std::size_t const idx = static_cast<std::size_t>(i) * n + j;
                float const value = got[idx];
                double const expected = gold[idx];
                if(!std::isfinite(static_cast<double>(value)))
                {
                    if(failures < 5)
                        printf("non-finite output at (%d,%d): %g\n", i, j, static_cast<double>(value));
                    ++failures;
                    continue;
                }
                if(value != static_cast<float>(expected))
                {
                    if(failures < 5)
                        printf(
                            "mismatch at (%d,%d): got %.9g expected %.9g\n",
                            i,
                            j,
                            static_cast<double>(value),
                            expected);
                    ++failures;
                }
            }
        }

        printf("checked %zu entries (n=%d, tolerance=exact): %s\n",
               count,
               n,
               failures ? "Verification FAILED" : "Verification PASSED");
        return failures ? EXIT_FAILURE : EXIT_SUCCESS;
    }
} // namespace matrixRotate
