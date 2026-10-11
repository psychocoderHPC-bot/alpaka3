/* Copyright 2026 René Widera
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

/** @file Independent host reference and verification for the HeCBench `laplace3d` benchmark.
 *
 * The HeCBench `laplace3d` benchmark performs `niter` Jacobi iterations of the seven-point
 * Laplacian stencil on a regular `Nx x Ny x Nz` grid. The field indexing is row-major with x
 * innermost:
 *
 *     ind = i + j*Nx + k*Nx*Ny
 *
 * Boundary cells (any of `i==0`, `i==Nx-1`, `j==0`, `j==Ny-1`, `k==0`, `k==Nz-1`) are Dirichlet
 * cells and keep their value across every iteration. Interior cells are updated with the
 * single-precision coefficients `1/6`:
 *
 *     u2[ind] = ( u1[ind-1] + u1[ind+1]
 *               + u1[ind-Nx] + u1[ind+Nx]
 *               + u1[ind-Nx*Ny] + u1[ind+Nx*Ny] ) * (1/6)
 *
 * This file re-implements that update in **double precision** on the host and compares it to the
 * (single precision) device result. Independence from the device kernel: this reference shares no
 * indexing helper, no halo handling, no tiling and no launch configuration with
 * `main_alpaka.cpp`; it re-derives the neighbours, the boundary predicate and the initial field. A
 * kernel with a wrong stencil, a wrong boundary condition or a wrong stride therefore cannot pass
 * by construction.
 *
 * Tolerance rationale: the device result is `float`, the reference is `double`. The update is a
 * convex combination of six neighbours (weights sum to one), so every value stays bounded by the
 * `{0,1}` initial/boundary data; one `float` ulp near 1 is `2^-24 = 5.96e-8`. With Dirichlet
 * boundaries the interior solution is a smooth, rapidly converging field, so the per-iteration
 * round-off growth is much slower than in a purely diffusive periodic problem. The bound
 * `1e-4 + 1e-6 * iters` comfortably exceeds the measured drift at every validated size without
 * being widened to hide a bug. NaN/Inf are rejected explicitly. A nonzero exit code signals
 * failure.
 */

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace laplace3d
{
    //! Dirichlet boundary predicate, shared only within this reference/verifier file.
    inline bool isBoundary(int i, int j, int k, int Nx, int Ny, int Nz)
    {
        return i == 0 || i == Nx - 1 || j == 0 || j == Ny - 1 || k == 0 || k == Nz - 1;
    }

    /** Reproduce the HeCBench initial field.
     *
     * Every boundary cell is `1.0f`, every interior cell is `0.0f` (the pinned reference writes
     * exactly this into `h_u1`).
     */
    inline std::vector<float> makeInitialField(int Nx, int Ny, int Nz)
    {
        std::vector<float> field(static_cast<std::size_t>(Nx) * static_cast<std::size_t>(Ny)
                                     * static_cast<std::size_t>(Nz),
                                 0.f);
        for(int k = 0; k < Nz; ++k)
            for(int j = 0; j < Ny; ++j)
                for(int i = 0; i < Nx; ++i)
                {
                    std::size_t const idx
                        = static_cast<std::size_t>(i)
                          + static_cast<std::size_t>(j) * static_cast<std::size_t>(Nx)
                          + static_cast<std::size_t>(k) * static_cast<std::size_t>(Nx)
                                * static_cast<std::size_t>(Ny);
                    if(isBoundary(i, j, k, Nx, Ny, Nz))
                        field[idx] = 1.f;
                }
        return field;
    }

    /** Advance one double-precision seven-point Jacobi step from @p in into @p out. */
    inline void stepD(
        std::vector<double> const& in,
        std::vector<double>& out,
        int Nx,
        int Ny,
        int Nz)
    {
        double const sixth = 1.0 / 6.0;
        std::size_t const strideJ = static_cast<std::size_t>(Nx);
        std::size_t const strideK = static_cast<std::size_t>(Nx) * static_cast<std::size_t>(Ny);
        for(int k = 0; k < Nz; ++k)
            for(int j = 0; j < Ny; ++j)
                for(int i = 0; i < Nx; ++i)
                {
                    std::size_t const ind = static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * strideJ
                                            + static_cast<std::size_t>(k) * strideK;
                    if(isBoundary(i, j, k, Nx, Ny, Nz))
                        out[ind] = in[ind];
                    else
                        out[ind] = (in[ind - 1] + in[ind + 1] + in[ind - strideJ] + in[ind + strideJ]
                                    + in[ind - strideK] + in[ind + strideK])
                                   * sixth;
                }
    }

    /** Run the double-precision reference for exactly @p iters steps starting from @p init. */
    inline std::vector<double> referenceField(std::vector<float> const& init, int Nx, int Ny, int Nz, int iters)
    {
        std::vector<double> a(init.begin(), init.end());
        std::vector<double> b(init.size(), 0.0);
        for(int it = 0; it < iters; ++it)
        {
            stepD(a, b, Nx, Ny, Nz);
            std::swap(a, b);
        }
        return a;
    }

    /** Host reference tolerance as a function of the iteration count.
     *
     * `1e-4 + 1e-6 * iters`: at the validated iteration counts this stays an order of magnitude
     * above the measured single-precision drift of the smooth Dirichlet solution. It is not
     * widened to hide a bug: a wrong stencil, a swapped axis or a wrong boundary value produces
     * O(1) differences, far above this bound.
     */
    inline double verifyTolerance(int iters)
    {
        return 1e-4 + 1e-6 * static_cast<double>(iters);
    }

    /** Verify the device result against the independent double-precision reference.
     *
     * Checks:
     *  - `Nx`, `Ny`, `Nz` and the iteration count are valid;
     *  - every device value is finite (NaN/Inf rejected explicitly);
     *  - the final field matches the double reference computed from the same initial field to the
     *    same iteration within @c laplace3d::verifyTolerance().
     *
     * @param init   initial single-precision field, `Nx*Ny*Nz` elements, row-major
     * @param Nx,Ny,Nz grid dimensions
     * @param iters  number of applied steps
     * @param result device result, `Nx*Ny*Nz` elements, row-major
     * @return EXIT_SUCCESS on pass, EXIT_FAILURE on failure
     */
    template<typename T_Result>
    int verify(std::vector<float> const& init, int Nx, int Ny, int Nz, int iters, T_Result const& result)
    {
        std::size_t const count = static_cast<std::size_t>(Nx) * static_cast<std::size_t>(Ny)
                                  * static_cast<std::size_t>(Nz);

        if(Nx <= 0 || Ny <= 0 || Nz <= 0 || iters < 0)
        {
            printf("laplace3d verification FAILED: invalid dimensions Nx=%d Ny=%d Nz=%d iters=%d\n", Nx, Ny, Nz, iters);
            return EXIT_FAILURE;
        }

        for(std::size_t k = 0; k < count; ++k)
        {
            float const v = static_cast<float>(result[k]);
            if(!std::isfinite(v))
            {
                printf("laplace3d verification FAILED: non-finite value at index %zu\n", k);
                return EXIT_FAILURE;
            }
        }

        std::vector<double> const reference = referenceField(init, Nx, Ny, Nz, iters);
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
                "laplace3d verification FAILED: max |alpaka - double ref| = %.6e at index %zu exceeds %.6e\n",
                maxDiff,
                maxIdx,
                tolerance);
            return EXIT_FAILURE;
        }

        printf(
            "laplace3d verification PASSED: Nx=%d Ny=%d Nz=%d iters=%d max |alpaka - double ref| = %.6e "
            "(tolerance %.6e)\n",
            Nx,
            Ny,
            Nz,
            iters,
            maxDiff,
            tolerance);
        return EXIT_SUCCESS;
    }
} // namespace laplace3d
