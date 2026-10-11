/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 * Copyright 2026 Rene Widera
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

/** @file Independent host reference and verification helper for the ising benchmark.
 *
 * The reference is an *independent* formulation of the Ising update: it stores the full `nx x ny` lattice and applies
 * a checkerboard update to one color at a time, in place, reading the frozen opposite color. The HeCBench reference
 * instead packs the two colors row-wise into two `nx x (ny/2)` arrays and updates one color from the other. Both
 * compute the same Markov chain, but the reference here uses a different layout, different neighbor indexing and
 * double-precision arithmetic, so it does not merely re-implement the packed indexing of the kernel.
 *
 * @attention The license differs from the benchmark implementation because this file contains no parts of the
 * HeCBench code base.
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

namespace ising
{
    //! Critical temperature constant of the square-lattice Ising model, identical to `TCRIT` in the HeCBench sources.
    constexpr float tcrit = 2.26918531421f;

    /** Reproduce the exact random sequence of the HeCBench reference (`srand(seed)` then `rand()/RAND_MAX`).
     *
     * The CUDA reference fills `nx*ny/2` floats once (its `#ifndef CURAND` path) and reuses that buffer for the whole
     * simulation, so the same buffer drives both colors of the checkerboard. `rand()` is implementation defined;
     * glibc's sequence is reproduced so host and device consume identical inputs.
     */
    inline std::vector<float> makeRandVals(std::size_t count, unsigned int seed)
    {
        std::vector<float> randvals(count);
        srand(seed);
        for(std::size_t i = 0; i < count; ++i)
            randvals[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        return randvals;
    }

    /** Independent double-precision checkerboard reference for the Ising model.
     *
     * Spins live on the full `nx x ny` lattice as `+1` or `-1`. One sweep updates the cells with `(i + c) % 2 == 0`
     * (black) and then the cells with `(i + c) % 2 == 1` (white); within a color every cell is updated in parallel
     * from the frozen opposite color, so the update is in place and race free (a black cell only reads white cells
     * and itself, and white cells are not written during the black pass). The four orthogonal neighbors use periodic
     * boundary conditions.
     *
     * The packed HeCBench layout maps full-lattice cell `(i, c)` of color `black` to packed index
     * `i*(ny/2) + (c - i%2)/2` and of color `white` to `i*(ny/2) + (c - (1 - i%2))/2`; both colors consume the same
     * random value for a given full-lattice cell.
     *
     * The acceptance ratio `exp(-2 * beta * nn_sum * lij)` is evaluated in double precision, a stricter and
     * independent computation compared to the kernel's single-precision `expf`.
     */
    struct CheckerboardReference
    {
        long long nx; //!< number of rows
        long long ny; //!< number of columns (even)
        double invTemp;
        std::vector<float> const& randvals;
        std::vector<signed char> lattice; //!< full `nx x ny` lattice
        unsigned int numThreads;

        static std::size_t packedIndex(long long i, long long c, long long ny, bool black)
        {
            long long const j = (c - (black ? (i % 2) : (1 - (i % 2)))) / 2;
            return static_cast<std::size_t>(i * (ny / 2) + j);
        }

        CheckerboardReference(
            long long nx_,
            long long ny_,
            float alpha,
            std::vector<float> const& randvals_,
            unsigned int numThreads_ = 0)
            : nx(nx_)
            , ny(ny_)
            , invTemp(1.0 / (static_cast<double>(alpha) * static_cast<double>(tcrit)))
            , randvals(randvals_)
            , lattice(static_cast<std::size_t>(nx_ * ny_))
            , numThreads(numThreads_)
        {
            if(numThreads == 0)
                numThreads = std::max(1u, std::thread::hardware_concurrency());

            for(long long i = 0; i < nx; ++i)
                for(long long c = 0; c < ny; ++c)
                {
                    bool const black = ((i + c) % 2 == 0);
                    std::size_t const p = packedIndex(i, c, ny, black);
                    lattice[static_cast<std::size_t>(i * ny + c)] = (randvals[p] < 0.5f) ? -1 : 1;
                }
        }

        //! Apply the update to all cells of one color whose neighbor reads come from the frozen opposite color.
        void updateColor(bool black)
        {
            auto const worker = [&](long long iBegin, long long iEnd)
            {
                for(long long i = iBegin; i < iEnd; ++i)
                    for(long long c = 0; c < ny; ++c)
                    {
                        if(((i + c) % 2 == 0) != black)
                            continue;
                        long long const ip = (i + 1 < nx) ? i + 1 : 0;
                        long long const im = (i - 1 >= 0) ? i - 1 : nx - 1;
                        long long const cp = (c + 1 < ny) ? c + 1 : 0;
                        long long const cm = (c - 1 >= 0) ? c - 1 : ny - 1;

                        long long const nn = lattice[static_cast<std::size_t>(im * ny + c)]
                                             + lattice[static_cast<std::size_t>(ip * ny + c)]
                                             + lattice[static_cast<std::size_t>(i * ny + cm)]
                                             + lattice[static_cast<std::size_t>(i * ny + cp)];
                        signed char const lij = lattice[static_cast<std::size_t>(i * ny + c)];
                        double const acceptance = std::exp(-2.0 * invTemp * static_cast<double>(nn) * lij);
                        std::size_t const p = packedIndex(i, c, ny, black);
                        if(static_cast<double>(randvals[p]) < acceptance)
                            lattice[static_cast<std::size_t>(i * ny + c)] = static_cast<signed char>(-lij);
                    }
            };

            unsigned int const workers = static_cast<unsigned int>(
                std::min<long long>(static_cast<long long>(numThreads), std::max<long long>(nx, 1)));
            if(workers <= 1 || nx < 2)
            {
                worker(0, nx);
                return;
            }

            std::vector<std::thread> pool;
            pool.reserve(workers - 1);
            long long const chunk = nx / workers;
            long long const remainder = nx % workers;
            long long begin = 0;
            for(unsigned int t = 0; t < workers; ++t)
            {
                long long const count = chunk + (static_cast<long long>(t) < remainder ? 1 : 0);
                long long const end = begin + count;
                if(t + 1 == workers)
                    worker(begin, end);
                else
                    pool.emplace_back(worker, begin, end);
                begin = end;
            }
            for(auto& th : pool)
                th.join();
        }

        void sweep()
        {
            updateColor(true);
            updateColor(false);
        }

        void run(int steps)
        {
            for(int i = 0; i < steps; ++i)
                sweep();
        }

        //! Fetch the full-lattice spin of the cell that packed index `(i, j)` of the given color corresponds to.
        signed char packedSpin(long long i, long long j, bool black) const
        {
            long long const c = 2 * j + (black ? (i % 2) : (1 - (i % 2)));
            return lattice[static_cast<std::size_t>(i * ny + c)];
        }
    };

    /** Verify the two packed accelerator buffers against the independent checkerboard reference.
     *
     * The reference is recomputed from the same random buffer, so the check is self-contained. Because the lattice is
     * a discrete `{-1,+1}` state, the comparison is exact with no floating-point tolerance: any mismatch is a real
     * algorithmic divergence (wrong update order, wrong neighbor set or a different RNG stream), never rounding.
     *
     * A domain check rejects any lattice entry that is not exactly `+1` or `-1`, the meaningful non-finite /
     * corrupt-buffer check for an integer state (a stray byte from an uninitialized or partially copied buffer is
     * caught here instead of comparing unequal silently).
     *
     * @return EXIT_SUCCESS when both colors match the reference and the domain check passes, EXIT_FAILURE otherwise.
     */
    inline int verify(
        long long nx,
        long long ny,
        float alpha,
        std::vector<float> const& randvals,
        int steps,
        signed char const* blackOut,
        signed char const* whiteOut)
    {
        CheckerboardReference reference(nx, ny, alpha, randvals);
        reference.run(steps);

        long long const n2 = nx * (ny / 2);
        int error = EXIT_SUCCESS;

        long long invalid = 0;
        for(long long idx = 0; idx < n2; ++idx)
        {
            if((blackOut[idx] != 1 && blackOut[idx] != -1) || (whiteOut[idx] != 1 && whiteOut[idx] != -1))
            {
                if(invalid < 5)
                    printf(
                        "invalid spin at packed index %lld: black=%d white=%d\n",
                        idx,
                        static_cast<int>(blackOut[idx]),
                        static_cast<int>(whiteOut[idx]));
                error = EXIT_FAILURE;
                ++invalid;
            }
        }
        if(invalid != 0)
            printf("%lld lattice entries are outside {-1,+1}\n", invalid);

        long long failures = 0;
        for(long long i = 0; i < nx; ++i)
        {
            for(long long j = 0; j < ny / 2; ++j)
            {
                long long const idx = i * (ny / 2) + j;
                signed char const bRef = reference.packedSpin(i, j, true);
                signed char const wRef = reference.packedSpin(i, j, false);
                if(blackOut[idx] != bRef || whiteOut[idx] != wRef)
                {
                    if(failures < 5)
                        printf(
                            "mismatch at packed index %lld: black got %d expected %d, white got %d expected %d\n",
                            idx,
                            static_cast<int>(blackOut[idx]),
                            static_cast<int>(bRef),
                            static_cast<int>(whiteOut[idx]),
                            static_cast<int>(wRef));
                    error = EXIT_FAILURE;
                    ++failures;
                }
                if(failures >= 5)
                    break;
            }
            if(failures >= 5)
                break;
        }

        if(error == EXIT_SUCCESS)
            printf("Verification PASSED (%lld spins per color)\n", n2);
        else
            printf("Verification FAILED (%lld mismatches reported)\n", failures);
        return error;
    }

} // namespace ising
