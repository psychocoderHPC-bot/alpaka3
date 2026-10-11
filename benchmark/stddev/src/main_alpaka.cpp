/* Copyright (c) 2018-2020, NVIDIA CORPORATION.
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** @file Portable, single-source alpaka implementation of the HeCBench `stddev` benchmark.
 *
 * The pinned HeCBench reference (`src/stddev-cuda/main.cu`, derived from NVIDIA RAPIDS `raft`)
 * computes, for every column `c` of an `N x D` row-major `float` matrix,
 *
 *     std[c] = sqrt( sum_{r=0}^{N-1} data[r*D + c]^2 / sampleSize )
 *
 * with `sampleSize = N - 1` (the reference always passes `sample = true`). The device pipeline has
 * two kernels: `sopKernel` accumulates the per-column sum of squares with thread-block and global
 * atomics, and `sampleKernel` normalises by `sampleSize` and takes the square root. `main` zeroes the
 * output with `cudaMemset` before every launch because the accumulation uses atomics.
 *
 * This port keeps the exact computation, the exact `float` value type, the two-stage pipeline
 * (accumulate sum of squares, then normalise and take the square root), the atomic accumulation into
 * the per-column output and the `cudaMemset` equivalent (`onHost::memset` to zero) between launches.
 * A single kernel source runs on the host CPU and on CUDA GPUs; only the launch configuration
 * changes. There is no native CUDA/HIP/OpenMP kernel and no backend branch.
 *
 * Grid choice: the reference uses a 2D grid that splits both columns (fast) and rows (slow) so that
 * a warp reads 32 consecutive columns and the total thread count stays high even for a narrow matrix.
 * This port reproduces that property with a flat worker index `i = s*D + c`: consecutive workers own
 * consecutive columns (coalesced 128-byte warp transactions) and the row dimension is split into
 * `rowSplits` chunks so that `D * rowSplits` stays near a target worker count regardless of `D`. Each
 * worker accumulates the sum of squares of its row chunk with four independent accumulators and adds
 * the partial into `std[c]` with a device-scope atomic. This works on the host CPU executors as well,
 * because alpaka maps host atomics to `std::atomic_ref`. The accumulation order differs from the
 * reference, which is why the comparison uses the reference's own absolute tolerance (see `helper.hpp`).
 */

#include "helper.hpp"
#include "misc.hpp"

#include <alpaka/alpaka.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace stddevAlpaka
{
    //! Threads (logical workers) per frame. Mirrors the reference `TPB = 256`.
    constexpr uint32_t frameExtent = 256u;
    //! Target number of logical workers; the row dimension is split until `D * rowSplits` reaches it.
    constexpr uint32_t targetWorkers = 1'048'576u;
    //! Minimal rows a single worker processes; bounds the row split for very large column counts.
    constexpr uint32_t minRowsPerWorker = 8u;

    /** Pass 1: per-column sum of squares, accumulated with device-scope atomics.
     *
     * The flat worker index maps to `(column = i % D, split = i / D)`. Consecutive workers own
     * consecutive columns, so a warp reads contiguous memory. Worker `(c, s)` covers rows
     * `[s*rowsPerWorker, min(N, (s+1)*rowsPerWorker))`; the row loop is unrolled by four independent
     * accumulators. The partial is added into `std[c]`, so the output must be zeroed beforehand.
     */
    struct SumOfProductsKernel
    {
        ALPAKA_FN_ACC void operator()(
            auto const& acc,
            alpaka::concepts::IMdSpan auto std,
            alpaka::concepts::IDataSource auto const& data,
            uint32_t D,
            uint32_t N,
            uint32_t rowsPerWorker,
            uint32_t rowSplits) const
        {
            using namespace alpaka;

            uint32_t const total = D * rowSplits;
            for(auto workerIdx :
                onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{Vec<uint32_t, 1u>{total}}))
            {
                uint32_t const i = workerIdx.x();
                uint32_t const c = i % D;
                uint32_t const s = i / D;

                uint32_t const rowBegin = s * rowsPerWorker;
                uint32_t const rowEnd = (rowBegin + rowsPerWorker < N) ? (rowBegin + rowsPerWorker) : N;

                float a0 = 0.0f;
                float a1 = 0.0f;
                float a2 = 0.0f;
                float a3 = 0.0f;

                uint32_t r = rowBegin;
                for(; r + 4u <= rowEnd; r += 4u)
                {
                    float const v0 = data[static_cast<uint64_t>(r) * D + c];
                    float const v1 = data[static_cast<uint64_t>(r + 1u) * D + c];
                    float const v2 = data[static_cast<uint64_t>(r + 2u) * D + c];
                    float const v3 = data[static_cast<uint64_t>(r + 3u) * D + c];
                    a0 += v0 * v0;
                    a1 += v1 * v1;
                    a2 += v2 * v2;
                    a3 += v3 * v3;
                }
                for(; r < rowEnd; ++r)
                {
                    float const v = data[static_cast<uint64_t>(r) * D + c];
                    a0 += v * v;
                }

                onAcc::atomicAdd(acc, &std[c], (a0 + a1) + (a2 + a3));
            }
        }
    };

    /** Pass 2: normalise the per-column sum of squares and take the square root. */
    struct NormalizeKernel
    {
        ALPAKA_FN_ACC void operator()(
            auto const& acc,
            alpaka::concepts::IMdSpan auto std,
            uint32_t D,
            uint32_t sampleSize) const
        {
            using namespace alpaka;

            float const inv = 1.0f / static_cast<float>(sampleSize);
            for(auto idx : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{Vec<uint32_t, 1u>{D}}))
            {
                uint32_t const c = idx.x();
                std[c] = math::sqrt(std[c] * inv);
            }
        }
    };

    /** Choose the row split so `D * rowSplits` approaches @ref targetWorkers without any worker
     * processing fewer than @ref minRowsPerWorker rows.
     */
    inline uint32_t chooseRowSplits(uint32_t D, uint32_t N)
    {
        uint32_t const desired = alpaka::divCeil(targetWorkers, D);
        uint32_t const maxByRows = std::max(1u, N / minRowsPerWorker);
        return std::min(std::max(1u, desired), maxByRows);
    }

    /** Run one configuration and return the exit code. */
    auto run(auto const deviceSpec, auto const exec, stddevBench::Params const& params) -> int
    {
        using namespace alpaka;

        uint32_t const D = params.D;
        uint32_t const N = params.N;
        uint32_t const sampleSize = N - 1u;
        uint64_t const flatSize64 = static_cast<uint64_t>(D) * N;
        if(flatSize64 > 0xFFFF'FFFFull)
        {
            std::cerr << "Error: D*N = " << flatSize64 << " exceeds the 32-bit element extent supported here.\n";
            return EXIT_FAILURE;
        }
        uint32_t const flatSize = static_cast<uint32_t>(flatSize64);
        uint32_t const rowSplits = chooseRowSplits(D, N);
        uint32_t const rowsPerWorker = alpaka::divCeil(N, rowSplits);

        std::cout << "Columns (D): " << D << std::endl;
        std::cout << "Rows (N): " << N << std::endl;
        std::cout << "Repeat: " << params.repeat << std::endl;
        std::cout << "Samples: " << params.samples << std::endl;
        std::cout << "Element type float, sample stddev (divide by N-1 = " << sampleSize << ")" << std::endl;
        std::cout << "Row splits: " << rowSplits << " (rows per worker " << rowsPerWorker << ", workers "
                  << static_cast<uint64_t>(D) * rowSplits << ")" << std::endl;
        std::cout << "Using alpaka accelerator " << onHost::demangledName(exec) << " for "
                  << deviceSpec.getApi().getName() << " " << deviceSpec.getDeviceKind().getName() << std::endl;

        auto devSelector = onHost::makeDeviceSelector(deviceSpec);
        onHost::Device device = devSelector.makeDevice(0);
        onHost::Queue queue = device.makeQueue(queueKind::blocking);

        auto hData = onHost::allocHost<float>(Vec<uint32_t, 1u>{flatSize});
        auto hResult = onHost::allocHost<float>(Vec<uint32_t, 1u>{D});
        std::vector<double> hReference(static_cast<size_t>(D));

        stddevBench::generateInput(hData.data(), static_cast<size_t>(flatSize));

        auto dData = onHost::alloc<float>(device, Vec<uint32_t, 1u>{flatSize});
        auto dStd = onHost::alloc<float>(device, Vec<uint32_t, 1u>{D});

        onHost::memcpy(queue, dData, hData);
        onHost::wait(queue);

        uint32_t const totalWorkers = D * rowSplits;
        uint32_t const sopFrames = alpaka::divCeil(totalWorkers, frameExtent);
        onHost::FrameSpec const sopFrameSpec{Vec<uint32_t, 1u>{sopFrames}, Vec<uint32_t, 1u>{frameExtent}, exec};
        uint32_t const normFrames = alpaka::divCeil(D, frameExtent);
        onHost::FrameSpec const normFrameSpec{Vec<uint32_t, 1u>{normFrames}, Vec<uint32_t, 1u>{frameExtent}, exec};

        SumOfProductsKernel sopKernel;
        NormalizeKernel normalizeKernel;

        auto runOnce = [&]()
        {
            // the atomic accumulation requires a zeroed output for every launch.
            onHost::memset(queue, dStd, 0);
            queue.enqueue(sopFrameSpec, KernelBundle{sopKernel, dStd, dData, D, N, rowsPerWorker, rowSplits});
            queue.enqueue(normFrameSpec, KernelBundle{normalizeKernel, dStd, D, sampleSize});
        };

        // Warm-up (also produces the buffer that is validated below).
        runOnce();
        onHost::wait(queue);

        std::vector<double> samples;
        samples.reserve(static_cast<size_t>(params.samples));
        for(int s = 0; s < params.samples; ++s)
        {
            auto const start = std::chrono::steady_clock::now();
            for(int i = 0; i < params.repeat; ++i)
                runOnce();
            onHost::wait(queue);
            auto const end = std::chrono::steady_clock::now();
            double const secondsTotal
                = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count())
                  * 1e-9;
            double const perPipeline = secondsTotal / static_cast<double>(params.repeat);
            samples.push_back(perPipeline);
            std::printf("sample %d: average stddev pipeline time %.9f (s)\n", s, perPipeline);
        }

        std::vector<double> sorted = samples;
        std::sort(sorted.begin(), sorted.end());
        double const median = sorted[sorted.size() / 2u];
        double const min = sorted.front();
        double const max = sorted.back();
        double const spread = (median > 0.0) ? (max - min) / median : 0.0;
        std::printf(
            "Average execution time of stddev kernels (median of %d): %.9f (s)  min %.9f  max %.9f  spread %.3f\n",
            params.samples,
            median,
            min,
            max,
            spread);

        onHost::memcpy(queue, hResult, dStd);
        onHost::wait(queue);

        int const refRc = stddevBench::hostReference(hReference.data(), hData.data(), D, N, true);
        if(refRc != EXIT_SUCCESS)
            return refRc;

        int const verifyRc = stddevBench::verify(D, hResult.data(), hReference.data());
        std::printf("%s\n", verifyRc == EXIT_SUCCESS ? "PASS" : "FAIL");
        return verifyRc;
    }

    auto example(auto const deviceSpec, auto const exec, stddevBench::Params const& params) -> int
    {
        return run(deviceSpec, exec, params);
    }
} // namespace stddevAlpaka

auto main(int argc, char* argv[]) -> int
{
    stddevBench::Params params;
    if(int const ret = stddevBench::parseCmd(argc, argv, params))
        return ret;

    using namespace alpaka;

    /* Execute the example once for each backend (device specification + executor). */
    return onHost::executeForEachIfHasDevice(
        [&](auto const& backend)
        { return stddevAlpaka::example(onHost::makeDeviceSpec(backend), getExecutor(backend), params); },
        onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors));
}
