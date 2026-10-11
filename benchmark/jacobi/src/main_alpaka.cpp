/* Copyright 2026 René Widera
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** @file Portable, single-source alpaka implementation of the HeCBench `jacobi` benchmark.
 *
 * The HeCBench reference solves the 2D Laplace equation on an `N x N` grid with a five-point Jacobi
 * stencil. The grid edges hold a fixed sinusoidal boundary condition, the interior starts at zero,
 * and the solver iterates until the normalised L2 change per step falls below `tol`. Every iteration
 * launches one stencil kernel that also reduces the local squared change over the block and adds it
 * to a global error scalar; the host then reads that scalar and checks the convergence criterion,
 * exactly like the reference which copies the error to the host each iteration.
 *
 * There is a single kernel source. The alpaka backend maps a frame to a host OpenMP thread or to a
 * CUDA thread block; only the launch configuration changes, never the algorithm. `N`, `tol`, the
 * iteration cap, the boundary conditions and the initial data match the pinned HeCBench source.
 */

#include "helper.hpp"
#include "misc.hpp"

#include <alpaka/alpaka.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <limits>
#include <vector>

namespace jacobiAlpaka
{
    //! Threads per block; 16x16 = 256 matches the HeCBench reference block.
    constexpr uint32_t tileX = 16u;
    constexpr uint32_t tileY = 16u;
    constexpr uint32_t blockSize = tileX * tileY;

    /** One five-point Jacobi relaxation step with a portable block reduction.
     *
     * Reads the interior of @p src, writes the new interior into @p dst and accumulates the squared
     * change into the single-element @p error buffer with an atomic add. Only interior points
     * `1..n-2` are updated; the boundary is fixed and comes from the initial data in @p dst.
     *
     * The field is a flat `n*n` buffer in row-major order, so the kernel indexes it linearly; that
     * keeps the comparison with the row-major host reference exact and avoids backend-dependent
     * multi-dimensional pitch conventions. Each block owns one `tileX x tileY` tile and each thread
     * maps its in-block id to one tile element, mirroring the reference's 16x16 tile. Tiles that
     * extend beyond the domain are clipped. The block reduction uses shared memory and
     * @c syncBlockThreads, so it is correct for any backend and any runtime block size (the host
     * executor maps a block to one thread).
     */
    struct JacobiStepKernel
    {
        ALPAKA_FN_ACC void operator()(
            auto const& acc,
            alpaka::concepts::IMdSpan auto dst,
            alpaka::concepts::IMdSpan auto src,
            alpaka::concepts::IMdSpan auto error,
            int n) const
        {
            using namespace alpaka;

            //! per-block shared reduction buffer
            auto& shared = onAcc::declareSharedVar<float[blockSize], uniqueId()>(acc);

            auto const numThreadsInBlock = acc.getExtentsOf(onAcc::origin::block, onAcc::unit::threads);
            uint32_t const numThreads = static_cast<uint32_t>(numThreadsInBlock.product());
            auto const threadIdx = acc.getIdxWithin(onAcc::origin::block, onAcc::unit::threads);
            uint32_t const tid = static_cast<uint32_t>(alpaka::linearize(numThreadsInBlock, threadIdx));

            uint32_t const tilesX = divCeil(static_cast<uint32_t>(n), tileX);
            uint32_t const tilesY = divCeil(static_cast<uint32_t>(n), tileY);
            uint32_t const numBlocks = tilesX * tilesY;
            int64_t const n64 = n;

            for(auto blockIdx : onAcc::makeIdxMap(acc, onAcc::worker::blocksInGrid, IdxRange{Vec{numBlocks}}))
            {
                int const bx = static_cast<int>(blockIdx.x() % tilesX);
                int const by = static_cast<int>(blockIdx.x() / tilesX);
                int const i0 = bx * static_cast<int>(tileX);
                int const j0 = by * static_cast<int>(tileY);

                float blockErr = 0.f;
                // 1D thread range; map each thread id to one tile element (row-major within the tile).
                for(auto localId :
                    onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{Vec{blockSize}}))
                {
                    uint32_t const lt = static_cast<uint32_t>(localId.x());
                    int const i = i0 + static_cast<int>(lt % tileX);
                    int const j = j0 + static_cast<int>(lt / tileX);
                    if(i >= 1 && i <= n - 2 && j >= 1 && j <= n - 2)
                    {
                        int64_t const k = static_cast<int64_t>(i) + static_cast<int64_t>(j) * n64;
                        float const v = 0.25f * (src[k - 1] + src[k + 1] + src[k - n64] + src[k + n64]);
                        float const df = v - src[k];
                        dst[k] = v;
                        blockErr += df * df;
                    }
                }

                // Portable block reduction over shared memory. The tree form works for any thread
                // count, not only powers of two.
                shared[tid] = blockErr;
                onAcc::syncBlockThreads(acc);
                for(uint32_t stride = 1u; stride < numThreads; stride <<= 1u)
                {
                    if((tid % (2u * stride)) == 0u && tid + stride < numThreads)
                        shared[tid] += shared[tid + stride];
                    onAcc::syncBlockThreads(acc);
                }
                if(tid == 0u)
                    alpaka::onAcc::atomicAdd(acc, error.data(), shared[0]);
                onAcc::syncBlockThreads(acc);
            }
        }
    };

    /** HeCBench initial condition written into a single-precision host buffer.
     *
     * Mirrors `initialize_data` of the pinned reference exactly, including the single-precision
     * `sinf` evaluation of the sinusoidal boundary. Both the current and the previous device field
     * are initialised this way so the fixed boundary is present in both buffers.
     */
    inline void initializeData(std::vector<float>& f, int n)
    {
        double const denom = static_cast<double>(n - 1);
        for(int j = 0; j < n; ++j)
        {
            for(int i = 0; i < n; ++i)
            {
                std::size_t const idx = static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * n;
                if(i == 0 || i == n - 1)
                    f[idx] = static_cast<float>(std::sin(j * 2.0 * M_PI / denom));
                else if(j == 0 || j == n - 1)
                    f[idx] = static_cast<float>(std::sin(i * 2.0 * M_PI / denom));
                else
                    f[idx] = 0.f;
            }
        }
    }

    /** Execute one full convergence solve and return its duration in seconds and its iteration count.
     *
     * The timed region is exactly the reference's: the `while (error > tol && iters < maxIters)` loop,
     * including the per-iteration error reduction and the device-to-host copy that the reference
     * performs synchronously. Initialisation and the final result copy are outside the timed region.
     */
    template<
        typename T_Queue,
        typename T_FrameSpec,
        typename T_BufferF,
        typename T_HostF,
        typename T_BufferE,
        typename T_HostE>
    double runSolve(
        T_Queue& queue,
        T_FrameSpec const& frameSpec,
        T_BufferF& d_f,
        T_BufferF& d_f_old,
        T_HostF const& h_init,
        T_BufferE& d_error,
        T_HostE& h_error,
        int n,
        float tolerance,
        int maxIters,
        int& itersOut)
    {
        using namespace alpaka;

        onHost::memcpy(queue, d_f, h_init);
        onHost::memcpy(queue, d_f_old, h_init);
        onHost::wait(queue);

        float error = std::numeric_limits<float>::max();
        int iters = 0;

        auto const start = std::chrono::steady_clock::now();
        while(error > tolerance && iters < maxIters)
        {
            onHost::memset(queue, d_error, 0x00);
            queue.enqueue(frameSpec, KernelBundle{JacobiStepKernel{}, d_f, d_f_old, d_error, n});

            // The reference swaps the two field buffers after every step.
            std::swap(d_f, d_f_old);

            // The reference copies the error scalar to the host synchronously each iteration.
            onHost::memcpy(queue, h_error, d_error);
            onHost::wait(queue);
            error = std::sqrt(h_error[0] / static_cast<float>(n) / static_cast<float>(n));

            ++iters;
        }
        auto const end = std::chrono::steady_clock::now();
        itersOut = iters;
        return std::chrono::duration<double>(end - start).count();
    }

    auto example(auto const deviceSpec, auto const exec, jacobi::Config const& cfg) -> int
    {
        using namespace alpaka;

        auto devSelector = onHost::makeDeviceSelector(deviceSpec);
        if(!devSelector.isAvailable())
        {
            std::cout << "No device available for " << deviceSpec.getName() << std::endl;
            return EXIT_SUCCESS;
        }
        onHost::Device device = devSelector.makeDevice(0);
        onHost::Queue queue = device.makeQueue(queueKind::nonBlocking);

        int const n = cfg.n;
        // The field is a flat 1D buffer in row-major order; the kernel indexes it linearly.
        Vec<int, 1u> const extent{static_cast<int>(static_cast<std::size_t>(n) * n)};

        std::cout << "Grid side length: " << n << std::endl;
        std::cout << "Tolerance: " << cfg.tolerance << std::endl;
        std::cout << "Iteration cap: " << cfg.maxIters << std::endl;
        std::cout << "Measured solves: " << cfg.repeat << std::endl;
        std::cout << "Using alpaka accelerator " << onHost::demangledName(exec) << " for "
                  << deviceSpec.getApi().getName() << " " << deviceSpec.getDeviceKind().getName() << std::endl;

        auto hInit = onHost::allocHost<float>(extent);
        {
            std::vector<float> tmp(static_cast<std::size_t>(n) * n);
            initializeData(tmp, n);
            for(std::size_t k = 0; k < tmp.size(); ++k)
                hInit.data()[k] = tmp[k];
        }
        auto hResult = onHost::allocHost<float>(extent);
        auto hError = onHost::allocHost<float>(1);

        auto d_f = onHost::alloc<float>(device, extent);
        auto d_f_old = onHost::alloc<float>(device, extent);
        auto d_error = onHost::alloc<float>(device, 1);

        // One logical frame per 16x16 tile; the kernel computes the tile grid from n. A logical frame
        // maps to one physical block on the GPU and to one OpenMP thread (one block per thread) on
        // the host.
        uint32_t const numFrames
            = divCeil(static_cast<uint32_t>(n), tileX) * divCeil(static_cast<uint32_t>(n), tileY);
        onHost::FrameSpec const frameSpec{Vec{numFrames}, Vec{blockSize}, exec};
        std::cout << "Launch frames=" << numFrames << " frameExtent=" << blockSize << std::endl;

        // Warm-up solve (untimed), then the measured independent solves.
        int warmupIters = 0;
        double const warmup = runSolve(
            queue,
            frameSpec,
            d_f,
            d_f_old,
            hInit,
            d_error,
            hError,
            n,
            cfg.tolerance,
            cfg.maxIters,
            warmupIters);
        std::cout << "Warm-up: " << warmupIters << " iterations, " << warmup << " s\n";

        std::vector<double> samples;
        int iters = 0;
        for(int r = 0; r < cfg.repeat; ++r)
        {
            double const seconds = runSolve(
                queue,
                frameSpec,
                d_f,
                d_f_old,
                hInit,
                d_error,
                hError,
                n,
                cfg.tolerance,
                cfg.maxIters,
                iters);
            samples.push_back(seconds);
            std::cout << "Solve " << r << ": " << iters << " iterations, " << seconds << " s\n";
        }

        // --- report median + variation of the per-solve time ---
        std::vector<double> sorted = samples;
        std::sort(sorted.begin(), sorted.end());
        double const median = sorted[sorted.size() / 2];
        double const minS = sorted.front();
        double const maxS = sorted.back();
        double const variationPct = median > 0.0 ? (maxS - minS) / median * 100.0 : 0.0;
        double const perIterationUs = median / static_cast<double>(iters) * 1e6;
        std::cout.precision(6);
        std::cout << std::fixed;
        std::cout << "jacobi_result backend=" << onHost::demangledName(exec) << " n=" << n
                  << " iters=" << iters << " repeat=" << cfg.repeat << " median_s=" << median
                  << " variation_pct=" << variationPct << " per_iteration_us=" << perIterationUs << std::endl;

        // --- validation against the independent double-precision host reference ---
        // After the loop the newest field is in d_f_old (the kernel writes d_f then the buffers swap).
        onHost::memcpy(queue, hResult, d_f_old);
        onHost::wait(queue);
        std::vector<float> result(static_cast<std::size_t>(n) * n);
        for(std::size_t k = 0; k < result.size(); ++k)
            result[k] = hResult.data()[k];
        return jacobi::verify(n, iters, result);
    }
} // namespace jacobiAlpaka

auto main(int argc, char* argv[]) -> int
{
    jacobi::Config cfg;
    if(int const ret = jacobi::parseCmd(argc, argv, cfg))
        return ret;

    using namespace alpaka;

    return onHost::executeForEachIfHasDevice(
        [=](auto const& backend)
        { return jacobiAlpaka::example(onHost::makeDeviceSpec(backend), getExecutor(backend), cfg); },
        onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors));
}
