/* Copyright 2026 René Widera
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** @file Portable, single-source alpaka implementation of the HeCBench `heat2d` benchmark.
 *
 * The HeCBench reference advances the 2D explicit heat equation with one five-point Laplacian
 * kernel on a fully periodic `Lx x Ly` grid:
 *
 *     out[y*Lx + x] = norm * in[y*Lx + x]
 *                     + delta * ( in[y*Lx + (x+1)%Lx] + in[y*Lx + (Lx+x-1)%Lx]
 *                                 + in[((y+1)%Ly)*Lx + x] + in[((Ly+y-1)%Ly)*Lx + x] )
 *
 * with `delta = sigma/(1+4*sigma)`, `norm = 1/(1+4*sigma)` and `sigma = 0.01`. There is no halo:
 * the grid wraps in both directions. The algorithm is one kernel, iterated `niter` times with the
 * two field buffers swapped after every step.
 *
 * There is a single kernel source. The alpaka backend maps a frame to a host OpenMP thread or to a
 * CUDA thread; only the launch configuration changes, never the algorithm. The field is a flat 1D
 * buffer in row-major order so the linear indexing is independent of any backend pitch convention.
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

namespace heat2dAlpaka
{
    //! Threads per frame; 256 is a portable block size for the host and the GPU backends.
    constexpr uint32_t frameExtent = 256u;

    /** One explicit heat-equation step with periodic boundary conditions.
     *
     * Each thread owns one cell of the flat row-major buffer. It reads the four wrapped neighbours,
     * computes the new value with the same single-precision coefficients and arithmetic order as the
     * pinned reference and writes it to the destination buffer. Because reads and writes use separate
     * buffers there is no aliasing, exactly like the reference's `out`/`in` pair.
     */
    struct HeatStepKernel
    {
        ALPAKA_FN_ACC void operator()(
            auto const& acc,
            alpaka::concepts::IMdSpan auto out,
            alpaka::concepts::IMdSpan auto in,
            float delta,
            float norm,
            int Lx,
            int Ly) const
        {
            using namespace alpaka;

            // `int` indexing matches the pinned reference and keeps the index arithmetic on 32-bit
            // integers. `Lx*Ly` must fit in `int` (validated by the caller). The four wrapped
            // neighbours are derived relative to the linear centre index to avoid re-deriving `y*Lx`
            // for every neighbour; the wrap conditions are the same as the reference's `%`.
            int const total = Lx * Ly;
            for(auto idx : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{Vec{total}}))
            {
                int const v00 = idx.x();
                int const x = v00 % Lx;
                int const y = v00 / Lx;
                int const v0p = (x + 1 == Lx) ? v00 - (Lx - 1) : v00 + 1;
                int const v0m = (x == 0) ? v00 + (Lx - 1) : v00 - 1;
                int const vp0 = (y + 1 == Ly) ? x : v00 + Lx;
                int const vm0 = (y == 0) ? v00 + (Ly - 1) * Lx : v00 - Lx;
                out[v00] = norm * in[v00] + delta * (in[v0p] + in[v0m] + in[vp0] + in[vm0]);
            }
        }
    };

    /** Run `iters` back-to-back kernel launches between two host timestamps and return seconds.
     *
     * This reproduces the HeCBench reference timing scope exactly: the reference enqueues `niter`
     * kernels, each reading the current buffer and swapping, synchronises once at the end and divides
     * the wall time by `niter`. Here the measured region also contains the host-side buffer-handle
     * swap, which is negligible compared with a launch. Both device buffers are reset to the initial
     * field before the region so every measured run starts from the identical state, like a fresh
     * reference invocation.
     */
    template<typename T_Queue, typename T_Launch, typename T_Buffer, typename T_Host>
    double runTimed(
        T_Queue& queue,
        T_Launch const& launch,
        T_Buffer& bufA,
        T_Buffer& bufB,
        T_Host const& hInit,
        float delta,
        float norm,
        int Lx,
        int Ly,
        int iters)
    {
        using namespace alpaka;
        onHost::memcpy(queue, bufA, hInit);
        onHost::memcpy(queue, bufB, hInit);
        onHost::wait(queue);

        auto const start = std::chrono::steady_clock::now();
        for(int i = 0; i < iters; ++i)
        {
            queue.enqueue(launch, KernelBundle{HeatStepKernel{}, bufB, bufA, delta, norm, Lx, Ly});
            std::swap(bufA, bufB);
        }
        onHost::wait(queue);
        auto const end = std::chrono::steady_clock::now();
        return std::chrono::duration<double>(end - start).count();
    }

    auto example(auto const deviceSpec, auto const exec, heat2d::Config const& cfg) -> int
    {
        using namespace alpaka;

        auto devSelector = onHost::makeDeviceSelector(deviceSpec);
        if(!devSelector.isAvailable())
        {
            std::cout << "No device available for " << deviceSpec.getName() << std::endl;
            return EXIT_SUCCESS;
        }
        onHost::Device device = devSelector.makeDevice(0);
        // Non-blocking queue: enqueue returns before the kernel finishes so the whole iteration loop
        // is submitted back-to-back and synchronised once, exactly like the reference's timed region.
        onHost::Queue queue = device.makeQueue(queueKind::nonBlocking);

        int const Lx = cfg.lx;
        int const Ly = cfg.ly;
        int const iters = cfg.iters;
        // The kernel and the reference index with `int`; reject sizes whose flat index would overflow.
        if(static_cast<int64_t>(Lx) * static_cast<int64_t>(Ly) > static_cast<int64_t>(std::numeric_limits<int>::max()))
        {
            std::cerr << "Error: Lx*Ly exceeds INT_MAX and is not supported.\n";
            return EXIT_FAILURE;
        }
        int const total = Lx * Ly;
        Vec<int, 1u> const extent{total};

        // Coefficients in single precision, exactly as in the pinned reference.
        float const delta = heat2d::sigma / (1.f + 4.f * heat2d::sigma);
        float const norm = 1.f / (1.f + 4.f * heat2d::sigma);

        std::cout << "Lx: " << Lx << std::endl;
        std::cout << "Ly: " << Ly << std::endl;
        std::cout << "Iterations: " << iters << std::endl;
        std::cout << "Measured runs: " << cfg.repeat << std::endl;
        std::cout << "Using alpaka accelerator " << onHost::demangledName(exec) << " for "
                  << deviceSpec.getApi().getName() << " " << deviceSpec.getDeviceKind().getName() << std::endl;

        std::vector<float> const init = heat2d::makeInitialField(Lx, Ly);

        auto hInit = onHost::allocHost<float>(extent);
        for(std::size_t k = 0; k < init.size(); ++k)
            hInit.data()[k] = init[k];
        auto hResult = onHost::allocHost<float>(extent);

        auto dA = onHost::alloc<float>(device, extent);
        auto dB = onHost::alloc<float>(device, extent);

        // One frame per `frameExtent` cells; the kernel walks the whole grid with threadsInGrid.
        uint32_t const numFrames = divCeil(static_cast<uint32_t>(total), frameExtent);
        onHost::FrameSpec const launch{Vec{numFrames}, Vec{frameExtent}, exec};
        std::cout << "Launch frames=" << numFrames << " frameExtent=" << frameExtent << std::endl;

        // Warm-up (untimed), then the measured independent runs.
        double const warmup = runTimed(queue, launch, dA, dB, hInit, delta, norm, Lx, Ly, iters);
        std::cout << "Warm-up: " << iters << " iterations, " << warmup << " s\n";

        std::vector<double> samples;
        samples.reserve(static_cast<std::size_t>(cfg.repeat));
        for(int r = 0; r < cfg.repeat; ++r)
        {
            double const seconds = runTimed(queue, launch, dA, dB, hInit, delta, norm, Lx, Ly, iters);
            samples.push_back(seconds);
            std::cout << "Run " << r << ": " << iters << " iterations, " << seconds << " s\n";
        }

        std::vector<double> sorted = samples;
        std::sort(sorted.begin(), sorted.end());
        double const median = sorted[sorted.size() / 2];
        double const minS = sorted.front();
        double const maxS = sorted.back();
        double const variationPct = median > 0.0 ? (maxS - minS) / median * 100.0 : 0.0;
        double const perIterationUs = median / static_cast<double>(iters) * 1e6;
        std::cout.precision(6);
        std::cout << std::fixed;
        std::cout << "heat2d_result backend=" << onHost::demangledName(exec) << " Lx=" << Lx << " Ly=" << Ly
                  << " iters=" << iters << " repeat=" << cfg.repeat << " median_s=" << median
                  << " variation_pct=" << variationPct << " per_iteration_us=" << perIterationUs << std::endl;

        // After the final swap the newest field is in dA.
        onHost::memcpy(queue, hResult, dA);
        onHost::wait(queue);
        std::vector<float> result(static_cast<std::size_t>(total));
        for(std::size_t k = 0; k < result.size(); ++k)
            result[k] = hResult.data()[k];
        return heat2d::verify(init, Lx, Ly, iters, result);
    }
} // namespace heat2dAlpaka

auto main(int argc, char* argv[]) -> int
{
    heat2d::Config cfg;
    if(int const ret = heat2d::parseCmd(argc, argv, cfg))
        return ret;

    using namespace alpaka;

    return onHost::executeForEachIfHasDevice(
        [=](auto const& backend)
        { return heat2dAlpaka::example(onHost::makeDeviceSpec(backend), getExecutor(backend), cfg); },
        onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors));
}
