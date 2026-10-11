/* Copyright 2026 René Widera
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** @file Portable, single-source alpaka implementation of the HeCBench `laplace3d` benchmark.
 *
 * The HeCBench reference performs `niter` Jacobi iterations of the seven-point Laplacian stencil on
 * a regular `Nx x Ny x Nz` grid. The field is row-major with x innermost (`ind = i + j*Nx + k*Nx*Ny`).
 * Every boundary cell is a Dirichlet cell and keeps its value; every interior cell is replaced by
 * the average of its six axis neighbours:
 *
 *     out[ind] = ( in[ind-1] + in[ind+1]
 *                + in[ind-Nx] + in[ind+Nx]
 *                + in[ind-Nx*Ny] + in[ind+Nx*Ny] ) * (1/6)
 *
 * The two field buffers are swapped after every step, exactly like the reference's `d_u1`/`d_u2`
 * ping-pong. There is a single kernel source. The alpaka backend maps a frame to a host OpenMP
 * thread or to a CUDA thread; only the launch configuration changes, never the algorithm. The field
 * is a flat 1D buffer in row-major order so the linear indexing is independent of any backend pitch
 * convention.
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

namespace laplace3dAlpaka
{
    //! Threads per frame; 256 is a portable block size for the host and the GPU backends.
    constexpr uint32_t frameExtent = 256u;

    /** One seven-point Jacobi step with Dirichlet boundaries.
     *
     * Each thread owns one cell of the flat row-major buffer. It reads the six axis neighbours,
     * applies the same single-precision `1/6` factor as the pinned reference and writes the result
     * to the destination buffer. Because reads and writes use separate buffers there is no aliasing,
     * exactly like the reference's `u1`/`u2` pair.
     */
    struct LapStepKernel
    {
        ALPAKA_FN_ACC void operator()(
            auto const& acc,
            alpaka::concepts::IMdSpan auto out,
            alpaka::concepts::IMdSpan auto in,
            int Nx,
            int Ny,
            int Nz) const
        {
            using namespace alpaka;

            // `int` indexing matches the pinned reference and keeps the index arithmetic on 32-bit
            // integers. `Nx*Ny*Nz` must fit in `int` (validated by the caller).
            int const total = Nx * Ny * Nz;
            int const strideJ = Nx;
            int const strideK = Nx * Ny;
            float const sixth = 1.f / 6.f;
            for(auto idx : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{Vec{total}}))
            {
                int const v = idx.x();
                int const k = v / strideK;
                int const rem = v - k * strideK;
                int const j = rem / strideJ;
                int const i = rem - j * strideJ;
                if(i == 0 || i == Nx - 1 || j == 0 || j == Ny - 1 || k == 0 || k == Nz - 1)
                    out[v] = in[v]; // Dirichlet boundary: keep the value
                else
                    out[v] = (in[v - 1] + in[v + 1] + in[v - strideJ] + in[v + strideJ] + in[v - strideK]
                              + in[v + strideK])
                             * sixth;
            }
        }
    };

    /** Run `iters` back-to-back kernel launches between two host timestamps and return seconds.
     *
     * This reproduces the HeCBench reference timing scope exactly: the reference enqueues `niter`
     * kernels, each reading the current buffer and swapping, synchronises once at the end and
     * divides the wall time by `niter`. Both device buffers are reset to the initial field before
     * the region so every measured run starts from the identical state, like a fresh reference
     * invocation.
     */
    template<typename T_Queue, typename T_Launch, typename T_Buffer, typename T_Host>
    double runTimed(
        T_Queue& queue,
        T_Launch const& launch,
        T_Buffer& bufA,
        T_Buffer& bufB,
        T_Host const& hInit,
        int Nx,
        int Ny,
        int Nz,
        int iters)
    {
        using namespace alpaka;
        onHost::memcpy(queue, bufA, hInit);
        onHost::memcpy(queue, bufB, hInit);
        onHost::wait(queue);

        auto const start = std::chrono::steady_clock::now();
        for(int i = 0; i < iters; ++i)
        {
            queue.enqueue(launch, KernelBundle{LapStepKernel{}, bufB, bufA, Nx, Ny, Nz});
            std::swap(bufA, bufB);
        }
        onHost::wait(queue);
        auto const end = std::chrono::steady_clock::now();
        return std::chrono::duration<double>(end - start).count();
    }

    auto example(auto const deviceSpec, auto const exec, laplace3d::Config const& cfg) -> int
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

        int const Nx = cfg.nx;
        int const Ny = cfg.ny;
        int const Nz = cfg.nz;
        int const iters = cfg.iters;
        // The kernel and the reference index with `int`; reject sizes whose flat index would overflow.
        if(static_cast<int64_t>(Nx) * static_cast<int64_t>(Ny) * static_cast<int64_t>(Nz)
           > static_cast<int64_t>(std::numeric_limits<int>::max()))
        {
            std::cerr << "Error: Nx*Ny*Nz exceeds INT_MAX and is not supported.\n";
            return EXIT_FAILURE;
        }
        int const total = Nx * Ny * Nz;
        Vec<int, 1u> const extent{total};

        std::cout << "Nx: " << Nx << std::endl;
        std::cout << "Ny: " << Ny << std::endl;
        std::cout << "Nz: " << Nz << std::endl;
        std::cout << "Iterations: " << iters << std::endl;
        std::cout << "Measured runs: " << cfg.repeat << std::endl;
        std::cout << "Using alpaka accelerator " << onHost::demangledName(exec) << " for "
                  << deviceSpec.getApi().getName() << " " << deviceSpec.getDeviceKind().getName() << std::endl;

        std::vector<float> const init = laplace3d::makeInitialField(Nx, Ny, Nz);

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
        double const warmup = runTimed(queue, launch, dA, dB, hInit, Nx, Ny, Nz, iters);
        std::cout << "Warm-up: " << iters << " iterations, " << warmup << " s\n";

        std::vector<double> samples;
        samples.reserve(static_cast<std::size_t>(cfg.repeat));
        for(int r = 0; r < cfg.repeat; ++r)
        {
            double const seconds = runTimed(queue, launch, dA, dB, hInit, Nx, Ny, Nz, iters);
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
        std::cout << "laplace3d_result backend=" << onHost::demangledName(exec) << " Nx=" << Nx << " Ny=" << Ny
                  << " Nz=" << Nz << " iters=" << iters << " repeat=" << cfg.repeat << " median_s=" << median
                  << " variation_pct=" << variationPct << " per_iteration_us=" << perIterationUs << std::endl;

        // After the final swap the newest field is in dA.
        onHost::memcpy(queue, hResult, dA);
        onHost::wait(queue);
        std::vector<float> result(static_cast<std::size_t>(total));
        for(std::size_t k = 0; k < result.size(); ++k)
            result[k] = hResult.data()[k];
        return laplace3d::verify(init, Nx, Ny, Nz, iters, result);
    }
} // namespace laplace3dAlpaka

auto main(int argc, char* argv[]) -> int
{
    laplace3d::Config cfg;
    if(int const ret = laplace3d::parseCmd(argc, argv, cfg))
        return ret;

    using namespace alpaka;

    return onHost::executeForEachIfHasDevice(
        [=](auto const& backend)
        { return laplace3dAlpaka::example(onHost::makeDeviceSpec(backend), getExecutor(backend), cfg); },
        onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors));
}
