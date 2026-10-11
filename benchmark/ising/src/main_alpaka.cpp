/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 * Copyright 2026 Rene Widera
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** @file Portable, single-source alpaka implementation of the HeCBench `ising` benchmark.
 *
 * The benchmark runs a 2D square-lattice Ising Monte-Carlo simulation with a checkerboard (black/white) update and a
 * periodic boundary. The exact same kernel source is compiled for the host CPU and for CUDA GPUs; there is no backend
 * branch, no native CUDA/HIP/OpenMP kernel and no separate host/device algorithm. The lattice, the random sequence,
 * the update rule, the iteration counts and the verification match the pinned HeCBench sources
 * `src/ising-cuda/main.cu` and `src/ising-omp/main.cpp`.
 */

#include "helper.hpp"
#include "misc.hpp"

#include <alpaka/alpaka.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <vector>

namespace isingAlpaka
{
    /** Initialize the packed spins from the random buffer.
     *
     * The packed lattice holds `nx * (ny/2)` spins. Each element is `+1` if its random value is `>= 0.5`, otherwise
     * `-1`, matching `init_spins` of the HeCBench reference.
     */
    struct InitKernel
    {
        ALPAKA_FN_ACC void operator()(
            auto const& acc,
            alpaka::concepts::IMdSpan auto lattice,
            alpaka::concepts::IMdSpan auto randvals,
            int64_t count) const
        {
            using namespace alpaka;
            using element_type = alpaka::GetValueType_t<ALPAKA_TYPEOF(lattice)>;

            for(auto tid : onAcc::makeIdxMap(acc, onAcc::worker::linearThreadsInGrid, IdxRange{count}))
                lattice[tid] = static_cast<element_type>((randvals[tid] < 0.5f) ? -1 : 1);
        }
    };

    /** Update one checkerboard color of the packed lattice.
     *
     * `isBlack` selects which color is updated. The neighbor selection is exactly the one of the HeCBench
     * `update_lattice` reference: the row above/below are read from the opposite-color lattice, and the off-column
     * neighbor `joff` is chosen from the row parity and the color.
     *
     * The acceptance ratio is `exp(-2 * invTemp * nnSum * lij)` evaluated with the backend's single-precision
     * exponential (`expf` on CUDA, `std::exp(float)` on the host), matching the reference so both make identical
     * decisions on identical inputs.
     */
    template<bool isBlack>
    struct UpdateKernel
    {
        ALPAKA_FN_ACC void operator()(
            auto const& acc,
            alpaka::concepts::IMdSpan auto lattice,
            alpaka::concepts::IMdSpan auto opLattice,
            alpaka::concepts::IMdSpan auto randvals,
            float invTemp,
            int64_t nx64,
            int64_t nyHalf64) const
        {
            using namespace alpaka;
            using element_type = alpaka::GetValueType_t<ALPAKA_TYPEOF(lattice)>;

            // 32-bit indices: the packed lattice has at most a few 10^8 entries for the sizes used here, so 32-bit
            // division/multiply is sufficient and matches the reference's `int` index arithmetic.
            int32_t const nx = static_cast<int32_t>(nx64);
            int32_t const nyHalf = static_cast<int32_t>(nyHalf64);
            int32_t const count = nx * nyHalf;

            for(auto tid : onAcc::makeIdxMap(acc, onAcc::worker::linearThreadsInGrid, IdxRange{count}))
            {
                int32_t const t = tid.x();
                int32_t const i = t / nyHalf;
                int32_t const j = t % nyHalf;

                // periodic stencil in the row direction
                int32_t const ipp = (i + 1 < nx) ? i + 1 : 0;
                int32_t const inn = (i - 1 >= 0) ? i - 1 : nx - 1;
                // periodic stencil in the packed column direction
                int32_t const jpp = (j + 1 < nyHalf) ? j + 1 : 0;
                int32_t const jnn = (j - 1 >= 0) ? j - 1 : nyHalf - 1;
                int32_t const joff = isBlack ? ((i % 2) ? jpp : jnn) : ((i % 2) ? jnn : jpp);

                // The four neighbor spins are in {-1,+1}, so the sum is in [-4,4] and `int` arithmetic is exact.
                int32_t const nnSum = static_cast<int32_t>(opLattice[inn * nyHalf + j])
                                      + static_cast<int32_t>(opLattice[i * nyHalf + j])
                                      + static_cast<int32_t>(opLattice[ipp * nyHalf + j])
                                      + static_cast<int32_t>(opLattice[i * nyHalf + joff]);

                element_type const lij = lattice[t];
                float const acceptance = alpaka::math::exp(
                    -2.0f * invTemp * static_cast<float>(nnSum) * static_cast<float>(lij));
                if(randvals[t] < acceptance)
                    lattice[t] = static_cast<element_type>(-lij);
            }
        }
    };

    /** Build a 1D frame specification that exposes enough parallelism to cover the packed lattice. */
    auto makeFrameSpec(auto const& device, auto const& exec, int64_t count)
    {
        using namespace alpaka;
        // 128 threads per frame matches the HeCBench reference launch configuration (`THREADS = 128`).
        constexpr int64_t chunkSize = 128;
        int64_t const numFrames = std::max<int64_t>(1, alpaka::divCeil(std::max<int64_t>(count, 1), chunkSize));
        (void) device;
        return onHost::FrameSpec{numFrames, chunkSize, exec};
    }

    auto example(auto const deviceSpec, auto const exec, ising::Params const& params) -> int
    {
        using namespace alpaka;

        int64_t const nx = params.nx;
        int64_t const ny = params.ny;
        int64_t const n2 = nx * (ny / 2);
        float const alpha = static_cast<float>(params.alpha);
        float const invTemp = 1.0f / (alpha * ising::tcrit);

        std::cout << "Using alpaka accelerator " << onHost::demangledName(exec) << " for "
                  << deviceSpec.getApi().getName() << " " << deviceSpec.getDeviceKind().getName() << std::endl;
        std::cout << "lattice dimensions: " << nx << " x " << ny << std::endl;
        std::cout << "temperature: " << alpha << " * " << ising::tcrit << std::endl;
        std::cout << "seed: " << params.seed << std::endl;
        std::cout << "warmup iterations: " << params.nwarmup << std::endl;
        std::cout << "trial iterations: " << params.niters << std::endl;
        std::cout << "measured repetitions: " << params.repeat << std::endl;

        auto devSelector = onHost::makeDeviceSelector(deviceSpec);
        onHost::Device device = devSelector.makeDevice(0);
        // A non-blocking queue lets the backend enqueue the two update kernels of one iteration back to back and
        // only synchronize once per measured block, matching the asynchronous launch pattern of the reference.
        onHost::Queue queue = device.makeQueue(queueKind::nonBlocking);

        Vec<int64_t, 1u> const extent(n2);

        std::vector<float> const randvals = ising::makeRandVals(static_cast<std::size_t>(n2), params.seed);

        auto hRand = onHost::allocHost<float>(extent);
        for(int64_t i = 0; i < n2; ++i)
            hRand[i] = randvals[static_cast<std::size_t>(i)];

        auto dRand = onHost::alloc<float>(device, extent);
        auto dBlack = onHost::alloc<signed char>(device, extent);
        auto dWhite = onHost::alloc<signed char>(device, extent);

        onHost::memcpy(queue, dRand, hRand);

        auto const initFrameSpec = makeFrameSpec(device, exec, n2);
        auto const updFrameSpec = makeFrameSpec(device, exec, n2);
        auto const blackInitKernel = KernelBundle{InitKernel{}, dBlack, dRand, n2};
        auto const whiteInitKernel = KernelBundle{InitKernel{}, dWhite, dRand, n2};

        // One full update = black pass followed by white pass.
        auto const blackKernel = KernelBundle{UpdateKernel<true>{}, dBlack, dWhite, dRand, invTemp, nx, ny / 2};
        auto const whiteKernel = KernelBundle{UpdateKernel<false>{}, dWhite, dBlack, dRand, invTemp, nx, ny / 2};

        /* Each measured repetition starts from the same initialized lattice, performs the untimed warm-up and then
         * times exactly `niters` full updates. This makes every repetition an independent, identical measurement and
         * leaves the final lattice at `nwarmup + niters` sweeps, which is the state the reference verification checks.
         * Re-running the warm-up between repetitions costs no measured time and matches the HeCBench iteration
         * schedule.
         */
        std::vector<double> samples;
        samples.reserve(static_cast<std::size_t>(params.repeat));
        for(int rep = 0; rep < params.repeat; ++rep)
        {
            queue.enqueue(initFrameSpec, blackInitKernel);
            queue.enqueue(initFrameSpec, whiteInitKernel);
            for(int i = 0; i < params.nwarmup; ++i)
            {
                queue.enqueue(updFrameSpec, blackKernel);
                queue.enqueue(updFrameSpec, whiteKernel);
            }
            onHost::wait(queue);

            auto const start = std::chrono::steady_clock::now();
            for(int i = 0; i < params.niters; ++i)
            {
                queue.enqueue(updFrameSpec, blackKernel);
                queue.enqueue(updFrameSpec, whiteKernel);
            }
            onHost::wait(queue);
            auto const end = std::chrono::steady_clock::now();
            double const us = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() * 1e-3;
            samples.push_back(us);
            printf("  measurement %d: %.3f us\n", rep, us);
        }

        std::sort(samples.begin(), samples.end());
        double const median = samples[samples.size() / 2u];
        double const updates = static_cast<double>(nx) * static_cast<double>(ny) * params.niters;
        printf("ising (niters of one full update) kernel runtime median: %.3f us\n", median);
        printf("  updates per ns: %.6f\n", updates / median * 1e-3);

        // Copy results back for verification.
        auto hBlack = onHost::allocHost<signed char>(extent);
        auto hWhite = onHost::allocHost<signed char>(extent);
        onHost::memcpy(queue, hBlack, dBlack);
        onHost::memcpy(queue, hWhite, dWhite);
        onHost::wait(queue);

        return ising::verify(
            nx,
            ny,
            alpha,
            randvals,
            params.nwarmup + params.niters,
            std::data(hBlack),
            std::data(hWhite));
    }
} // namespace isingAlpaka

auto main(int argc, char* argv[]) -> int
{
    ising::Params params;
    if(int const ret = ising::parseCmd(argc, argv, params))
        return ret;

    using namespace alpaka;

    return onHost::executeForEachIfHasDevice(
        [=](auto const& backend)
        { return isingAlpaka::example(onHost::makeDeviceSpec(backend), getExecutor(backend), params); },
        onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors));
}
