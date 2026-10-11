/* Copyright 2026 René Widera
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** @file Portable, single-source alpaka implementation of the HeCBench `stencil1d` benchmark.
 *
 * The HeCBench reference applies one radius-7 sliding-sum stencil to a 1D array. Each thread block
 * cooperatively stages a `BLOCK_SIZE + 2*RADIUS` window in shared memory, fills the two halo regions
 * at the block borders, synchronises and then sums `2*RADIUS + 1` neighbours per output element.
 *
 * This port keeps exactly that algorithm and the same shared-memory window, but the tile size is a
 * compile-time constant passed to `onHost::FrameSpec`. There is a single kernel source; the alpaka
 * backend maps a frame to a host thread or to a CUDA thread block. Only the launch configuration
 * changes between CPU and GPU, never the algorithm.
 *
 * The kernel is correct for arbitrary lengths, in particular lengths that are not a multiple of the
 * tile size: the last block loads only the in-range part of the padded input, out-of-domain halo
 * entries are zero and out-of-domain outputs are not written.
 */

#include <alpaka/alpaka.hpp>

#include "helper.hpp"
#include "misc.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <vector>

namespace stencil1dAlpaka
{
    /** One radius-@c RADIUS 1D stencil step, shared-memory window per thread block.
     *
     * Each block processes at most @c tileSize consecutive output elements. The shared window
     * `temp[tileSize + 2*radius]` holds the centre elements plus a radius-wide halo on both sides.
     * The threads with `localId < radius` load the left and right halo.
     */
    struct StencilKernel
    {
        ALPAKA_FN_ACC void operator()(
            auto const& acc,
            alpaka::concepts::IMdSpan auto out,
            alpaka::concepts::IMdSpan auto in,
            int length,
            uint32_t numBlocks) const
        {
            using namespace alpaka;

            constexpr uint32_t halo = static_cast<uint32_t>(stencil1d::radius);
            constexpr uint32_t tile = stencil1d::tileSize;
            //! shared window: radius left halo + tile centre + radius right halo
            auto& temp = declareSharedVar<int32_t[tile + 2u * halo], uniqueId()>(acc);

            // one block per tile; the block index selects the tile
            for(auto blockIdx : onAcc::makeIdxMap(acc, onAcc::worker::blocksInGrid, IdxRange{Vec{numBlocks}}))
            {
                // CUDA 13.1/13.2 bug workaround: concepts can not be used in a range based for loop
                static_assert(concepts::Dim<ALPAKA_TYPEOF(blockIdx), 1u>);
                uint32_t const blockStart = blockIdx.x() * tile;

                for(auto localId : onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{Vec{tile}}))
                {
                    static_assert(concepts::Dim<ALPAKA_TYPEOF(localId), 1u>);
                    int const gindex = static_cast<int>(blockStart) + static_cast<int>(localId.x());

                    // centre element; positions at or beyond the padded input are zero
                    int32_t centre = int32_t{0};
                    if(gindex < length + stencil1d::radius)
                        centre = in[gindex];
                    temp[localId.x() + halo] = centre;

                    if(localId.x() < halo)
                    {
                        // left halo, positions before the domain are zero
                        int const left = gindex - static_cast<int>(halo);
                        int32_t leftVal = int32_t{0};
                        if(left >= 0 && left < length + stencil1d::radius)
                            leftVal = in[left];
                        temp[localId.x()] = leftVal;

                        // right halo, positions beyond the padded input are zero
                        int const right = gindex + static_cast<int>(tile);
                        int32_t rightVal = int32_t{0};
                        if(right < length + stencil1d::radius)
                            rightVal = in[right];
                        temp[localId.x() + halo + tile] = rightVal;
                    }
                }

                onAcc::syncBlockThreads(acc);

                for(auto localId : onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{Vec{tile}}))
                {
                    static_assert(concepts::Dim<ALPAKA_TYPEOF(localId), 1u>);
                    int const gindex = static_cast<int>(blockStart) + static_cast<int>(localId.x());
                    if(gindex < length)
                    {
                        int32_t result = 0;
                        for(int offset = -stencil1d::radius; offset <= stencil1d::radius; ++offset)
                            result += temp[localId.x() + halo + offset];
                        out[gindex] = result;
                    }
                }

                // avoid a data race when a block computes more than one tile
                onAcc::syncBlockThreads(acc);
            }
        }
    };

    /** Time the kernel with the same scope as the HeCBench reference.
     *
     * The reference enqueues `repeat` back-to-back kernel launches and divides the wall time of the
     * whole region by `repeat`, with a single device synchronisation at the end. This function
     * reproduces that scope exactly and reports the per-kernel time. The per-launch end-to-end time
     * (one launch + one synchronisation) is reported as a secondary metric.
     *
     * Run the program several times to obtain independent samples; the reference provides one value
     * per invocation too.
     */
    template<typename T_Launch, typename T_KernelBundle>
    void timeKernel(auto& queue, T_Launch const& launch, T_KernelBundle const& kernel, int repeat)
    {
        using namespace alpaka;
        queue.enqueue(launch, kernel);
        onHost::wait(queue);

        // reference scope: `repeat` launches in one region, one synchronisation
        auto const start = std::chrono::steady_clock::now();
        for(int i = 0; i < repeat; ++i)
            queue.enqueue(launch, kernel);
        onHost::wait(queue);
        auto const end = std::chrono::steady_clock::now();
        double const us
            = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() * 1e-3 / repeat;
        printf("stencil1d kernel runtime (batch of %d): %.3f us/kernel\n", repeat, us);

        // secondary: end-to-end single launch
        auto const s2 = std::chrono::steady_clock::now();
        queue.enqueue(launch, kernel);
        onHost::wait(queue);
        auto const e2 = std::chrono::steady_clock::now();
        printf(
            "stencil1d per-launch end-to-end: %.3f us\n",
            std::chrono::duration_cast<std::chrono::nanoseconds>(e2 - s2).count() * 1e-3);
    }

    auto example(auto const deviceSpec, auto const exec, int length, int repeat) -> int
    {
        using namespace alpaka;

        std::cout << "Output length: " << length << std::endl;
        std::cout << "Stencil radius: " << stencil1d::radius << std::endl;
        std::cout << "Tile size (threads per block): " << stencil1d::tileSize << std::endl;
        std::cout << "Measured iterations: " << repeat << std::endl;
        std::cout << "Using alpaka accelerator " << onHost::demangledName(exec) << " for "
                  << deviceSpec.getApi().getName() << " " << deviceSpec.getDeviceKind().getName() << std::endl;

        auto devSelector = onHost::makeDeviceSelector(deviceSpec);
        onHost::Device device = devSelector.makeDevice(0);
        // Non-blocking queue: enqueue returns before completion so several launches can be in flight,
        // matching the HeCBench reference which times a batch of launches between two synchronisations.
        onHost::Queue queue = device.makeQueue(queueKind::nonBlocking);

        int const padded = length + stencil1d::radius;

        Vec<int, 1u> const inExtent(padded);
        Vec<int, 1u> const outExtent(length);
        auto hIn = onHost::allocHost<int32_t>(inExtent);
        auto hOut = onHost::allocHost<int32_t>(outExtent);

        /* Non-trivial deterministic input. A monotone ramp a[i] = i would hide an aliasing halo bug,
         * because a shifted or duplicated halo can reproduce the same sum of consecutive integers. */
        stencil1d::makeInput(padded, hIn);
        for(int i = 0; i < length; ++i)
            hOut[i] = 0;

        // Guard the guard: prove the comparison rejects a deliberately broken halo scheme on this input.
        if(int const rc = stencil1d::oracleSanityCheck(length, hIn); rc != EXIT_SUCCESS)
            return rc;

        auto dIn = onHost::alloc<int32_t>(device, inExtent);
        auto dOut = onHost::alloc<int32_t>(device, outExtent);
        onHost::memcpy(queue, dIn, hIn);
        onHost::memcpy(queue, dOut, hOut);

        uint32_t const numBlocks
            = alpaka::divCeil(static_cast<uint32_t>(length), static_cast<uint32_t>(stencil1d::tileSize));
        /* A FrameSpec keeps the tile as a logical frame. The host executor maps one frame to one CPU
         * thread, the CUDA executor maps the frame extent to the CUDA block dimension, so the same kernel
         * runs on both backends. */
        onHost::FrameSpec const launch{Vec{numBlocks}, Vec{stencil1d::tileSize}, exec};
        std::cout << "Launch frames=" << numBlocks << " frameExtent=" << stencil1d::tileSize << std::endl;

        KernelBundle const kernel{StencilKernel{}, dOut, dIn, length, numBlocks};
        timeKernel(queue, launch, kernel, repeat);

        onHost::memcpy(queue, hOut, dOut);
        onHost::wait(queue);
        return stencil1d::verify(length, hOut, hIn);
    }
} // namespace stencil1dAlpaka

auto main(int argc, char* argv[]) -> int
{
    int length = stencil1d::defaultLength;
    int repeat = stencil1d::defaultRepeat;

    if(int const ret = stencil1d::parseCmd(argc, argv, length, repeat))
        return ret;

    using namespace alpaka;

    return onHost::executeForEachIfHasDevice(
        [=](auto const& backend)
        { return stencil1dAlpaka::example(onHost::makeDeviceSpec(backend), getExecutor(backend), length, repeat); },
        onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors));
}
