/* Copyright 2026 Rene Widera
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** @file Portable, single-source alpaka implementation of the HeCBench `axpby` benchmark.
 *
 * The benchmark computes the fused elementwise update `out = a * x + b * y` over one or more independent dense
 * tensors. It reproduces the HeCBench multi-tensor dispatch: the work is split into fixed-size chunks and a flat
 * list of `(tensor, chunk)` blocks is built, so a single kernel launch covers every tensor exactly like the
 * reference `multi_tensor_apply` launcher. The exact same kernel source is compiled for the host CPU and for CUDA
 * GPUs; there is no backend branch and no native CUDA/HIP/OpenMP kernel.
 *
 * The fused per-element math `a * static_cast<float>(x[i]) + b * static_cast<float>(y[i])` and the chunked
 * multi-tensor dispatch match `src/axpby-cuda/main.cu` at HeCBench commit
 * 0a17ee47ca6072192101036d4bf292e6612f22e6 (and the equivalent OpenMP reference `src/axpby-omp/main.cpp`).
 */

#include "helper.hpp"
#include "misc.hpp"

#include <alpaka/alpaka.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

namespace axpbyAlpaka
{
    /** Fused `out = a * x + b * y` for one chunk of one tensor.
     *
     * Each frame (block) is responsible for exactly one entry of the flat chunk list and therefore for exactly one
     * chunk of one tensor, matching the reference `multi_tensor_apply_kernel<<<blocks, BLOCK_SIZE>>>` mapping.
     * Threads within the frame stride over the elements of the chunk.
     */
    struct AxpbyKernel
    {
        ALPAKA_FN_ACC void operator()(
            auto const& acc,
            alpaka::concepts::IMdSpan auto out,
            alpaka::concepts::IMdSpan auto x,
            alpaka::concepts::IMdSpan auto y,
            alpaka::concepts::IMdSpan auto chunkTensor,
            alpaka::concepts::IMdSpan auto chunkIndex,
            alpaka::concepts::IMdSpan auto offsets,
            alpaka::concepts::IMdSpan auto lengths,
            int32_t numChunks,
            int64_t chunkSize,
            float a,
            float b) const
        {
            using namespace alpaka;

            for(auto blockIdx : onAcc::makeIdxMap(acc, onAcc::worker::blocksInGrid, IdxRange{numChunks}))
            {
                int32_t const block = static_cast<int32_t>(blockIdx.x());
                if(block >= numChunks)
                    continue;

                int32_t const tensor = chunkTensor[static_cast<std::size_t>(block)];
                int64_t const base = chunkIndex[static_cast<std::size_t>(block)] * chunkSize;
                int64_t const len
                    = std::min(lengths[static_cast<std::size_t>(tensor)] - base, chunkSize);
                int64_t const off = offsets[static_cast<std::size_t>(tensor)] + base;

                for(auto i : onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{len}))
                {
                    int64_t const idx = off + static_cast<int64_t>(i);
                    out[static_cast<std::size_t>(idx)]
                        = a * static_cast<float>(x[static_cast<std::size_t>(idx)])
                          + b * static_cast<float>(y[static_cast<std::size_t>(idx)]);
                }
            }
        }
    };

    /** Build the flat `(tensor, chunk)` block list exactly like the reference `multi_tensor_apply` launcher. */
    inline void makeChunkList(
        std::vector<int64_t> const& lengths,
        int64_t chunkSize,
        std::vector<int32_t>& chunkTensor,
        std::vector<int64_t>& chunkIndex)
    {
        chunkTensor.clear();
        chunkIndex.clear();
        for(std::size_t t = 0; t < lengths.size(); ++t)
        {
            int64_t const chunksThisTensor = (lengths[t] + chunkSize - 1) / chunkSize;
            for(int64_t c = 0; c < chunksThisTensor; ++c)
            {
                chunkTensor.push_back(static_cast<int32_t>(t));
                chunkIndex.push_back(c);
            }
        }
    }

    /** Run `repeat` measured iterations of the kernel after one warm-up iteration.
     *
     * Only the kernel enqueue and the device synchronization are timed, matching the reference timing scope which
     * brackets the launch with `cudaDeviceSynchronize()`.
     *
     * @return the median runtime in microseconds.
     */
    template<typename T_KernelBundle>
    double timeKernel(auto& queue, auto const& frameSpec, T_KernelBundle const& kernel, int repeat)
    {
        using namespace alpaka;
        // warm-up
        queue.enqueue(frameSpec, kernel);
        onHost::wait(queue);

        std::vector<double> samples;
        samples.reserve(static_cast<std::size_t>(repeat));
        for(int i = 0; i < repeat; ++i)
        {
            auto const start = std::chrono::steady_clock::now();
            queue.enqueue(frameSpec, kernel);
            onHost::wait(queue);
            auto const end = std::chrono::steady_clock::now();
            double const us = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() * 1e-3;
            samples.push_back(us);
            printf("  iteration %d: %.3f us\n", i, us);
        }
        std::sort(samples.begin(), samples.end());
        double const median = samples.empty() ? 0.0 : samples[samples.size() / 2u];
        return median;
    }

    int run(
        auto const deviceSpec,
        auto const exec,
        int numTensors,
        int chunkSizeArg,
        int repeat,
        int fixedLength)
    {
        using namespace alpaka;

        auto devSelector = onHost::makeDeviceSelector(deviceSpec);
        onHost::Device device = devSelector.makeDevice(0);
        // A blocking queue reduces the launch overhead for host devices.
        onHost::Queue queue = device.makeQueue(queueKind::blocking);

        std::cout << "Using alpaka accelerator " << onHost::demangledName(exec) << " for "
                  << deviceSpec.getApi().getName() << " " << deviceSpec.getDeviceKind().getName() << std::endl;

        std::vector<int64_t> lengths;
        std::vector<int64_t> offsets;
        std::vector<float> xHost;
        std::vector<float> yHost;
        axpby::generateInputs(numTensors, fixedLength, lengths, offsets, xHost, yHost);

        int64_t const totalElems = static_cast<int64_t>(xHost.size());
        int64_t const chunkSize = chunkSizeArg;
        printf(
            "Tensors %d, total elements %lld, chunk size %lld\n",
            numTensors,
            (long long)totalElems,
            (long long)chunkSize);

        std::vector<int32_t> chunkTensor;
        std::vector<int64_t> chunkIndex;
        makeChunkList(lengths, chunkSize, chunkTensor, chunkIndex);
        int32_t const numChunks = static_cast<int32_t>(chunkTensor.size());

        // Frame extent is the requested block size; alpaka clamps it to the device limit on GPUs.
        // 512 matches the reference `BLOCK_SIZE` of the HeCBench launcher.
        constexpr uint32_t threadsPerBlock = 512u;

        Vec<int64_t, 1u> const elemExtent(totalElems);
        Vec<int32_t, 1u> const chunkExtent(numChunks);
        Vec<int64_t, 1u> const tensorExtent(static_cast<int64_t>(lengths.size()));

        auto hX = onHost::allocHost<float>(elemExtent);
        auto hY = onHost::allocHost<float>(elemExtent);
        auto hOut = onHost::allocHost<float>(elemExtent);
        for(int64_t i = 0; i < totalElems; ++i)
        {
            hX[static_cast<std::size_t>(i)] = xHost[static_cast<std::size_t>(i)];
            hY[static_cast<std::size_t>(i)] = yHost[static_cast<std::size_t>(i)];
        }

        auto hChunkTensor = onHost::allocHost<int32_t>(chunkExtent);
        auto hChunkIndex = onHost::allocHost<int64_t>(chunkExtent);
        auto hOffsets = onHost::allocHost<int64_t>(tensorExtent);
        auto hLengths = onHost::allocHost<int64_t>(tensorExtent);
        for(std::size_t t = 0; t < lengths.size(); ++t)
        {
            hOffsets[t] = offsets[t];
            hLengths[t] = lengths[t];
        }
        for(int32_t c = 0; c < numChunks; ++c)
        {
            hChunkTensor[static_cast<std::size_t>(c)] = chunkTensor[static_cast<std::size_t>(c)];
            hChunkIndex[static_cast<std::size_t>(c)] = chunkIndex[static_cast<std::size_t>(c)];
        }

        auto dX = onHost::alloc<float>(device, elemExtent);
        auto dY = onHost::alloc<float>(device, elemExtent);
        auto dOut = onHost::alloc<float>(device, elemExtent);
        auto dChunkTensor = onHost::alloc<int32_t>(device, chunkExtent);
        auto dChunkIndex = onHost::alloc<int64_t>(device, chunkExtent);
        auto dOffsets = onHost::alloc<int64_t>(device, tensorExtent);
        auto dLengths = onHost::alloc<int64_t>(device, tensorExtent);

        onHost::memcpy(queue, dX, hX);
        onHost::memcpy(queue, dY, hY);
        onHost::memcpy(queue, dChunkTensor, hChunkTensor);
        onHost::memcpy(queue, dChunkIndex, hChunkIndex);
        onHost::memcpy(queue, dOffsets, hOffsets);
        onHost::memcpy(queue, dLengths, hLengths);

        float const a = 1.f;
        float const b = 1.f;

        // One frame (block) per chunk mirrors the reference launcher.
        auto const frameSpec
            = onHost::FrameSpec{Vec<uint32_t, 1u>(static_cast<uint32_t>(numChunks)), Vec<uint32_t, 1u>(threadsPerBlock), exec};
        KernelBundle const kernel{
            AxpbyKernel{},
            dOut,
            dX,
            dY,
            dChunkTensor,
            dChunkIndex,
            dOffsets,
            dLengths,
            numChunks,
            chunkSize,
            a,
            b};

        printf("axpby kernel runtime median: %.3f us\n", timeKernel(queue, frameSpec, kernel, repeat));

        onHost::memcpy(queue, hOut, dOut);
        onHost::wait(queue);

        return axpby::verify(lengths, offsets, std::data(hX), std::data(hY), std::data(hOut), a, b);
    }

    auto example(
        auto const deviceSpec,
        auto const exec,
        int numTensors,
        int chunkSize,
        int repeat,
        int fixedLength) -> int
    {
        using namespace alpaka;
        return run(deviceSpec, exec, numTensors, chunkSize, repeat, fixedLength);
    }
} // namespace axpbyAlpaka

auto main(int argc, char* argv[]) -> int
{
    int numTensors = axpby::defaultNumTensors;
    int chunkSize = axpby::defaultChunkSize;
    int repeat = axpby::defaultRepeat;
    int fixedLength = 0;

    if(int const ret = axpby::parseCmd(argc, argv, numTensors, chunkSize, repeat, fixedLength))
        return ret;

    using namespace alpaka;

    return onHost::executeForEachIfHasDevice(
        [=](auto const& backend)
        {
            return axpbyAlpaka::example(
                onHost::makeDeviceSpec(backend),
                getExecutor(backend),
                numTensors,
                chunkSize,
                repeat,
                fixedLength);
        },
        onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors));
}
