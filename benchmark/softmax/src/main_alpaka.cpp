/* Copyright 2026 Rene Widera
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** @file Portable, single-source alpaka implementation of the HeCBench `softmax` benchmark.
 *
 * For every row (slice) of a dense `(numSlice, sliceSize)` tensor, the kernel computes the numerically
 * stable softmax
 *
 *   m   = max_j x[j]
 *   e_j = exp(x[j] - m)
 *   s   = sum_j e_j
 *   y_j = e_j / s
 *
 * The exact same kernel source is compiled for the host CPU and for CUDA GPUs; there is no backend branch
 * and no native CUDA/HIP/OpenMP kernel.
 *
 * The scheme, precision, sizes and verification mirror `src/softmax-cuda/main.cu` (the measured naive
 * `softMax` kernel, implementation 0) and `src/softmax-omp/main.cpp` at HeCBench commit
 * 0a17ee47ca6072192101036d4bf292e6612f22e6. One worker owns one full row, which is exactly the structure
 * of the reference `softMax` kernel (one thread per slice).
 */

#include "helper.hpp"
#include "misc.hpp"

#include <alpaka/alpaka.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace softmaxAlpaka
{
    /** Row-wise softmax with one worker (thread) per slice.
     *
     * Each worker loads its row, finds the row maximum, accumulates the exponentials, and writes the
     * normalized result. This is the portable equivalent of the HeCBench `softMax` kernel.
     */
    struct SoftmaxKernel
    {
        ALPAKA_FN_ACC void operator()(
            auto const& acc,
            alpaka::concepts::IMdSpan auto out,
            alpaka::concepts::IMdSpan auto in,
            int numSlice,
            int sliceSize) const
        {
            using namespace alpaka;
            using T = alpaka::GetValueType_t<ALPAKA_TYPEOF(in)>;

            for(auto [row] : onAcc::makeIdxMap(acc, onAcc::worker::linearThreadsInGrid, IdxRange{numSlice}))
            {
                int const base = static_cast<int>(row) * sliceSize;

                T maxValue = in[base];
                for(int j = 1; j < sliceSize; ++j)
                    maxValue = math::max(maxValue, in[base + j]);

                T sum = static_cast<T>(0);
                for(int j = 0; j < sliceSize; ++j)
                    sum += math::exp(in[base + j] - maxValue);

                for(int j = 0; j < sliceSize; ++j)
                    out[base + j] = math::exp(in[base + j] - maxValue) / sum;
            }
        }
    };

    /** Build a frame specification with a fixed 256-worker block, matching the HeCBench `BLOCK_SIZE`. */
    auto makeFrameSpec(auto const& exec, uint32_t numSlice)
    {
        using namespace alpaka;
        constexpr uint32_t chunkSize = 256u;
        uint32_t const blocks = alpaka::divCeil(std::max(numSlice, 1u), chunkSize);
        return onHost::FrameSpec{std::max(blocks, 1u), chunkSize, exec};
    }

    /** Run `repeat` measured iterations of a kernel after one warm-up iteration.
     *
     * Only the kernel enqueue and the device synchronization are timed; allocation and host/device copies
     * are not. The reported value is the median of the `repeat` samples in microseconds.
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

    /** Fill the input exactly like the HeCBench references: `srand(2)` and `rand() % 13` per element. */
    template<typename T>
    void fillInput(int numElems, std::vector<T>& input)
    {
        input.resize(static_cast<std::size_t>(numElems));
        std::srand(2);
        for(int i = 0; i < numElems; ++i)
            input[static_cast<std::size_t>(i)] = static_cast<T>(std::rand() % 13);
    }

    template<typename T>
    int runType(auto const& deviceSpec, auto const& exec, int numSlice, int sliceSize, int repeat)
    {
        using namespace alpaka;

        auto devSelector = onHost::makeDeviceSelector(deviceSpec);
        onHost::Device device = devSelector.makeDevice(0);
        onHost::Queue queue = device.makeQueue(queueKind::blocking);

        int const numElems = numSlice * sliceSize;
        Vec<int, 1u> const extent(numElems);

        std::cout << "Data type " << (std::is_same_v<T, float> ? "float" : "double") << ", slices " << numSlice
                  << ", slice size " << sliceSize << ", elements " << numElems << std::endl;

        std::vector<T> input;
        fillInput<T>(numElems, input);

        // Independent double-precision host reference.
        std::vector<double> gold(static_cast<std::size_t>(numElems));
        softmax::reference<T>(numSlice, sliceSize, std::data(input), std::data(gold));

        auto hIn = onHost::allocHost<T>(extent);
        auto hOut = onHost::allocHost<T>(extent);
        for(int i = 0; i < numElems; ++i)
            hIn[static_cast<std::size_t>(i)] = input[static_cast<std::size_t>(i)];

        auto dIn = onHost::alloc<T>(device, extent);
        auto dOut = onHost::alloc<T>(device, extent);
        onHost::memcpy(queue, dIn, hIn);

        auto const frameSpec = makeFrameSpec(exec, static_cast<uint32_t>(numSlice));
        KernelBundle const kernel{SoftmaxKernel{}, dOut, dIn, numSlice, sliceSize};
        printf("softmax kernel runtime median: %.3f us\n", timeKernel(queue, frameSpec, kernel, repeat));

        onHost::memcpy(queue, hOut, dOut);
        onHost::wait(queue);

        return softmax::verify<T>(numSlice, sliceSize, std::data(input), std::data(hOut));
    }

    auto example(auto const deviceSpec, auto const exec, int numSlice, int sliceSize, int repeat, std::string const& dtype)
        -> int
    {
        using namespace alpaka;

        std::cout << "Number of slices: " << numSlice << std::endl;
        std::cout << "Slice size: " << sliceSize << std::endl;
        std::cout << "Measured iterations: " << repeat << std::endl;
        std::cout << "Using alpaka accelerator " << onHost::demangledName(exec) << " for "
                  << deviceSpec.getApi().getName() << " " << deviceSpec.getDeviceKind().getName() << std::endl;

        int result = EXIT_SUCCESS;

        if(dtype == "float" || dtype == "both")
        {
            printf("=========== Data type is FP32 ==========\n");
            result |= runType<float>(deviceSpec, exec, numSlice, sliceSize, repeat);
        }
        if(dtype == "double" || dtype == "both")
        {
            printf("=========== Data type is FP64 ==========\n");
            result |= runType<double>(deviceSpec, exec, numSlice, sliceSize, repeat);
        }
        return result;
    }
} // namespace softmaxAlpaka

auto main(int argc, char* argv[]) -> int
{
    int numSlice = softmax::defaultNumSlice;
    int sliceSize = softmax::defaultSliceSize;
    int repeat = softmax::defaultRepeat;
    std::string dtype = "float";

    if(int const ret = softmax::parseCmd(argc, argv, numSlice, sliceSize, repeat, dtype))
        return ret;

    if(dtype != "float" && dtype != "double" && dtype != "both")
    {
        std::cerr << "Error: --dtype must be one of float, double or both.\n";
        return EXIT_FAILURE;
    }

    using namespace alpaka;

    return onHost::executeForEachIfHasDevice(
        [=](auto const& backend)
        {
            return softmaxAlpaka::example(
                onHost::makeDeviceSpec(backend),
                getExecutor(backend),
                numSlice,
                sliceSize,
                repeat,
                dtype);
        },
        onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors));
}
