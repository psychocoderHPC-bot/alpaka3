/* Copyright 2026 René Widera
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** @file Portable, single-source alpaka implementation of the HeCBench `popcount` benchmark.
 *
 * The HeCBench reference program evaluates six different ways to count the set bits of a 64 bit word
 * (SWAR bit tricks, Kernighan's `x &= x - 1` loop, a 256 entry lookup table and the hardware
 * `__popcll` instruction). This port benchmarks the operation itself through the alpaka intrinsic
 * abstraction `alpaka::popcount`, which lowers to `std::popcount` on the host and to the native
 * `__popcll` on NVIDIA hardware. The exact same kernel source is compiled for the host CPU and for
 * CUDA GPUs. There is no backend branch and no native CUDA/HIP/OpenMP kernel.
 */

#include <alpaka/alpaka.hpp>

#include "helper.hpp"
#include "misc.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <vector>

namespace popcountAlpaka
{
    /** Count the set bits of every 64 bit word.
     *
     * `alpaka::popcount` is the portable intrinsic: the alpaka backend selects the best available
     * implementation (hardware popcount on CUDA/HIP/SYCL, `std::popcount` on the host). Using the
     * abstraction keeps a single algorithm for all backends.
     */
    struct PopcountKernel
    {
        ALPAKA_FN_ACC void operator()(
            auto const& acc,
            alpaka::concepts::IMdSpan auto output,
            alpaka::concepts::IMdSpan auto input,
            int count) const
        {
            using namespace alpaka;
            for(auto i : onAcc::makeIdxMap(acc, onAcc::worker::linearThreadsInGrid, IdxRange{count}))
            {
                output[i] = static_cast<int32_t>(alpaka::popcount(static_cast<uint64_t>(input[i])));
            }
        }
    };

    /** Build a frame specification that gives every worker a useful amount of work. */
    template<typename T_Data>
    auto makeFrameSpec(auto const& device, auto const& exec, uint32_t extent)
    {
        using namespace alpaka;
        constexpr uint32_t chunkSize = 256u;
        uint32_t const elementsPerWorker = getNumElemPerThread<T_Data>(device);
        uint32_t const blocks = alpaka::divCeil(std::max(extent, 1u), chunkSize * elementsPerWorker);
        return onHost::FrameSpec{std::max(blocks, 1u), chunkSize, exec};
    }

    /** Validate that every host result is a well formed 64 bit popcount.
     *
     * The valid range is `[0, 64]`, i.e. `[0, 8 * sizeof(uint64_t)]`. This is an explicit
     * self-validation guard on the host results, independent of the bit-exact comparison against
     * the reference; it runs on the host, never inside the timed kernel.
     *
     * @return EXIT_SUCCESS when every value is in range, otherwise EXIT_FAILURE.
     */
    inline int checkRange(int count, auto const& out)
    {
        constexpr int maxPopcount = static_cast<int>(8u * sizeof(uint64_t));
        for(int i = 0; i < count; ++i)
        {
            int const value = static_cast<int>(out[i]);
            if(value < 0 || value > maxPopcount)
            {
                printf("range error at result[%d]: %d outside [0, %d]\n", i, value, maxPopcount);
                return EXIT_FAILURE;
            }
        }
        return EXIT_SUCCESS;
    }

    /** Run `repeat` measured iterations of a kernel after one warm-up iteration.
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

    auto example(auto const deviceSpec, auto const exec, int count, int repeat) -> int
    {
        using namespace alpaka;

        std::cout << "Problem size (elements): " << count << std::endl;
        std::cout << "Measured iterations: " << repeat << std::endl;
        std::cout << "Using alpaka accelerator " << onHost::demangledName(exec) << " for "
                  << deviceSpec.getApi().getName() << " " << deviceSpec.getDeviceKind().getName() << std::endl;

        auto devSelector = onHost::makeDeviceSelector(deviceSpec);
        onHost::Device device = devSelector.makeDevice(0);
        // A blocking queue reduces the launch overhead for host devices.
        onHost::Queue queue = device.makeQueue(queueKind::blocking);

        Vec<int, 1u> const extent(count);
        auto hIn = onHost::allocHost<uint64_t>(extent);
        auto hOut = onHost::allocHost<int32_t>(extent);

        // Input generation matches the HeCBench reference: srand(2) and two rand() draws per element,
        // the first shifted into the high 32 bits.
        std::srand(2);
        for(int i = 0; i < count; ++i)
        {
            uint64_t const high = static_cast<uint64_t>(static_cast<uint32_t>(std::rand())) << 32;
            uint64_t const low = static_cast<uint32_t>(std::rand());
            hIn[i] = high | low;
        }

        auto dIn = onHost::alloc<uint64_t>(device, extent);
        auto dOut = onHost::alloc<int32_t>(device, extent);
        onHost::memcpy(queue, dIn, hIn);

        auto const frameSpec = makeFrameSpec<uint64_t>(device, exec, static_cast<uint32_t>(count));
        KernelBundle const kernel{PopcountKernel{}, dOut, dIn, count};
        printf("Popcount kernel runtime median: %.3f us\n", timeKernel(queue, frameSpec, kernel, repeat));

        onHost::memcpy(queue, hOut, dOut);
        onHost::wait(queue);
        // Explicit host-side range self-validation, independent of the reference comparison.
        if(int const rangeError = checkRange(count, hOut); rangeError != EXIT_SUCCESS)
            return rangeError;
        return popcount::verify(count, hOut, hIn);
    }
} // namespace popcountAlpaka

auto main(int argc, char* argv[]) -> int
{
    int count = popcount::defaultCount;
    int repeat = popcount::defaultRepeat;

    if(int const ret = popcount::parseCmd(argc, argv, count, repeat))
        return ret;

    using namespace alpaka;

    return onHost::executeForEachIfHasDevice(
        [=](auto const& backend)
        { return popcountAlpaka::example(onHost::makeDeviceSpec(backend), getExecutor(backend), count, repeat); },
        onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors));
}
