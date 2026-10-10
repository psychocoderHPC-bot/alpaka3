/* Copyright 2026 René Widera
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** @file Portable, single-source alpaka implementation of the HeCBench `reverse` benchmark.
 *
 * The HeCBench reference reverses an `int` array of `len` elements in place on the device. Each
 * launch is a permutation: element `i` receives the value that was at `len - 1 - i`. Applying the
 * reverse an even number of times restores the identity, an odd number of times leaves the array
 * reversed. For every test case the reference draws a random count in `[100, 9999]`, launches that
 * many back-to-back reverses, and verifies the result against `gold_even[i] = i` (even count) or
 * `gold_odd[i] = len - i - 1` (odd count) with `memcmp`.
 *
 * This port keeps the same in-place reverse, the same `std::default_random_engine(123)` draw order
 * and the same even/odd `memcmp` verification. A single kernel source runs on the host CPU and on
 * CUDA GPUs; only the launch configuration changes.
 *
 * In-place reversal without a barrier: the reverse is a set of independent swaps `i <-> len-1-i` for
 * `i < len/2`. Exactly one logical worker owns each swap, so no two workers read or write the same
 * slot and no block-level or grid-level synchronization is required. For odd `len` the middle element
 * is untouched, matching the reference permutation. The result is algebraically identical to the
 * reference kernel, which stages the whole array through shared memory and therefore also never lets
 * a worker overwrite a value another worker still has to read.
 */

#include <alpaka/alpaka.hpp>

#include "helper.hpp"
#include "misc.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <random>
#include <vector>

namespace reverseAlpaka
{
    //! Threads per frame. Mirrors the reference `reverse<<<1, len>>>` for the default `len = 256`.
    constexpr uint32_t frameExtent = 256u;

    /** One in-place reverse of @p len elements by independent mirrored swaps.
     *
     * Each frame (thread block) covers one contiguous range of the first half of the array; the worker
     * at index `i < len/2` swaps `d[i]` with `d[len-1-i]`. Every pair is owned by exactly one worker,
     * so the writes never race and the operation is safe on any backend without a barrier. Launches for
     * a single test case are enqueued on an in-order queue, so each reverse observes the previous one.
     */
    struct ReverseKernel
    {
        ALPAKA_FN_ACC void operator()(
            auto const& acc,
            alpaka::concepts::IMdSpan auto d,
            int len,
            uint32_t numBlocks) const
        {
            using namespace alpaka;

            for(auto blockIdx : onAcc::makeIdxMap(acc, onAcc::worker::blocksInGrid, IdxRange{Vec{numBlocks}}))
            {
                static_assert(concepts::Dim<ALPAKA_TYPEOF(blockIdx), 1u>);
                for(auto localId : onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{Vec{frameExtent}}))
                {
                    static_assert(concepts::Dim<ALPAKA_TYPEOF(localId), 1u>);
                    int const i = static_cast<int>(blockIdx.x() * frameExtent) + static_cast<int>(localId.x());
                    if(i < len / 2)
                    {
                        int const j = len - 1 - i;
                        auto const tmp = d[i];
                        d[i] = d[j];
                        d[j] = tmp;
                    }
                }
            }
        }
    };

    /** Draw the per-test-case reverse counts exactly like the reference.
     *
     * `std::default_random_engine generator(123)` with `std::uniform_int_distribution<int>(100, 9999)`,
     * drawn once per test case. This reproduces the reference sequence and therefore the same odd/even
     * parity used for verification.
     */
    inline std::vector<int> drawCounts(int iterations)
    {
        std::default_random_engine generator(123);
        std::uniform_int_distribution<int> distribution(100, 9999);
        std::vector<int> counts;
        counts.reserve(static_cast<std::size_t>(iterations));
        for(int i = 0; i < iterations; ++i)
            counts.push_back(distribution(generator));
        return counts;
    }

    auto run(auto const deviceSpec, auto const exec, int iterations, int length, int repeat) -> int
    {
        using namespace alpaka;

        std::cout << "Number of elements: " << length << std::endl;
        std::cout << "Test cases (iterations): " << iterations << std::endl;
        std::cout << "Measured repetitions: " << repeat << std::endl;
        std::cout << "Using alpaka accelerator " << onHost::demangledName(exec) << " for "
                  << deviceSpec.getApi().getName() << " " << deviceSpec.getDeviceKind().getName() << std::endl;

        auto devSelector = onHost::makeDeviceSelector(deviceSpec);
        onHost::Device device = devSelector.makeDevice(0);
        /* In-order non-blocking queue. The enqueue path must not synchronize per launch: the reference
         * enqueues `count` back-to-back launches on one stream and synchronizes once at the end, so the
         * queue is what makes the timing scope comparable. In-order execution guarantees that each
         * in-place reverse observes the writes of the previous one. */
        onHost::Queue queue = device.makeQueue(queueKind::nonBlocking);

        Vec<int, 1u> const extent(length);
        auto hBuffer = onHost::allocHost<int>(extent);
        auto hResult = onHost::allocHost<int>(extent);
        auto dBuffer = onHost::alloc<int>(device, extent);

        std::vector<int> hGoldEven(static_cast<std::size_t>(length));
        for(int i = 0; i < length; ++i)
            hGoldEven[static_cast<std::size_t>(i)] = i;

        uint32_t const numBlocks = static_cast<uint32_t>(alpaka::divCeil(length, static_cast<int>(frameExtent)));
        onHost::FrameSpec const launch{Vec{numBlocks}, Vec{frameExtent}, exec};
        KernelBundle const kernel{ReverseKernel{}, dBuffer, length, numBlocks};

        std::cout << "Launch frames=" << numBlocks << " frameExtent=" << frameExtent << std::endl;

        std::vector<int> const counts = drawCounts(iterations);
        int const* goldEven = hGoldEven.data();

        //! Run the whole iteration loop once. When @p check is false the device->host copy and the
        //! verification are skipped, so only the timed launch region remains.
        auto runOnce = [&](bool check) -> std::pair<double, bool>
        {
            double totalSeconds = 0.0;
            bool verified = true;
            for(std::size_t caseIdx = 0; caseIdx < counts.size(); ++caseIdx)
            {
                int const count = counts[caseIdx];

                // reset the device buffer to the identity sequence (reference `gold_even`)
                std::copy(goldEven, goldEven + length, hBuffer.data());
                onHost::memcpy(queue, dBuffer, hBuffer);
                onHost::wait(queue);

                auto const start = std::chrono::steady_clock::now();
                for(int j = 0; j < count; ++j)
                    queue.enqueue(launch, kernel);
                onHost::wait(queue);
                auto const end = std::chrono::steady_clock::now();
                totalSeconds += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() * 1e-9;

                if(check)
                {
                    onHost::memcpy(queue, hBuffer, dBuffer);
                    onHost::wait(queue);
                    std::copy(hBuffer.data(), hBuffer.data() + length, hResult.data());
                    bool const odd = (count % 2) != 0;
                    if(reverse::verify(length, hResult.data(), odd) != EXIT_SUCCESS)
                    {
                        verified = false;
                        break;
                    }
                }
            }
            return {totalSeconds, verified};
        };

        // Warm-up and full correctness check (all test cases, exact verification).
        auto const warm = runOnce(true);
        bool allVerified = warm.second;

        std::vector<double> samples;
        samples.reserve(static_cast<std::size_t>(repeat));
        for(int r = 0; r < repeat; ++r)
        {
            auto const res = runOnce(false);
            samples.push_back(res.first);
            printf("repetition %d: total kernel execution time %.6f (s)\n", r, res.first);
        }

        std::vector<double> sorted = samples;
        std::sort(sorted.begin(), sorted.end());
        double const median = sorted[sorted.size() / 2u];
        printf("Total kernel execution time (median of %d): %.6f (s)\n", repeat, median);
        printf("%s\n", allVerified ? "PASS" : "FAIL");
        return allVerified ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    auto example(auto const deviceSpec, auto const exec, int iterations, int length, int repeat) -> int
    {
        return run(deviceSpec, exec, iterations, length, repeat);
    }
} // namespace reverseAlpaka

auto main(int argc, char* argv[]) -> int
{
    int iterations = reverse::defaultIterations;
    int length = reverse::defaultLength;
    int repeat = reverse::defaultRepeat;

    if(int const ret = reverse::parseCmd(argc, argv, iterations, length, repeat))
        return ret;

    using namespace alpaka;

    return onHost::executeForEachIfHasDevice(
        [=](auto const& backend)
        {
            return reverseAlpaka::example(
                onHost::makeDeviceSpec(backend),
                getExecutor(backend),
                iterations,
                length,
                repeat);
        },
        onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors));
}
