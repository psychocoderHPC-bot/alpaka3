/* Copyright 2026 Rene Widera
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** @file Portable, single-source alpaka implementation of the HeCBench `cross` benchmark.
 *
 * The benchmark computes the 3D cross product of two dense tensors with shape `(numRows, 3)`. For every row the
 * output vector is `out = x1 x x2`. The exact same kernel source is compiled for the host CPU and for CUDA GPUs;
 * there is no backend branch and no native CUDA/HIP/OpenMP kernel.
 *
 * The computation matches `src/cross-cuda/main.cu` at HeCBench commit
 * 0a17ee47ca6072192101036d4bf292e6612f22e6 (the `cross_kernel` variant, strides of 1).
 */

#include "helper.hpp"
#include "misc.hpp"

#include <alpaka/alpaka.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace crossAlpaka
{
    /** 3D cross product `out = x1 x x2` over `numRows` rows of a `(numRows, 3)` tensor, one row per worker. */
    struct CrossKernel
    {
        ALPAKA_FN_ACC void operator()(
            auto const& acc,
            alpaka::concepts::IMdSpan auto out,
            alpaka::concepts::IMdSpan auto x1,
            alpaka::concepts::IMdSpan auto x2,
            int numRows) const
        {
            using namespace alpaka;
            using T = alpaka::GetValueType_t<ALPAKA_TYPEOF(x1)>;

            for(auto i : onAcc::makeIdxMap(acc, onAcc::worker::linearThreadsInGrid, IdxRange{numRows}))
            {
                int const row = 3 * static_cast<int>(i);

                T const x1_c0 = x1[row + 0];
                T const x1_c1 = x1[row + 1];
                T const x1_c2 = x1[row + 2];
                T const x2_c0 = x2[row + 0];
                T const x2_c1 = x2[row + 1];
                T const x2_c2 = x2[row + 2];

                out[row + 0] = x1_c1 * x2_c2 - x1_c2 * x2_c1;
                out[row + 1] = x1_c2 * x2_c0 - x1_c0 * x2_c2;
                out[row + 2] = x1_c0 * x2_c1 - x1_c1 * x2_c0;
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

    /** Run `repeat` measured iterations of a kernel after one warm-up iteration.
     *
     * Only the kernel enqueue and the device synchronization are timed; allocation and host/device copies are not.
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

    /** Fill the two input tensors exactly like the HeCBench reference: `std::default_random_engine(123)` and a
     * uniform distribution over `[-2, 2)`.
     */
    template<typename T>
    void fillInputs(int numElems, std::vector<T>& a, std::vector<T>& b)
    {
        a.resize(static_cast<std::size_t>(numElems));
        b.resize(static_cast<std::size_t>(numElems));
        std::default_random_engine g(123);
        std::uniform_real_distribution<T> distr(-2.f, 2.f);
        for(int i = 0; i < numElems; ++i)
        {
            a[static_cast<std::size_t>(i)] = distr(g);
            b[static_cast<std::size_t>(i)] = distr(g);
        }
    }

    template<typename T>
    int runType(auto const& deviceSpec, auto const& exec, int numRows, int repeat)
    {
        using namespace alpaka;

        auto devSelector = onHost::makeDeviceSelector(deviceSpec);
        onHost::Device device = devSelector.makeDevice(0);
        // A blocking queue reduces the launch overhead for host devices.
        onHost::Queue queue = device.makeQueue(queueKind::blocking);

        int const numElems = numRows * 3;
        Vec<int, 1u> const extent(numElems);

        std::cout << "Data type " << (std::is_same_v<T, float> ? "float" : "double") << ", rows " << numRows
                  << ", elements " << numElems << std::endl;

        std::vector<T> a;
        std::vector<T> b;
        fillInputs<T>(numElems, a, b);

        auto hIn1 = onHost::allocHost<T>(extent);
        auto hIn2 = onHost::allocHost<T>(extent);
        auto hOut = onHost::allocHost<T>(extent);
        for(int i = 0; i < numElems; ++i)
        {
            hIn1[static_cast<std::size_t>(i)] = a[static_cast<std::size_t>(i)];
            hIn2[static_cast<std::size_t>(i)] = b[static_cast<std::size_t>(i)];
        }

        auto dIn1 = onHost::alloc<T>(device, extent);
        auto dIn2 = onHost::alloc<T>(device, extent);
        auto dOut = onHost::alloc<T>(device, extent);
        onHost::memcpy(queue, dIn1, hIn1);
        onHost::memcpy(queue, dIn2, hIn2);

        auto const frameSpec = makeFrameSpec<T>(device, exec, static_cast<uint32_t>(numRows));
        KernelBundle const kernel{CrossKernel{}, dOut, dIn1, dIn2, numRows};
        printf("cross (rows) kernel runtime median: %.3f us\n", timeKernel(queue, frameSpec, kernel, repeat));

        onHost::memcpy(queue, hOut, dOut);
        onHost::wait(queue);

        return cross::verify<T>(numRows, std::data(hOut), std::data(hIn1), std::data(hIn2));
    }

    auto example(auto const deviceSpec, auto const exec, int numRows, int repeat, std::string const& dtype) -> int
    {
        using namespace alpaka;

        std::cout << "Problem size (rows of 3D vectors): " << numRows << std::endl;
        std::cout << "Measured iterations: " << repeat << std::endl;
        std::cout << "Using alpaka accelerator " << onHost::demangledName(exec) << " for "
                  << deviceSpec.getApi().getName() << " " << deviceSpec.getDeviceKind().getName() << std::endl;

        int result = EXIT_SUCCESS;

        if(dtype == "float" || dtype == "both")
        {
            printf("=========== Data type is FP32 ==========\n");
            result |= runType<float>(deviceSpec, exec, numRows, repeat);
        }
        if(dtype == "double" || dtype == "both")
        {
            printf("=========== Data type is FP64 ==========\n");
            result |= runType<double>(deviceSpec, exec, numRows, repeat);
        }
        return result;
    }
} // namespace crossAlpaka

auto main(int argc, char* argv[]) -> int
{
    int numRows = cross::defaultNumRows;
    int repeat = cross::defaultRepeat;
    std::string dtype = "both";

    if(int const ret = cross::parseCmd(argc, argv, numRows, repeat, dtype))
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
            return crossAlpaka::example(onHost::makeDeviceSpec(backend), getExecutor(backend), numRows, repeat, dtype);
        },
        onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors));
}
