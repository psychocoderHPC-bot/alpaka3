/* Copyright 2026 Rene Widera
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** @file Portable, single-source alpaka implementation of the HeCBench `relu` benchmark.
 *
 * Two operations are covered, matching the HeCBench reference programs:
 *  - ReluGrad: elementwise fp16 `(feature > 0) ? gradient : 0` (TensorFlow ReluGrad).
 *  - Relu:     elementwise `max(value, 0)` applied independently to the four signed bytes of every int32 word.
 *
 * The exact same kernel source is compiled for the host CPU and for CUDA GPUs. There is no backend branch and no
 * native CUDA/HIP/OpenMP kernel.
 */

#include <alpaka/alpaka.hpp>

#include "helper.hpp"
#include "misc.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

namespace reluAlpaka
{
    /** Packed-integer ReLU: apply `max(byte, 0)` to each of the four signed bytes of an int32 word. */
    struct ReluIntKernel
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
                int32_t const value = input[i];
                auto const c0 = static_cast<int8_t>(value & 0xFF);
                auto const c1 = static_cast<int8_t>((value >> 8) & 0xFF);
                auto const c2 = static_cast<int8_t>((value >> 16) & 0xFF);
                auto const c3 = static_cast<int8_t>((value >> 24) & 0xFF);
                uint32_t const r0 = static_cast<uint32_t>(c0 > 0 ? c0 : 0);
                uint32_t const r1 = static_cast<uint32_t>(c1 > 0 ? c1 : 0);
                uint32_t const r2 = static_cast<uint32_t>(c2 > 0 ? c2 : 0);
                uint32_t const r3 = static_cast<uint32_t>(c3 > 0 ? c3 : 0);
                output[i] = static_cast<int32_t>((r3 << 24) | (r2 << 16) | (r1 << 8) | r0);
            }
        }
    };

    /** Per-SIMD-package fp16 ReluGrad evaluated on the fp16 bit patterns.
     *
     * `(feature > 0) ? gradient : 0` depends only on the sign, magnitude and NaN-ness of the binary16 pattern, so it
     * is evaluated on the raw 16 bit lanes instead of converting to float and back. The branchless predicate
     * `(uint16_t)(bits - 1) < 0x7C00` is true exactly when the binary16 value is strictly positive:
     *   - `0x0000` and `0x8000` (both zeros) wrap to `0xFFFF`/`0x7FFF` and are rejected,
     *   - every negative pattern is `>= 0x8000` and rejected,
     *   - positive NaN payloads are `>= 0x7C00` after the decrement and rejected,
     *   - positive normals, positive subnormals and `+inf` (`0x7C00`) are accepted.
     * The remaining result is the gradient gated by the predicate `gradient & (predicate ? 0xFFFF : 0)`.
     */
    struct ReluGradOp
    {
        constexpr void operator()(auto const&, auto&& out, auto const& feature, auto const& gradient) const
        {
            auto const f = feature.load();
            auto const g = gradient.load();
            using SimdType = ALPAKA_TYPEOF(f);
            constexpr uint32_t width = SimdType::width();
            SimdType result = SimdType::fill(uint16_t{0});
            for(uint32_t i = 0u; i < width; ++i)
            {
                // (uint16_t)(bits - 1) < 0x7C00  <=>  binary16 value > 0 (see struct documentation)
                auto const bits = static_cast<uint16_t>(f[i]);
                bool const positive = static_cast<uint16_t>(bits - 1u) < 0x7C00u;
                result[i] = positive ? static_cast<uint16_t>(g[i]) : uint16_t{0};
            }
            out = result;
        }
    };

    /** fp16 ReluGrad: `(feature > 0) ? gradient : 0`, evaluated on the fp16 bit patterns.
     *
     * The operation is evaluated with the alpaka SIMD facilities so that wide vector loads/stores are issued on every
     * backend and the per-element software float conversion of the scalar implementation is avoided. One SIMD
     * register's worth of fp16 is processed per thread and invocation; the frame specification provides several such
     * invocations per thread so the memory system is kept busy. No backend branch or native kernel is used.
     */
    struct ReluGradKernel
    {
        ALPAKA_FN_ACC void operator()(
            auto const& acc,
            alpaka::concepts::IMdSpan auto backprop,
            alpaka::concepts::IMdSpan auto feature,
            alpaka::concepts::IMdSpan auto gradient) const
        {
            using namespace alpaka;
            // Concurrency budget of exactly one SIMD register width in bytes for the fp16 value type. This keeps one
            // vector load in flight per invocation and avoids the extra register pressure of multiple simultaneous
            // packages.
            constexpr uint32_t simdWidthInByte
                = getArchSimdWidth<uint16_t>(ALPAKA_TYPEOF(acc.getApi()){}, ALPAKA_TYPEOF(acc.getDeviceKind()){})
                  * sizeof(uint16_t);
            auto simdGrid = onAcc::SimdAlgo{onAcc::worker::threadsInGrid};
            simdGrid.template concurrent<simdWidthInByte>(
                acc,
                feature.getExtents(),
                ReluGradOp{},
                backprop,
                feature,
                gradient);
        }
    };

    /** Build a frame specification that gives every worker a useful amount of work.
     *
     * @param elementsPerWorker elements each worker should process. Pass 0 to use the device minimum. The value must
     *        be a multiple of the elements consumed by a single `onAcc::SimdAlgo::concurrent()` invocation, otherwise
     *        the SIMD loop is skipped and all work falls into the scalar remainder loop.
     */
    template<typename T_Data>
    auto makeFrameSpec(auto const& device, auto const& exec, uint32_t extent, uint32_t elementsPerWorker)
    {
        using namespace alpaka;
        if(elementsPerWorker == 0u)
            elementsPerWorker = getNumElemPerThread<T_Data>(device);
        constexpr uint32_t chunkSize = 256u;
        uint32_t const blocks = alpaka::divCeil(std::max(extent, 1u), chunkSize * elementsPerWorker);
        return onHost::FrameSpec{std::max(blocks, 1u), chunkSize, exec};
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
        double const median
            = samples.empty() ? 0.0 : samples[samples.size() / 2u];
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

        int result = EXIT_SUCCESS;

        // ---------------------------------------------------------------------------------------------
        // Relu: packed int8x4
        // ---------------------------------------------------------------------------------------------
        {
            Vec<int, 1u> const extent(count);
            auto hIn = onHost::allocHost<int32_t>(extent);
            auto hOut = onHost::allocHost<int32_t>(extent);
            auto hRef = onHost::allocHost<int32_t>(extent);

            std::mt19937 engine(19937);
            std::uniform_real_distribution<float> realDist(-1.f, 1.f);
            // The HeCBench reference consumes the fp16 random numbers before generating the integers. Reproduce the
            // same engine state so that the integer inputs match the reference exactly.
            for(int i = 0; i < count; ++i)
                (void) realDist(engine);
            std::uniform_int_distribution<unsigned char> intDist(0, 255);
            for(int i = 0; i < count; ++i)
            {
                hIn[i] = static_cast<int32_t>(
                    static_cast<unsigned>(intDist(engine)) | (static_cast<unsigned>(intDist(engine)) << 8)
                    | (static_cast<unsigned>(intDist(engine)) << 16)
                    | (static_cast<unsigned>(intDist(engine)) << 24));
                hRef[i] = relu::reluReference(hIn[i]);
            }

            auto dIn = onHost::alloc<int32_t>(device, extent);
            auto dOut = onHost::alloc<int32_t>(device, extent);
            onHost::memcpy(queue, dIn, hIn);

            auto const frameSpec = makeFrameSpec<int32_t>(device, exec, static_cast<uint32_t>(count), 0u);
            KernelBundle const kernel{ReluIntKernel{}, dOut, dIn, count};
            printf(
                "Relu (packed int8x4) kernel runtime median: %.3f us\n",
                timeKernel(queue, frameSpec, kernel, repeat));

            onHost::memcpy(queue, hOut, dOut);
            onHost::wait(queue);
            result |= relu::verifyInt(count, hOut, hIn);
        }

        // ---------------------------------------------------------------------------------------------
        // ReluGrad: fp16
        // ---------------------------------------------------------------------------------------------
        {
            Vec<int, 1u> const extent(count);
            auto hFeature = onHost::allocHost<uint16_t>(extent);
            auto hGradient = onHost::allocHost<uint16_t>(extent);
            auto hBackprop = onHost::allocHost<uint16_t>(extent);

            std::mt19937 engine(19937);
            std::uniform_real_distribution<float> realDist(-1.f, 1.f);
            for(int i = 0; i < count; ++i)
            {
                hFeature[i] = relu::floatToHalf(realDist(engine));
                hGradient[i] = relu::floatToHalf(1.0f);
            }

            auto dFeature = onHost::alloc<uint16_t>(device, extent);
            auto dGradient = onHost::alloc<uint16_t>(device, extent);
            auto dBackprop = onHost::alloc<uint16_t>(device, extent);
            onHost::memcpy(queue, dFeature, hFeature);
            onHost::memcpy(queue, dGradient, hGradient);

            // Each thread processes several SIMD registers of fp16 so that enough independent memory requests are in
            // flight to saturate the memory system, matching the work-per-thread of the HeCBench reference. The
            // per-thread element count is a multiple of the per-invocation consumption (one SIMD package) and is
            // derived from the device instead of hard-coded, so it adapts to every backend.
            uint32_t const elementsPerWorker = 4u * getNumElemPerThread<uint16_t>(device);
            auto const frameSpec
                = makeFrameSpec<uint16_t>(device, exec, static_cast<uint32_t>(count), elementsPerWorker);
            KernelBundle const kernel{ReluGradKernel{}, dBackprop, dFeature, dGradient};
            printf(
                "ReluGrad (fp16) kernel runtime median: %.3f us\n",
                timeKernel(queue, frameSpec, kernel, repeat));

            onHost::memcpy(queue, hBackprop, dBackprop);
            onHost::wait(queue);
            result |= relu::verifyReluGrad(count, hBackprop, hFeature, hGradient);
        }

        return result;
    }
} // namespace reluAlpaka

auto main(int argc, char* argv[]) -> int
{
    int count = 10000000;
    int repeat = 100;

    if(int const ret = relu::parseCmd(argc, argv, count, repeat))
        return ret;

    using namespace alpaka;

    return onHost::executeForEachIfHasDevice(
        [=](auto const& backend)
        { return reluAlpaka::example(onHost::makeDeviceSpec(backend), getExecutor(backend), count, repeat); },
        onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors));
}
