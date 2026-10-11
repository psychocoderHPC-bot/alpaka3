/* Copyright 2026 René Widera
 * SPDX-License-Identifier: Apache-2.0
 */

/** @file Portable, single-source alpaka implementation of the HeCBench `resize` benchmark.
 *
 * The pinned HeCBench reference (`src/resize-cuda/main.cu`, OpenMP twin `src/resize-omp/main.cpp`)
 * resizes a stack of `num_channels` single-channel images, each `in_height x in_width`, to
 * `out_height x out_width`, with a nearest-neighbour kernel (`resize`) and a bilinear kernel
 * (`resize_bilinear`). `resize_image<T>` runs each kernel `repeat` times and reports the average
 * kernel time.
 *
 * This port keeps the reference pipeline exactly:
 *  - the same linear work-item decomposition: `iters_required = (num_channels / 8) * out_image_size`
 *    and `c_start = (iter / out_image_size) * 8`, i.e. channels beyond the last full group of eight
 *    are not written;
 *  - the same **integer-division** scaling ratios `in_width / out_width`, `in_height / out_height`
 *    (converted to float), and the reference's swapped argument order `(o2i_fy, o2i_fx) = (fx, fy)`;
 *  - the same nearest mapping `lroundf((idx + 0.5f) * ratio)` and bilinear mapping
 *    `fmaxf((idx + 0.5f) * ratio - 0.5f, 0)`;
 *  - the same bilinear update
 *    `v00 + T(fy)*T(v10-v00) + T(fx)*T(v01-v00) + T(fy)*T(fx)*T(v11-v01-v10+v00)`, evaluated in the
 *    pixel type `T` (the integer weight cast is what makes the reference's bilinear kernel degenerate
 *    to `v00` for the integer types);
 *  - an independent host reference recomputed from the same input and an exact (tolerance 0)
 *    element-wise comparison, plus explicit non-finite checks.
 *
 * One kernel template drives the host CPU backend and every device backend. The launch
 * configuration changes per backend, the algorithm does not. The reference sources are kept outside
 * this repository and are not vendored.
 */

#include <alpaka/alpaka.hpp>

#include "helper.hpp"
#include "misc.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <type_traits>
#include <vector>

namespace resizeAlpaka
{
    using namespace resizeBench;

    /** Nearest-neighbour resize kernel (portable).
     *
     * `o2i_fy` receives `in_width / out_width` and `o2i_fx` receives `in_height / out_height`, exactly
     * as `resize_image` passes them to the reference kernel.
     */
    template<typename T>
    struct NearestKernel
    {
        ALPAKA_FN_ACC void operator()(
            auto const& acc,
            alpaka::concepts::IMdSpan auto input,
            alpaka::concepts::IMdSpan auto output,
            std::uint32_t inWidth,
            std::uint32_t inHeight,
            std::uint32_t outWidth,
            std::uint32_t outHeight,
            std::uint32_t numChannels,
            float o2i_fy,
            float o2i_fx) const
        {
            using namespace alpaka;

            // Matches the reference's 32-bit index arithmetic (`int in_image_size`, `int in_idx`).
            int const inImageSize = static_cast<int>(inWidth) * static_cast<int>(inHeight);
            int const outImageSize = static_cast<int>(outWidth) * static_cast<int>(outHeight);
            int const numChannelGroups = static_cast<int>(numChannels) / static_cast<int>(channelsPerIter);
            int const itersRequired = numChannelGroups * outImageSize;

            for(auto flatIdx : onAcc::makeIdxMap(
                    acc,
                    onAcc::worker::threadsInGrid,
                    IdxRange{Vec{itersRequired}}))
            {
                int const iter = static_cast<int>(flatIdx.x());
                int const cStart = (iter / outImageSize) * static_cast<int>(channelsPerIter);
                int const y = (iter % outImageSize) / static_cast<int>(outWidth);
                int const x = iter % static_cast<int>(outWidth);

                int inY = static_cast<int>(math::lround((static_cast<float>(y) + 0.5f) * o2i_fy));
                int inX = static_cast<int>(math::lround((static_cast<float>(x) + 0.5f) * o2i_fx));
                inY = math::min(inY, static_cast<int>(inHeight) - 1);
                inX = math::min(inX, static_cast<int>(inWidth) - 1);
                inY = math::max(inY, 0);
                inX = math::max(inX, 0);

                int inIdx = cStart * inImageSize + inY * static_cast<int>(inWidth) + inX;
                int outIdx = cStart * outImageSize + y * static_cast<int>(outWidth) + x;

                // Gather all CHANNELS_PER_ITER values before writing any of them. This mirrors the
                // reference's `__restrict__` guarantee and keeps the eight independent loads from being
                // serialised by a presumed load/store alias between the input and output buffers.
                T values[channelsPerIter];
                for(std::uint32_t i = 0; i < channelsPerIter; ++i)
                {
                    values[i] = input[inIdx];
                    inIdx += inImageSize;
                }
                for(std::uint32_t i = 0; i < channelsPerIter; ++i)
                {
                    output[outIdx] = values[i];
                    outIdx += outImageSize;
                }
            }
        }
    };

    /** Bilinear resize kernel (portable).
     *
     * Reproduces the reference's integer-weight cast semantics through `static_cast<T>` of the float
     * fractional parts.
     */
    template<typename T>
    struct BilinearKernel
    {
        ALPAKA_FN_ACC void operator()(
            auto const& acc,
            alpaka::concepts::IMdSpan auto input,
            alpaka::concepts::IMdSpan auto output,
            std::uint32_t inWidth,
            std::uint32_t inHeight,
            std::uint32_t outWidth,
            std::uint32_t outHeight,
            std::uint32_t numChannels,
            float o2i_fy,
            float o2i_fx) const
        {
            using namespace alpaka;

            // Matches the reference's 32-bit index arithmetic (`int in_image_size`, `int in_idx`).
            int const inImageSize = static_cast<int>(inWidth) * static_cast<int>(inHeight);
            int const outImageSize = static_cast<int>(outWidth) * static_cast<int>(outHeight);
            int const numChannelGroups = static_cast<int>(numChannels) / static_cast<int>(channelsPerIter);
            int const itersRequired = numChannelGroups * outImageSize;

            for(auto flatIdx : onAcc::makeIdxMap(
                    acc,
                    onAcc::worker::threadsInGrid,
                    IdxRange{Vec{itersRequired}}))
            {
                int const iter = static_cast<int>(flatIdx.x());
                int const cStart = (iter / outImageSize) * static_cast<int>(channelsPerIter);
                int const y = (iter % outImageSize) / static_cast<int>(outWidth);
                int const x = iter % static_cast<int>(outWidth);

                float inYf = (static_cast<float>(y) + 0.5f) * o2i_fy - 0.5f;
                inYf = inYf > 0.0f ? inYf : 0.0f;
                float inXf = (static_cast<float>(x) + 0.5f) * o2i_fx - 0.5f;
                inXf = inXf > 0.0f ? inXf : 0.0f;

                int const inY0 = static_cast<int>(inYf);
                int const inX0 = static_cast<int>(inXf);
                int const inX1 = math::min(inX0 + 1, static_cast<int>(inWidth) - 1);
                int const inY1 = math::min(inY0, static_cast<int>(inHeight) - 1);
                int const inY2 = math::min(inY0 + 1, static_cast<int>(inHeight) - 1);

                T const wx = static_cast<T>(inXf - static_cast<float>(inX0));
                T const wy = static_cast<T>(inYf - static_cast<float>(inY0));

                int inOffsetR0 = cStart * inImageSize + inY1 * static_cast<int>(inWidth);
                int inOffsetR1 = cStart * inImageSize + inY2 * static_cast<int>(inWidth);
                int outIdx = cStart * outImageSize + y * static_cast<int>(outWidth) + x;

                // Gather all CHANNELS_PER_ITER results before writing them, mirroring the reference's
                // `__restrict__` guarantee and breaking a presumed load/store alias.
                T values[channelsPerIter];
                for(std::uint32_t i = 0; i < channelsPerIter; ++i)
                {
                    T const v00 = input[inOffsetR0 + inX0];
                    T const v01 = input[inOffsetR0 + inX1];
                    T const v10 = input[inOffsetR1 + inX0];
                    T const v11 = input[inOffsetR1 + inX1];

                    values[i] = static_cast<T>(
                        v00 + wy * static_cast<T>(v10 - v00) + wx * static_cast<T>(v01 - v00)
                        + wy * wx * static_cast<T>(v11 - v01 - v10 + v00));

                    inOffsetR0 += inImageSize;
                    inOffsetR1 += inImageSize;
                }
                for(std::uint32_t i = 0; i < channelsPerIter; ++i)
                {
                    output[outIdx] = values[i];
                    outIdx += outImageSize;
                }
            }
        }
    };

    struct Stats
    {
        double median = 0.0;
        double min = 0.0;
        double max = 0.0;
        double spread = 0.0;
    };

    inline Stats summarize(std::vector<double> samples)
    {
        Stats s;
        std::sort(samples.begin(), samples.end());
        s.min = samples.front();
        s.max = samples.back();
        std::size_t const n = samples.size();
        s.median = (n % 2u == 0u) ? 0.5 * (samples[n / 2u - 1u] + samples[n / 2u]) : samples[n / 2u];
        s.spread = (s.median > 0.0) ? (s.max - s.min) / s.median : 0.0;
        return s;
    }

    //! One resize pass (either the nearest or the bilinear kernel) for pixel type `T`.
    template<typename T, bool T_bilinear, typename T_Queue, typename T_Exec, typename T_In, typename T_Out>
    void runOnce(
        T_Queue& queue,
        T_Exec const& exec,
        T_In& dInput,
        T_Out& dOutput,
        std::uint32_t inWidth,
        std::uint32_t inHeight,
        std::uint32_t outWidth,
        std::uint32_t outHeight,
        std::uint32_t numChannels,
        float o2i_fy,
        float o2i_fx)
    {
        using namespace alpaka;

        // Flat 1D work-item decomposition, exactly like the reference. The reference launches a
        // grid-stride loop over `(num_channels / 8) * out_image_size` work items; the frame spec below
        // provides the same iteration space and the kernels use the reference's 32-bit index math.
        std::uint32_t const outImageSize = outWidth * outHeight;
        std::uint32_t const itersRequired = (numChannels / channelsPerIter) * outImageSize;
        std::uint32_t const numFrames
            = itersRequired > 0u ? static_cast<std::uint32_t>(divExZero(itersRequired, 256u)) : 1u;
        auto const frameSpec = onHost::FrameSpec{Vec{numFrames}, Vec{256u}, exec};

        if constexpr(T_bilinear)
            queue.enqueue(
                frameSpec,
                KernelBundle{
                    BilinearKernel<T>{},
                    dInput,
                    dOutput,
                    inWidth,
                    inHeight,
                    outWidth,
                    outHeight,
                    numChannels,
                    o2i_fy,
                    o2i_fx});
        else
            queue.enqueue(
                frameSpec,
                KernelBundle{
                    NearestKernel<T>{},
                    dInput,
                    dOutput,
                    inWidth,
                    inHeight,
                    outWidth,
                    outHeight,
                    numChannels,
                    o2i_fy,
                    o2i_fx});
    }

    template<typename T>
    std::size_t typeSize()
    {
        return sizeof(T);
    }

    /** Run and validate one (pixel type, mode) combination. Returns EXIT_SUCCESS/EXIT_FAILURE. */
    template<typename T, typename T_Device, typename T_Queue, typename T_Exec>
    int runCase(
        T_Device& device,
        T_Queue& queue,
        T_Exec const& exec,
        resizeBench::Params const& params,
        std::string const& typeName,
        bool bilinear)
    {
        using namespace alpaka;

        std::uint32_t const inWidth = params.inWidth;
        std::uint32_t const inHeight = params.inHeight;
        std::uint32_t const outWidth = params.outWidth;
        std::uint32_t const outHeight = params.outHeight;
        std::uint32_t const numChannels = params.channels;

        std::size_t const inSize = static_cast<std::size_t>(numChannels) * inWidth * inHeight;
        std::size_t const outSize = static_cast<std::size_t>(numChannels) * outWidth * outHeight;

        std::vector<T> hInput(inSize);
        resizeBench::generateInput(hInput.data(), inSize);

        // Host reference; output is zero-initialised exactly like the reference's cudaMemset.
        std::vector<T> hReference(outSize, T{0});
        if(bilinear)
            resizeBench::resizeBilinearGold(hInput, hReference, inWidth, inHeight, outWidth, outHeight, numChannels);
        else
            resizeBench::resizeNearestGold(hInput, hReference, inWidth, inHeight, outWidth, outHeight, numChannels);

        if(int rc = resizeBench::checkSane(hInput, hReference, "reference"); rc != EXIT_SUCCESS)
            return rc;

        auto hInputDev = onHost::allocHost<T>(inSize);
        std::copy(hInput.begin(), hInput.end(), onHost::data(hInputDev));
        auto hOutput = onHost::allocHost<T>(outSize);

        auto dInput = onHost::alloc<T>(device, inSize);
        auto dOutput = onHost::alloc<T>(device, outSize);

        onHost::memcpy(queue, dInput, hInputDev);
        onHost::memset(queue, dOutput, static_cast<std::uint8_t>(0));
        onHost::wait(queue);

        float const o2i_fy = static_cast<float>(inWidth / outWidth);
        float const o2i_fx = static_cast<float>(inHeight / outHeight);

        std::string const mode = bilinear ? "bilinear" : "nearest";

        if(!params.verifyOnly)
        {
            auto pass = [&](int repeat)
            {
                if(bilinear)
                    for(int r = 0; r < repeat; ++r)
                        runOnce<T, true>(
                            queue,
                            exec,
                            dInput,
                            dOutput,
                            inWidth,
                            inHeight,
                            outWidth,
                            outHeight,
                            numChannels,
                            o2i_fy,
                            o2i_fx);
                else
                    for(int r = 0; r < repeat; ++r)
                        runOnce<T, false>(
                            queue,
                            exec,
                            dInput,
                            dOutput,
                            inWidth,
                            inHeight,
                            outWidth,
                            outHeight,
                            numChannels,
                            o2i_fy,
                            o2i_fx);
            };

            pass(1); // warm-up
            onHost::wait(queue);

            std::vector<double> samples;
            samples.reserve(static_cast<std::size_t>(params.samples));
            for(int s = 0; s < params.samples; ++s)
            {
                onHost::wait(queue);
                auto const start = std::chrono::steady_clock::now();
                pass(params.repeat);
                onHost::wait(queue);
                auto const end = std::chrono::steady_clock::now();
                double const ns
                    = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
                samples.push_back(ns * 1e-3 / static_cast<double>(params.repeat));
            }

            Stats const stats = summarize(samples);
            std::printf(
                "RESULT type=%s mode=%s size=%ux%u->%ux%u channels=%u median_us=%.4f min_us=%.4f "
                "max_us=%.4f relspread=%.4f\n",
                typeName.c_str(),
                mode.c_str(),
                inWidth,
                inHeight,
                outWidth,
                outHeight,
                numChannels,
                stats.median,
                stats.min,
                stats.max,
                stats.spread);
            std::printf("  raw_samples_us=");
            for(std::size_t i = 0; i < samples.size(); ++i)
                std::printf("%s%.4f", i == 0 ? "" : ",", samples[i]);
            std::printf("\n");
        }

        // Validation: one fresh pass, copy back, compare exactly.
        if(bilinear)
            runOnce<T, true>(
                queue,
                exec,
                dInput,
                dOutput,
                inWidth,
                inHeight,
                outWidth,
                outHeight,
                numChannels,
                o2i_fy,
                o2i_fx);
        else
            runOnce<T, false>(
                queue,
                exec,
                dInput,
                dOutput,
                inWidth,
                inHeight,
                outWidth,
                outHeight,
                numChannels,
                o2i_fy,
                o2i_fx);
        onHost::memcpy(queue, hOutput, dOutput);
        onHost::wait(queue);

        std::vector<T> computed(onHost::data(hOutput), onHost::data(hOutput) + outSize);

        if(int rc = resizeBench::checkSane(hInput, computed, "output"); rc != EXIT_SUCCESS)
            return rc;
        if(int rc = resizeBench::verify(computed, hReference, "resize"); rc != EXIT_SUCCESS)
        {
            std::printf("resize %s/%s verification FAILED\n", typeName.c_str(), mode.c_str());
            return rc;
        }
        std::printf("resize %s/%s verification PASSED\n", typeName.c_str(), mode.c_str());
        return EXIT_SUCCESS;
    }

    auto example(auto const& deviceSpec, auto const& exec, resizeBench::Params const& params) -> int
    {
        using namespace alpaka;

        auto devSelector = onHost::makeDeviceSelector(deviceSpec);
        if(!devSelector.isAvailable())
            return EXIT_SUCCESS;
        onHost::Device device = devSelector.makeDevice(0);
        std::string const backend = deviceSpec.getApi().getName();

        std::printf(
            "Using alpaka backend %s on %s, in=%ux%u out=%ux%u channels=%u repeat=%d samples=%d types=%s "
            "modes=%s\n",
            backend.c_str(),
            deviceSpec.getDeviceKind().getName().c_str(),
            params.inWidth,
            params.inHeight,
            params.outWidth,
            params.outHeight,
            params.channels,
            params.repeat,
            params.samples,
            params.typeName().c_str(),
            params.modeName().c_str());

        auto queue = device.makeQueue(queueKind::blocking);

        int rc = EXIT_SUCCESS;
        for(bool bilinear : params.modes())
        {
            for(std::uint32_t type : params.types())
            {
                int caseRc = EXIT_SUCCESS;
                switch(type)
                {
                case 1u:
                    caseRc = runCase<std::uint8_t>(device, queue, exec, params, "u8", bilinear);
                    break;
                case 2u:
                    caseRc = runCase<std::uint16_t>(device, queue, exec, params, "u16", bilinear);
                    break;
                default:
                    caseRc = runCase<std::uint32_t>(device, queue, exec, params, "u32", bilinear);
                    break;
                }
                if(caseRc != EXIT_SUCCESS)
                    rc = EXIT_FAILURE;
            }
        }
        return rc;
    }
} // namespace resizeAlpaka

auto main(int argc, char* argv[]) -> int
{
    resizeBench::Params params;
    if(int const ret = resizeBench::parseCmd(argc, argv, params))
        return ret;

    using namespace alpaka;

    return onHost::executeForEachIfHasDevice(
        [&](auto const& backend)
        { return resizeAlpaka::example(onHost::makeDeviceSpec(backend), getExecutor(backend), params); },
        onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors));
}
