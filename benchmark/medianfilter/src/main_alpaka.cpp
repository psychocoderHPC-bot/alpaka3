/* Copyright 2026 René Widera
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** @file Portable, single-source alpaka implementation of the HeCBench `medianfilter` benchmark.
 *
 * The pinned HeCBench reference (`src/medianfilter-cuda/main.cu` and `MedianFilter.cu`, with the
 * OpenMP twin `src/medianfilter-omp/main.cpp`) applies a 2D 3x3 median filter to an RGBA image. Each
 * output pixel and each of the three colour channels is the median of the 3x3 neighbourhood computed
 * with an 8-iteration integer binary search over the 256 grey levels; alpha is dropped. Pixels outside
 * the image contribute zero. The harness compares the device result against `MedianFilterHost` with a
 * small tolerance (`shrCompareuit(..., 1.0f, 0.0001f)`), because the binary search can be off by one
 * grey level.
 *
 * The reference device kernel stages a 16x4 tile with a one-pixel apron in shared memory and pads the
 * border with zero. This port keeps the same algorithm, window and border semantics, but expresses the
 * median per output pixel with the identical binary search, so one kernel source drives the host CPU
 * backend and every device backend. There is no native CUDA/HIP kernel and no backend branch.
 *
 * An independent host reference (the same binary search on the host, `helper.hpp`) is recomputed from
 * the same input and compared exactly. The exact sorted median is also computed as a second oracle to
 * pin the binary search to within one grey level (see the README and `helper.hpp`).
 *
 * The reference sources are kept outside this repository and are not vendored.
 */

#include <alpaka/alpaka.hpp>

#include "helper.hpp"
#include "misc.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace medianfilterAlpaka
{
    using namespace medianfilterBench;

    //! Logical elements (pixels) per frame used to describe the parallelism.
    constexpr std::uint32_t frameExtent = 256u;

    /** 3x3 median filter kernel.
     *
     * Every output pixel `(x, y)` and every colour channel is the median of the 3x3 window centred on
     * it, with zero padding outside the image and an 8-iteration integer binary search identical to the
     * reference device and host code. The alpha channel is left zero.
     */
    struct MedianKernel
    {
        ALPAKA_FN_ACC void operator()(
            auto const& acc,
            alpaka::concepts::IMdSpan auto input,
            alpaka::concepts::IMdSpan auto output,
            std::uint32_t width,
            std::uint32_t height) const
        {
            using namespace alpaka;

            for(auto flatIdx :
                onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{Vec{width * height}}))
            {
                std::uint32_t const flat = flatIdx.x();
                std::uint32_t const y = flat / width;
                std::uint32_t const x = flat % width;

                // Stage the zero-padded 3x3 window once per pixel (3 channels x 9 taps). The reference
                // kernel tiles the window into shared memory and then reads it repeatedly; reading the
                // taps from global memory inside the 8-iteration binary search would instead issue 216
                // device loads per pixel. Sampling once into registers is bit-for-bit equivalent.
                std::array<std::uint8_t, 27> window{};
                int n = 0;
                for(std::int32_t dy = -1; dy <= 1; ++dy)
                {
                    for(std::int32_t dx = -1; dx <= 1; ++dx)
                    {
                        std::int32_t const xx = static_cast<std::int32_t>(x) + dx;
                        std::int32_t const yy = static_cast<std::int32_t>(y) + dy;
                        std::uint32_t pixel = 0u;
                        if(xx >= 0 && xx < static_cast<std::int32_t>(width) && yy >= 0
                           && yy < static_cast<std::int32_t>(height))
                            pixel = input[static_cast<std::uint32_t>(yy) * width + static_cast<std::uint32_t>(xx)];
                        // RGB in bits 0/8/16; alpha (bit 24) is dropped, as in the reference.
                        window[static_cast<std::size_t>(n) + 0u] = static_cast<std::uint8_t>(pixel & 0xffu);
                        window[static_cast<std::size_t>(n) + 9u] = static_cast<std::uint8_t>((pixel >> 8) & 0xffu);
                        window[static_cast<std::size_t>(n) + 18u] = static_cast<std::uint8_t>((pixel >> 16) & 0xffu);
                        ++n;
                    }
                }

                // Identical 8-iteration integer binary search as the reference, now over the staged window.
                auto binarySearch = [&](int channel) -> std::uint32_t
                {
                    std::size_t const base = static_cast<std::size_t>(channel) * 9u;
                    float estimate = 128.0f;
                    float minBound = 0.0f;
                    float maxBound = 255.0f;
                    for(int search = 0; search < 8; ++search)
                    {
                        unsigned int highCount = 0u;
                        for(std::size_t i = 0; i < 9; ++i)
                            highCount += (estimate < static_cast<float>(window[base + i])) ? 1u : 0u;

                        if(highCount > 4u)
                            minBound = estimate;
                        else
                            maxBound = estimate;
                        estimate = 0.5f * (maxBound + minBound);
                    }
                    return static_cast<std::uint32_t>(static_cast<unsigned int>(estimate + 0.5f) & 0xffu);
                };

                std::uint32_t packed = 0u;
                for(int channel = 0; channel < 3; ++channel)
                    packed |= (binarySearch(channel) & 0xffu) << (8 * channel);

                output[flat] = packed;
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

    template<typename T_Queue, typename T_Exec>
    void runOnce(T_Queue& queue, T_Exec const& exec, auto& dInput, auto& dOutput, std::uint32_t width, std::uint32_t height)
    {
        using namespace alpaka;

        std::uint32_t const numPixels = width * height;
        auto const frameSpec = onHost::FrameSpec{divExZero(numPixels, frameExtent), frameExtent, exec};

        queue.enqueue(frameSpec, KernelBundle{MedianKernel{}, dInput, dOutput, width, height});
    }

    auto example(auto const& deviceSpec, auto const& exec, medianfilterBench::Params const& params) -> int
    {
        using namespace alpaka;

        auto devSelector = onHost::makeDeviceSelector(deviceSpec);
        if(!devSelector.isAvailable())
            return EXIT_SUCCESS;
        onHost::Device device = devSelector.makeDevice(0);
        std::string const backend = deviceSpec.getApi().getName();

        std::uint32_t const width = params.width;
        std::uint32_t const height = params.height;
        std::size_t const numPixels = static_cast<std::size_t>(width) * height;

        std::printf(
            "Using alpaka backend %s on %s, width=%u, height=%u, repeat=%d, samples=%d\n",
            backend.c_str(),
            deviceSpec.getDeviceKind().getName().c_str(),
            width,
            height,
            params.repeat,
            params.samples);

        // Host input image, packed exactly like the reference `shrLoadPPM4ub`: RGBA bytes with a zero
        // alpha channel.
        auto hInput = onHost::allocHost<Uchar4>(numPixels);
        medianfilterBench::generateImage(onHost::data(hInput), numPixels, 0xC0FFEEu);

        auto hPacked = onHost::allocHost<std::uint32_t>(numPixels);
        for(std::size_t i = 0; i < numPixels; ++i)
        {
            hPacked[i] = static_cast<std::uint32_t>(hInput[i].x) | (static_cast<std::uint32_t>(hInput[i].y) << 8)
                         | (static_cast<std::uint32_t>(hInput[i].z) << 16)
                         | (static_cast<std::uint32_t>(hInput[i].w) << 24);
        }

        // Independent host references: the reference binary search and the exact sorted median.
        auto hRef = onHost::allocHost<std::uint32_t>(numPixels);
        auto hSorted = onHost::allocHost<std::uint32_t>(numPixels);
        medianfilterBench::medianGold(onHost::data(hInput), width, height, onHost::data(hRef));
        medianfilterBench::medianSorted(onHost::data(hInput), width, height, onHost::data(hSorted));

        std::size_t numDelta = 0;
        std::size_t maxDelta = 0;
        if(int rc = medianfilterBench::verifyReferenceAgainstSorted(
               onHost::data(hRef),
               onHost::data(hSorted),
               width,
               height,
               &numDelta,
               &maxDelta,
               "reference")
           != EXIT_SUCCESS)
            return rc;
        std::printf(
            "host reference binary-search vs sorted: %zu/%zu pixels differ, max channel delta=%zu\n",
            numDelta,
            numPixels,
            maxDelta);

        auto dInput = onHost::alloc<std::uint32_t>(device, numPixels);
        auto dOutput = onHost::alloc<std::uint32_t>(device, numPixels);
        auto hOutput = onHost::allocHost<std::uint32_t>(numPixels);

        auto queue = device.makeQueue(queueKind::blocking);
        onHost::memcpy(queue, dInput, hPacked);
        onHost::wait(queue);

        if(!params.verifyOnly)
        {
            // warm-up
            runOnce(queue, exec, dInput, dOutput, width, height);
            onHost::wait(queue);

            std::vector<double> samples;
            samples.reserve(static_cast<std::size_t>(params.samples));
            for(int s = 0; s < params.samples; ++s)
            {
                onHost::wait(queue);
                auto const start = std::chrono::steady_clock::now();
                for(int r = 0; r < params.repeat; ++r)
                    runOnce(queue, exec, dInput, dOutput, width, height);
                onHost::wait(queue);
                auto const end = std::chrono::steady_clock::now();
                double const ns
                    = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
                samples.push_back(ns * 1e-3 / static_cast<double>(params.repeat));
            }

            Stats const stats = summarize(samples);
            std::printf(
                "RESULT backend=%s size=%ux%u median_us=%.4f min_us=%.4f max_us=%.4f relspread=%.4f\n",
                backend.c_str(),
                width,
                height,
                stats.median,
                stats.min,
                stats.max,
                stats.spread);
            std::printf("  raw_samples_us=");
            for(std::size_t i = 0; i < samples.size(); ++i)
                std::printf("%s%.4f", i == 0 ? "" : ",", samples[i]);
            std::printf("\n");
        }

        // Validation: one fresh run, copied back, compared exactly against the host reference.
        runOnce(queue, exec, dInput, dOutput, width, height);
        onHost::memcpy(queue, hOutput, dOutput, numPixels);
        onHost::wait(queue);

        int rc = medianfilterBench::checkFinite(
            onHost::data(hInput),
            onHost::data(hOutput),
            width,
            height,
            "output");
        if(rc != EXIT_SUCCESS)
            return rc;

        rc = medianfilterBench::verify(
            onHost::data(hOutput),
            onHost::data(hRef),
            width,
            height,
            "medianfilter");
        std::printf("medianfilter verification %s\n", rc == EXIT_SUCCESS ? "PASSED" : "FAILED");
        return rc;
    }
} // namespace medianfilterAlpaka

auto main(int argc, char* argv[]) -> int
{
    medianfilterBench::Params params;
    if(int const ret = medianfilterBench::parseCmd(argc, argv, params))
        return ret;

    using namespace alpaka;

    return onHost::executeForEachIfHasDevice(
        [&](auto const& backend)
        { return medianfilterAlpaka::example(onHost::makeDeviceSpec(backend), getExecutor(backend), params); },
        onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors));
}
