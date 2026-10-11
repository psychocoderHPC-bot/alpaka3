/* Copyright 2026 René Widera
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** @file Portable, single-source alpaka implementation of the HeCBench `boxfilter` benchmark.
 *
 * The pinned HeCBench reference (`src/boxfilter-cuda/main.cu` and its OpenMP twin
 * `src/boxfilter-omp/main.cpp`) applies a separable 2D box blur of radius `RADIUS = 10` to an RGBA
 * image: a row pass writes an intermediate image with each channel replaced by the truncated average
 * of its horizontal window, and a column pass averages that intermediate image vertically. The
 * reference device and host (`BoxFilterHost` in `src/boxfilter-sycl/reference.cpp`) use the identical
 * arithmetic, and the harness compares the device result against the host result exactly.
 *
 * This port keeps the reference pipeline exactly:
 *  - the same 32-bit packed RGBA representation, decoded per channel,
 *  - the same row pass with zero padding outside the image horizontally,
 *  - the same column pass with a replicate (clamped) vertical window,
 *  - the same truncated integer quantization `floor(sum / (2r+1))`, which for the pinned radius
 *    coincides with the reference's single-precision scale (see README),
 *  - an independent host reference recomputed from the same input and an exact comparison over the
 *    ROI the HeCBench harness checks (`rows [radius, height-radius)`).
 *
 * One kernel source runs on the host CPU and on NVIDIA GPUs: a row kernel and a column kernel are
 * enqueued back to back, mirroring the two reference kernels. The launch configuration changes per
 * backend, the algorithm does not.
 *
 * The reference sources are kept outside this repository and are not vendored.
 */

#include <alpaka/alpaka.hpp>

#include "helper.hpp"
#include "misc.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace boxfilterAlpaka
{
    using namespace boxfilterBench;

    //! Logical elements (pixels) per frame used to describe the parallelism.
    constexpr std::uint32_t frameExtent = 256u;

    /** Horizontal (row) pass.
     *
     * Every output pixel `(x, y)` receives the truncated average of the input window
     * `[max(0,x-radius), min(width-1,x+radius)]` on its own row; pixels beyond a row end contribute
     * zero, matching the reference shared-memory padding.
     */
    struct RowKernel
    {
        ALPAKA_FN_ACC void operator()(
            auto const& acc,
            alpaka::concepts::IMdSpan auto input,
            alpaka::concepts::IMdSpan auto tmp,
            std::uint32_t width,
            std::uint32_t height,
            std::int32_t radius) const
        {
            using namespace alpaka;

            for(auto flatIdx : onAcc::makeIdxMap(
                    acc,
                    onAcc::worker::threadsInGrid,
                    IdxRange{Vec{width * height}}))
            {
                std::uint32_t const flat = flatIdx.x();
                std::uint32_t const y = flat / width;
                std::uint32_t const x = flat % width;

                std::int32_t const lo = (static_cast<std::int32_t>(x) > radius) ? static_cast<std::int32_t>(x) - radius : 0;
                std::int32_t hi = static_cast<std::int32_t>(x) + radius;
                if(hi >= static_cast<std::int32_t>(width))
                    hi = static_cast<std::int32_t>(width) - 1;

                std::uint32_t s0 = 0u, s1 = 0u, s2 = 0u, s3 = 0u;
                std::uint32_t const rowBase = y * width;
                for(std::int32_t xx = lo; xx <= hi; ++xx)
                {
                    std::uint32_t const c = input[rowBase + static_cast<std::uint32_t>(xx)];
                    s0 += c & 0xffu;
                    s1 += (c >> 8) & 0xffu;
                    s2 += (c >> 16) & 0xffu;
                    s3 += (c >> 24) & 0xffu;
                }

                std::uint32_t const divisor = 2u * static_cast<std::uint32_t>(radius) + 1u;
                std::uint32_t packed = 0u;
                packed |= ((s0 / divisor) & 0xffu);
                packed |= (((s1 / divisor) & 0xffu) << 8);
                packed |= (((s2 / divisor) & 0xffu) << 16);
                packed |= (((s3 / divisor) & 0xffu) << 24);
                tmp[flat] = packed;
            }
        }
    };

    /** Vertical (column) pass.
     *
     * Every output pixel `(x, y)` receives the truncated average of the intermediate window
     * `[max(0,y-radius), min(height-1,y+radius)]` in its own column (replicate boundary).
     */
    struct ColKernel
    {
        ALPAKA_FN_ACC void operator()(
            auto const& acc,
            alpaka::concepts::IMdSpan auto tmp,
            alpaka::concepts::IMdSpan auto output,
            std::uint32_t width,
            std::uint32_t height,
            std::int32_t radius) const
        {
            using namespace alpaka;

            for(auto flatIdx : onAcc::makeIdxMap(
                    acc,
                    onAcc::worker::threadsInGrid,
                    IdxRange{Vec{width * height}}))
            {
                std::uint32_t const flat = flatIdx.x();
                std::uint32_t const y = flat / width;
                std::uint32_t const x = flat % width;

                std::int32_t const lo = (static_cast<std::int32_t>(y) > radius) ? static_cast<std::int32_t>(y) - radius : 0;
                std::int32_t hi = static_cast<std::int32_t>(y) + radius;
                if(hi >= static_cast<std::int32_t>(height))
                    hi = static_cast<std::int32_t>(height) - 1;

                std::uint32_t s0 = 0u, s1 = 0u, s2 = 0u, s3 = 0u;
                for(std::int32_t yy = lo; yy <= hi; ++yy)
                {
                    std::uint32_t const c = tmp[static_cast<std::uint32_t>(yy) * width + x];
                    s0 += c & 0xffu;
                    s1 += (c >> 8) & 0xffu;
                    s2 += (c >> 16) & 0xffu;
                    s3 += (c >> 24) & 0xffu;
                }

                std::uint32_t const divisor = 2u * static_cast<std::uint32_t>(radius) + 1u;
                std::uint32_t packed = 0u;
                packed |= ((s0 / divisor) & 0xffu);
                packed |= (((s1 / divisor) & 0xffu) << 8);
                packed |= (((s2 / divisor) & 0xffu) << 16);
                packed |= (((s3 / divisor) & 0xffu) << 24);
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

    /** Run the complete filter once (row pass then column pass). */
    template<typename T_Queue, typename T_Exec>
    void runOnce(
        T_Queue& queue,
        T_Exec const& exec,
        auto& dInput,
        auto& dTmp,
        auto& dOutput,
        std::uint32_t width,
        std::uint32_t height,
        std::int32_t radius)
    {
        using namespace alpaka;

        std::uint32_t const numPixels = width * height;
        auto const frameSpec
            = onHost::FrameSpec{divExZero(numPixels, frameExtent), frameExtent, exec};

        queue.enqueue(frameSpec, KernelBundle{RowKernel{}, dInput, dTmp, width, height, radius});
        queue.enqueue(frameSpec, KernelBundle{ColKernel{}, dTmp, dOutput, width, height, radius});
    }

    auto example(auto const& deviceSpec, auto const& exec, boxfilterBench::Params const& params) -> int
    {
        using namespace alpaka;

        auto devSelector = onHost::makeDeviceSelector(deviceSpec);
        if(!devSelector.isAvailable())
            return EXIT_SUCCESS;
        onHost::Device device = devSelector.makeDevice(0);
        std::string const backend = deviceSpec.getApi().getName();

        std::uint32_t const width = params.width;
        std::uint32_t const height = params.height;
        std::int32_t const radius = params.radius;
        std::size_t const numPixels = static_cast<std::size_t>(width) * height;

        if(height <= 2u * static_cast<std::uint32_t>(radius))
        {
            std::printf(
                "error: height %u must be greater than 2*radius (%d) for a non-empty ROI\n",
                height,
                2 * radius);
            return EXIT_FAILURE;
        }

        std::printf(
            "Using alpaka backend %s on %s, width=%u, height=%u, radius=%d, repeat=%d, samples=%d\n",
            backend.c_str(),
            deviceSpec.getDeviceKind().getName().c_str(),
            width,
            height,
            radius,
            params.repeat,
            params.samples);

        // Host input image, packed exactly like the reference `shrLoadPPM4ub`: RGBA bytes with a zero
        // alpha channel packed into one 32-bit unsigned integer per pixel.
        auto hInput = onHost::allocHost<Uchar4>(numPixels);
        boxfilterBench::generateImage(onHost::data(hInput), numPixels, 0xC0FFEEu);

        auto hPacked = onHost::allocHost<std::uint32_t>(numPixels);
        for(std::size_t i = 0; i < numPixels; ++i)
        {
            hPacked[i] = static_cast<std::uint32_t>(hInput[i].x) | (static_cast<std::uint32_t>(hInput[i].y) << 8)
                         | (static_cast<std::uint32_t>(hInput[i].z) << 16)
                         | (static_cast<std::uint32_t>(hInput[i].w) << 24);
        }

        // Independent host reference.
        auto hTmp = onHost::allocHost<std::uint32_t>(numPixels);
        auto hRef = onHost::allocHost<std::uint32_t>(numPixels);
        boxfilterBench::boxFilterGold(
            onHost::data(hInput),
            width,
            height,
            radius,
            onHost::data(hTmp),
            onHost::data(hRef));

        auto dInput = onHost::alloc<std::uint32_t>(device, numPixels);
        auto dTmp = onHost::alloc<std::uint32_t>(device, numPixels);
        auto dOutput = onHost::alloc<std::uint32_t>(device, numPixels);

        auto hOutput = onHost::allocHost<std::uint32_t>(numPixels);

        auto queue = device.makeQueue(queueKind::blocking);
        onHost::memcpy(queue, dInput, hPacked);
        onHost::wait(queue);

        if(!params.verifyOnly)
        {
            // warm-up
            runOnce(queue, exec, dInput, dTmp, dOutput, width, height, radius);
            onHost::wait(queue);

            std::vector<double> samples;
            samples.reserve(static_cast<std::size_t>(params.samples));
            for(int s = 0; s < params.samples; ++s)
            {
                onHost::wait(queue);
                auto const start = std::chrono::steady_clock::now();
                for(int r = 0; r < params.repeat; ++r)
                    runOnce(queue, exec, dInput, dTmp, dOutput, width, height, radius);
                onHost::wait(queue);
                auto const end = std::chrono::steady_clock::now();
                double const ns
                    = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
                samples.push_back(ns * 1e-3 / static_cast<double>(params.repeat));
            }

            Stats const stats = summarize(samples);
            std::printf(
                "RESULT backend=%s size=%ux%u radius=%d median_us=%.4f min_us=%.4f max_us=%.4f relspread=%.4f\n",
                backend.c_str(),
                width,
                height,
                radius,
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
        runOnce(queue, exec, dInput, dTmp, dOutput, width, height, radius);
        onHost::memcpy(queue, hOutput, dOutput, numPixels);
        onHost::wait(queue);

        int rc = boxfilterBench::checkFinite(
            onHost::data(hInput),
            onHost::data(hOutput),
            width,
            height,
            radius,
            "output");
        if(rc != EXIT_SUCCESS)
            return rc;

        rc = boxfilterBench::verify(
            onHost::data(hOutput),
            onHost::data(hRef),
            width,
            height,
            radius,
            "boxfilter");
        std::printf("boxfilter verification %s\n", rc == EXIT_SUCCESS ? "PASSED" : "FAILED");
        return rc;
    }
} // namespace boxfilterAlpaka

auto main(int argc, char* argv[]) -> int
{
    boxfilterBench::Params params;
    if(int const ret = boxfilterBench::parseCmd(argc, argv, params))
        return ret;

    using namespace alpaka;

    return onHost::executeForEachIfHasDevice(
        [&](auto const& backend)
        { return boxfilterAlpaka::example(onHost::makeDeviceSpec(backend), getExecutor(backend), params); },
        onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors));
}
