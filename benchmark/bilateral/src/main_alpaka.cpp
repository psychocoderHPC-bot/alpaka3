/* Copyright 2026 René Widera
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** @file Portable, single-source alpaka implementation of the HeCBench `bilateral` benchmark.
 *
 * The pinned HeCBench reference (`src/bilateral-cuda/main.cu`, with the OpenMP twin
 * `src/bilateral-omp/main.cpp`) applies a bilateral image filter to a single-channel float image for
 * window radii 3, 6 and 9. For every output pixel `(idx, idy)` it accumulates a normalised weighted
 * average over the `(2R+1)^2` window:
 *
 *   range      = -(I - I_w)^2 / (2 * variance_I)
 *   spatial    = -((idk - idx)^2 + (idl - idy)^2) / (2 * variance_spatial)
 *   weight     = a_square * expf(spatial + range)
 *   out        = sum(I_w * weight) / sum(weight),  a_square = 0.5f / (variance_I * pi)
 *
 * The window is gathered with the reference's "mirror edges" index folding (see `helper.hpp`), which
 * is only well defined for images at least as large as the window and only for `R <= min(w,h)-1`.
 * The benchmark enforces those preconditions.
 *
 * This proc keeps the reference algorithm exactly: same float accumulation order, same `expf`, same
 * folding, same comparison. One portable kernel source drives the host CPU backend and every device
 * backend; only the launch configuration changes. There is no native CUDA/HIP kernel and no
 * backend-specific algorithm branch.
 *
 * The input is reproduced from `srand(123); rand() % 256` via an independent glibc-compatible RNG
 * (`GlibcRandom` in `helper.hpp`), so the exact pinned reference binary is reproducible outside the
 * repository. Validation compares the accelerator output against an independent double-precision
 * host reference with the pinned harness's own `1e-3` tolerance, after NaN/Inf and zero-normalisation
 * checks.
 *
 * The reference sources are kept outside this repository and are not vendored.
 */

#include "helper.hpp"
#include "misc.hpp"

#include <alpaka/alpaka.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace bilateralAlpaka
{
    using namespace bilateralBench;

    //! Logical elements (pixels) per frame used to describe the parallelism.
    constexpr std::uint32_t frameExtent = 256u;

    /** Bilateral filter of a single window radius.
     *
     * Each thread handles one output pixel: it gathers the `(2R+1)^2` window with the reference's
     * mirror-edge folding and accumulates the normalised weighted average in single precision, exactly
     * as the reference device and host code do. The radius is a compile-time parameter, as in the
     * pinned reference (`bilateralFilter<R>`), so the constant-trip window loops can be fully
     * unrolled. The arithmetic and traversal order are unchanged.
     */
    template<int R>
    struct BilateralKernel
    {
        ALPAKA_FN_ACC void operator()(
            auto const& acc,
            alpaka::concepts::IMdSpan auto input,
            alpaka::concepts::IMdSpan auto output,
            int w,
            int h,
            float a_square,
            float variance_I,
            float variance_spatial) const
        {
            using namespace alpaka;

            for(auto flatIdx : onAcc::makeIdxMap(
                    acc,
                    onAcc::worker::threadsInGrid,
                    IdxRange{Vec{static_cast<std::uint32_t>(w) * static_cast<std::uint32_t>(h)}}))
            {
                std::uint32_t const flat = flatIdx.x();
                int const idx = static_cast<int>(flat % static_cast<std::uint32_t>(w));
                int const idy = static_cast<int>(flat / static_cast<std::uint32_t>(w));

                int const id = idy * w + idx;
                float const I = input[static_cast<std::uint32_t>(id)];
                float res = 0.f;
                float normalization = 0.f;

                for(int i = -R; i <= R; ++i)
                {
                    for(int j = -R; j <= R; ++j)
                    {
                        int idk = 0;
                        int idl = 0;
                        mirrorIndex(idk, idl, idx, idy, i, j, w, h);
                        int const id_w = idl * w + idk;
                        float const I_w = input[static_cast<std::uint32_t>(id_w)];

                        float const range = -(I - I_w) * (I - I_w) / (2.f * variance_I);
                        float const dx = static_cast<float>(idk - idx);
                        float const dy = static_cast<float>(idl - idy);
                        float const spatial = -(dx * dx + dy * dy) / (2.f * variance_spatial);
                        float const weight = a_square * alpaka::math::exp(spatial + range);

                        normalization += weight;
                        res += I_w * weight;
                    }
                }
                output[static_cast<std::uint32_t>(id)] = res / normalization;
            }
        }
    };

    /** Same filter with a runtime radius, for radii other than the pinned {3, 6, 9}.
     *
     * The traversal order, folding and arithmetic are identical to `BilateralKernel`; only the loop
     * bound is a runtime value, so the loops are not unrolled. This exists solely so `--radius` accepts
     * any legal value for smaller sanity checks.
     */
    struct BilateralKernelRuntime
    {
        ALPAKA_FN_ACC void operator()(
            auto const& acc,
            alpaka::concepts::IMdSpan auto input,
            alpaka::concepts::IMdSpan auto output,
            int w,
            int h,
            float a_square,
            float variance_I,
            float variance_spatial,
            int radius) const
        {
            using namespace alpaka;

            for(auto flatIdx : onAcc::makeIdxMap(
                    acc,
                    onAcc::worker::threadsInGrid,
                    IdxRange{Vec{static_cast<std::uint32_t>(w) * static_cast<std::uint32_t>(h)}}))
            {
                std::uint32_t const flat = flatIdx.x();
                int const idx = static_cast<int>(flat % static_cast<std::uint32_t>(w));
                int const idy = static_cast<int>(flat / static_cast<std::uint32_t>(w));

                int const id = idy * w + idx;
                float const I = input[static_cast<std::uint32_t>(id)];
                float res = 0.f;
                float normalization = 0.f;

                for(int i = -radius; i <= radius; ++i)
                {
                    for(int j = -radius; j <= radius; ++j)
                    {
                        int idk = 0;
                        int idl = 0;
                        mirrorIndex(idk, idl, idx, idy, i, j, w, h);
                        int const id_w = idl * w + idk;
                        float const I_w = input[static_cast<std::uint32_t>(id_w)];

                        float const range = -(I - I_w) * (I - I_w) / (2.f * variance_I);
                        float const dx = static_cast<float>(idk - idx);
                        float const dy = static_cast<float>(idl - idy);
                        float const spatial = -(dx * dx + dy * dy) / (2.f * variance_spatial);
                        float const weight = a_square * alpaka::math::exp(spatial + range);

                        normalization += weight;
                        res += I_w * weight;
                    }
                }
                output[static_cast<std::uint32_t>(id)] = res / normalization;
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

    //! Enqueue `repeat` filter passes for one radius; the caller synchronises.
    template<typename T_Queue, typename T_Exec>
    void runRadius(
        T_Queue& queue,
        T_Exec const& exec,
        auto& dInput,
        auto& dOutput,
        int w,
        int h,
        float a_square,
        float variance_I,
        float variance_spatial,
        int radius,
        int repeat)
    {
        using namespace alpaka;

        std::uint32_t const numPixels = static_cast<std::uint32_t>(w) * static_cast<std::uint32_t>(h);
        auto const frameSpec = onHost::FrameSpec{divExZero(numPixels, frameExtent), frameExtent, exec};

        auto enqueueN = [&](auto kernelBundle)
        {
            for(int r = 0; r < repeat; ++r)
                queue.enqueue(frameSpec, kernelBundle);
        };

        // Dispatch on the pinned radii so the constant-trip window loops can be unrolled. The
        // algorithm is identical for every branch; only the compile-time R differs.
        switch(radius)
        {
        case 3:
            enqueueN(
                KernelBundle{BilateralKernel<3>{}, dInput, dOutput, w, h, a_square, variance_I, variance_spatial});
            break;
        case 6:
            enqueueN(
                KernelBundle{BilateralKernel<6>{}, dInput, dOutput, w, h, a_square, variance_I, variance_spatial});
            break;
        case 9:
            enqueueN(
                KernelBundle{BilateralKernel<9>{}, dInput, dOutput, w, h, a_square, variance_I, variance_spatial});
            break;
        default:
            // Fallback for arbitrary radii: the loop bounds are runtime values, the arithmetic is the
            // same, only the unrolling differs.
            for(int r = 0; r < repeat; ++r)
                queue.enqueue(
                    frameSpec,
                    KernelBundle{
                        BilateralKernelRuntime{},
                        dInput,
                        dOutput,
                        w,
                        h,
                        a_square,
                        variance_I,
                        variance_spatial,
                        radius});
            break;
        }
    }

    //! Run one radius: warm-up, timed samples, then a fresh validated run.
    int runOneRadius(
        auto& queue,
        auto const& exec,
        auto& device,
        auto& dInput,
        float const* hInput,
        std::uint32_t width,
        std::uint32_t height,
        float varianceI,
        float varianceSpatial,
        int radius,
        bilateralBench::Params const& params,
        char const* backend)
    {
        using namespace alpaka;

        int const w = static_cast<int>(width);
        int const h = static_cast<int>(height);
        std::size_t const numPixels = static_cast<std::size_t>(width) * height;
        float const a_square = 0.5f / (varianceI * static_cast<float>(M_PI));

        // Independent double-precision host reference for this radius.
        auto hRef = onHost::allocHost<float>(numPixels);
        switch(radius)
        {
        case 3:
            if(bilateralBench::bilateralReference<
                   3>(hInput, onHost::data(hRef), w, h, a_square, varianceI, varianceSpatial)
               != EXIT_SUCCESS)
                return EXIT_FAILURE;
            break;
        case 6:
            if(bilateralBench::bilateralReference<
                   6>(hInput, onHost::data(hRef), w, h, a_square, varianceI, varianceSpatial)
               != EXIT_SUCCESS)
                return EXIT_FAILURE;
            break;
        case 9:
            if(bilateralBench::bilateralReference<
                   9>(hInput, onHost::data(hRef), w, h, a_square, varianceI, varianceSpatial)
               != EXIT_SUCCESS)
                return EXIT_FAILURE;
            break;
        default:
            if(bilateralBench::bilateralReferenceRuntime(
                   hInput,
                   onHost::data(hRef),
                   w,
                   h,
                   a_square,
                   varianceI,
                   varianceSpatial,
                   radius)
               != EXIT_SUCCESS)
                return EXIT_FAILURE;
            break;
        }

        auto dOutput = onHost::alloc<float>(device, numPixels);
        auto hOutput = onHost::allocHost<float>(numPixels);

        if(!params.verifyOnly)
        {
            // warm-up
            runRadius(queue, exec, dInput, dOutput, w, h, a_square, varianceI, varianceSpatial, radius, 1);
            onHost::wait(queue);

            std::vector<double> samples;
            samples.reserve(static_cast<std::size_t>(params.samples));
            for(int s = 0; s < params.samples; ++s)
            {
                onHost::wait(queue);
                auto const start = std::chrono::steady_clock::now();
                runRadius(
                    queue,
                    exec,
                    dInput,
                    dOutput,
                    w,
                    h,
                    a_square,
                    varianceI,
                    varianceSpatial,
                    radius,
                    params.repeat);
                onHost::wait(queue);
                auto const end = std::chrono::steady_clock::now();
                double const ns
                    = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
                samples.push_back(ns * 1e-3 / static_cast<double>(params.repeat));
            }

            Stats const stats = summarize(samples);
            std::printf(
                "RESULT backend=%s size=%dx%d radius=%d median_us=%.4f min_us=%.4f max_us=%.4f relspread=%.4f\n",
                backend,
                w,
                h,
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

        // Validation: one fresh run, copied back, compared against the double host reference.
        runRadius(queue, exec, dInput, dOutput, w, h, a_square, varianceI, varianceSpatial, radius, 1);
        onHost::memcpy(queue, hOutput, dOutput, numPixels);
        onHost::wait(queue);

        char label[64];
        std::snprintf(label, sizeof(label), "radius %d", radius);
        if(bilateralBench::checkFinite(onHost::data(hOutput), numPixels, label) != EXIT_SUCCESS)
            return EXIT_FAILURE;
        if(bilateralBench::verify(onHost::data(hOutput), onHost::data(hRef), numPixels, 1e-3, label) != EXIT_SUCCESS)
            return EXIT_FAILURE;
        return EXIT_SUCCESS;
    }

    auto example(auto const& deviceSpec, auto const& exec, bilateralBench::Params const& params) -> int
    {
        using namespace alpaka;

        auto devSelector = onHost::makeDeviceSelector(deviceSpec);
        if(!devSelector.isAvailable())
            return EXIT_SUCCESS;
        onHost::Device device = devSelector.makeDevice(0);
        std::string const backend = deviceSpec.getApi().getName();

        std::uint32_t const width = static_cast<std::uint32_t>(params.width);
        std::uint32_t const height = static_cast<std::uint32_t>(params.height);
        std::size_t const numPixels = static_cast<std::size_t>(width) * height;

        if(params.radius >= 0)
        {
            if(params.radius > params.width - 1 || params.radius > params.height - 1)
            {
                std::printf(
                    "error: radius %d requires width and height >= %d (got %ux%u)\n",
                    params.radius,
                    params.radius + 1,
                    width,
                    height);
                return EXIT_FAILURE;
            }
        }

        std::printf(
            "Using alpaka backend %s on %s, width=%u, height=%u, intensity=%g, spatial=%g, repeat=%d, samples=%d\n",
            backend.c_str(),
            deviceSpec.getDeviceKind().getName().c_str(),
            width,
            height,
            static_cast<double>(params.varianceI),
            static_cast<double>(params.varianceSpatial),
            params.repeat,
            params.samples);

        // Input exactly as the reference: srand(123); rand() % 256.
        auto hInput = onHost::allocHost<float>(numPixels);
        bilateralBench::generateInput(onHost::data(hInput), numPixels);

        auto dInput = onHost::alloc<float>(device, numPixels);
        auto queue = device.makeQueue(queueKind::blocking);
        onHost::memcpy(queue, dInput, hInput);
        onHost::wait(queue);

        std::vector<int> radii;
        if(params.radius >= 0)
            radii.push_back(params.radius);
        else
            for(int r : bilateralBench::radii)
                radii.push_back(r);

        for(int radius : radii)
        {
            if(runOneRadius(
                   queue,
                   exec,
                   device,
                   dInput,
                   onHost::data(hInput),
                   width,
                   height,
                   params.varianceI,
                   params.varianceSpatial,
                   radius,
                   params,
                   backend.c_str())
               != EXIT_SUCCESS)
                return EXIT_FAILURE;
        }

        std::printf("bilateral verification PASSED\n");
        return EXIT_SUCCESS;
    }
} // namespace bilateralAlpaka

auto main(int argc, char* argv[]) -> int
{
    bilateralBench::Params params;
    if(int const ret = bilateralBench::parseCmd(argc, argv, params))
        return ret;

    using namespace alpaka;

    return onHost::executeForEachIfHasDevice(
        [&](auto const& backend)
        { return bilateralAlpaka::example(onHost::makeDeviceSpec(backend), getExecutor(backend), params); },
        onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors));
}
