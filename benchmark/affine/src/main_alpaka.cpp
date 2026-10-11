/* Copyright 2026 René Widera
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** @file Portable, single-source alpaka implementation of the HeCBench `affine` benchmark.
 *
 * The pinned HeCBench reference (`src/affine-cuda/kernel.h`, shared by the CUDA, HIP and SYCL
 * variants; `src/affine-omp/main.cpp` carries the same arithmetic inline) performs a fixed 2D affine
 * image transform: rotate by 30 degrees, expand by 0.5 in both axes, translate by zero. Each output
 * pixel `(x, y)` maps through the inverse transform to a source coordinate `(x_new, y_new)` and is
 * filled by a bilinear sample of the four surrounding source pixels, truncated to `unsigned short`.
 * Samples exactly on the last row or column use the nearest edge pixel; everything outside the image
 * is filled with `WHITE = 1`. All arithmetic is single precision (`float`).
 *
 * This port keeps the reference algorithm exactly:
 *  - the same coefficient formula, inverse transform and branch conditions,
 *  - the same bilinear sample and `unsigned short` truncation,
 *  - the same `WHITE = 1` fill and edge handling,
 *  - a single portable kernel body that runs on the host CPU and on CUDA GPUs, with no native
 *    CUDA/HIP kernel and no backend branch. Only the launch configuration is selected per backend.
 *
 * The forward coefficients are constant for the whole image, so they are computed once on the host
 * and passed as kernel arguments. This matches the reference, whose compiler folds the constant
 * `cos`/`sin` calls; doing the trigonometry per pixel would otherwise cost a device `cos`/`sin` per
 * element. The resulting inverse matrix is numerically identical to the reference's (same single
 * precision operations), so the per-pixel transform is unchanged.
 *
 * The parallelism is expressed with `onAcc::makeIdxMap` over a 2D `IdxRange{width, height}`, the
 * canonical alpaka device-agnostic form. `makeIdxMap` maps the logical index range onto every
 * backend's native thread hierarchy, so the same source drives the host CPU and CUDA. The logical
 * frame extents (64x4) are a launch-shape hint only; alpaka is free to remap them.
 *
 * The reference compares its result against the same host reference (`affine_reference`) and reports
 * the maximum absolute error. This port instead recomputes an independent host reference
 * (`affineBench::affineReference`) plus a double-precision oracle (`affineBench::affineReferenceDouble`)
 * and compares with a justified tolerance of one grey level, with explicit NaN/Inf checks. See
 * `helper.hpp` and the README for the tolerance rationale.
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

namespace affineAlpaka
{
    using namespace affineBench;

    //! Logical 2D frame extents (launch-shape hint for the index map).
    constexpr std::uint32_t frameExtentX = 64u;
    constexpr std::uint32_t frameExtentY = 4u;

    /** Fixed affine transform kernel.
     *
     * Every output pixel `(x, y)` is produced by the inverse of the fixed forward transform and a
     * bilinear sample of the source. The inverse-matrix coefficients and the zero offset are passed
     * in, computed once on the host. The per-pixel transform, branch conditions and truncation
     * reproduce the pinned reference bit-for-bit in single precision. `width`/`height` replace the
     * reference's compile-time `X_SIZE`/`Y_SIZE`.
     */
    struct AffineKernel
    {
        ALPAKA_FN_ACC void operator()(
            auto const& acc,
            alpaka::concepts::IMdSpan auto src,
            alpaka::concepts::IMdSpan auto dst,
            std::uint32_t width,
            std::uint32_t height,
            float ik00,
            float ik01,
            float ik10,
            float ik11,
            float ib0,
            float ib1) const
        {
            using namespace alpaka;

            for(auto [x, y] : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{Vec{width, height}}))
            {
                float const fw = static_cast<float>(width);
                float const fh = static_cast<float>(height);
                float const fx = static_cast<float>(x);
                float const fy = static_cast<float>(y);

                float const xNew = ib0 + ik00 * (fx - fw / 2.0f) + ik01 * (fy - fh / 2.0f) + fw / 2.0f;
                float const yNew = ib1 + ik10 * (fx - fw / 2.0f) + ik11 * (fy - fh / 2.0f) + fh / 2.0f;

                int const m = static_cast<int>(math::floor(xNew));
                int const n = static_cast<int>(math::floor(yNew));

                float const xFrac = xNew - static_cast<float>(m);
                float const yFrac = yNew - static_cast<float>(n);

                int const iw = static_cast<int>(width);
                int const ih = static_cast<int>(height);

                unsigned short outputBuffer;
                if((m >= 0) && (m + 1 < iw) && (n >= 0) && (n + 1 < ih))
                {
                    float const grayNew
                        = (1.0f - yFrac)
                              * ((1.0f - xFrac) * static_cast<float>(src[(n * iw) + m])
                                 + xFrac * static_cast<float>(src[(n * iw) + m + 1]))
                          + yFrac
                                * ((1.0f - xFrac) * static_cast<float>(src[((n + 1) * iw) + m])
                                   + xFrac * static_cast<float>(src[((n + 1) * iw) + m + 1]));
                    outputBuffer = static_cast<unsigned short>(grayNew);
                }
                else if(
                    ((m + 1 == iw) && (n >= 0) && (n < ih))
                    || ((n + 1 == ih) && (m >= 0) && (m < iw)))
                {
                    outputBuffer = src[(n * iw) + m];
                }
                else
                {
                    outputBuffer = white;
                }

                dst[(y * width) + x] = outputBuffer;
            }
        }
    };

    //! Inverse matrix and offset of the fixed transform.
    struct Inverse
    {
        float k00, k01, k10, k11, b0, b1;
    };

    /** Inverse of the fixed forward transform, in single precision on the host.
     *
     * Forecasts the reference's per-pixel coefficient computation exactly: rotation `(30, 0)`
     * degrees, expansion `(0.5, 0.5)`, translation `(0, 0)`. The reference recomputes these
     * constants for every pixel; the compiler folds them, so hoisting them is bit-identical.
     */
    inline Inverse makeInverse()
    {
        float const lxRot = 30.0f;
        float const lyRot = 0.0f;
        float const lxExpan = 0.5f;
        float const lyExpan = 0.5f;
        float const a00 = lxExpan * std::cos(lxRot * pi / 180.0f);
        float const a01 = lyExpan * std::sin(lyRot * pi / 180.0f);
        float const a10 = lxExpan * std::sin(lxRot * pi / 180.0f);
        float const a11 = lyExpan * std::cos(lyRot * pi / 180.0f);
        float const det = a00 * a11 - a01 * a10;
        Inverse inv{};
        if(det == 0.0f)
        {
            inv = {1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f};
        }
        else
        {
            inv.k00 = a11 / det;
            inv.k01 = -a01 / det;
            inv.k10 = -a10 / det;
            inv.k11 = a00 / det;
            inv.b0 = 0.0f;
            inv.b1 = 0.0f;
        }
        return inv;
    }

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

    /** Run the fixed transform once. */
    template<typename T_Queue, typename T_Exec>
    void runOnce(
        T_Queue& queue,
        T_Exec const& exec,
        auto& dSrc,
        auto& dDst,
        std::uint32_t width,
        std::uint32_t height,
        Inverse const& inv)
    {
        using namespace alpaka;

        Vec const frameExtents{frameExtentX, frameExtentY};
        auto const frameSpec = onHost::FrameSpec{
            divExZero(Vec{width, height}, frameExtents),
            frameExtents,
            exec};

        queue.enqueue(
            frameSpec,
            KernelBundle{
                AffineKernel{},
                dSrc,
                dDst,
                width,
                height,
                inv.k00,
                inv.k01,
                inv.k10,
                inv.k11,
                inv.b0,
                inv.b1});
    }

    auto example(auto const& deviceSpec, auto const& exec, affineBench::Params const& params) -> int
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

        // Deterministic 16-bit grayscale input image.
        auto hInput = onHost::allocHost<unsigned short>(numPixels);
        affineBench::generateImage(onHost::data(hInput), numPixels, 0xC0FFEEu);

        // Single-precision host reference, and the independent double-precision oracle.
        auto hRef = onHost::allocHost<unsigned short>(numPixels);
        auto hRefDouble = onHost::allocHost<unsigned short>(numPixels);
        affineBench::affineReference(onHost::data(hInput), onHost::data(hRef), width, height);
        affineBench::affineReferenceDouble(onHost::data(hInput), onHost::data(hRefDouble), width, height);

        // The two host references must agree within one grey level; any larger gap is a reference bug.
        if(int rc = affineBench::verifyReferenceAgainstDouble(
               onHost::data(hRef),
               onHost::data(hRefDouble),
               static_cast<int>(width),
               static_cast<int>(height),
               "single vs double reference",
               1);
           rc != EXIT_SUCCESS)
            return rc;

        auto dInput = onHost::alloc<unsigned short>(device, numPixels);
        auto dOutput = onHost::alloc<unsigned short>(device, numPixels);
        auto hOutput = onHost::allocHost<unsigned short>(numPixels);

        Inverse const inv = makeInverse();

        auto queue = device.makeQueue(queueKind::nonBlocking);
        onHost::memcpy(queue, dInput, hInput);
        onHost::wait(queue);

        if(!params.verifyOnly)
        {
            // warm-up
            runOnce(queue, exec, dInput, dOutput, width, height, inv);
            onHost::wait(queue);

            std::vector<double> samples;
            samples.reserve(static_cast<std::size_t>(params.samples));
            for(int s = 0; s < params.samples; ++s)
            {
                onHost::wait(queue);
                auto const start = std::chrono::steady_clock::now();
                for(int r = 0; r < params.repeat; ++r)
                    runOnce(queue, exec, dInput, dOutput, width, height, inv);
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

        // Validation: one fresh run, copied back, checked for NaN/Inf and compared to the reference.
        runOnce(queue, exec, dInput, dOutput, width, height, inv);
        onHost::memcpy(queue, hOutput, dOutput, numPixels);
        onHost::wait(queue);

        int rc = affineBench::checkFinite(
            onHost::data(hInput),
            onHost::data(hOutput),
            static_cast<int>(width),
            static_cast<int>(height),
            "output");
        if(rc != EXIT_SUCCESS)
            return rc;

        rc = affineBench::verify(
            onHost::data(hOutput),
            onHost::data(hRef),
            static_cast<int>(width),
            static_cast<int>(height),
            "affine",
            1);
        std::printf("affine verification %s\n", rc == EXIT_SUCCESS ? "PASSED" : "FAILED");
        return rc;
    }
} // namespace affineAlpaka

auto main(int argc, char* argv[]) -> int
{
    affineBench::Params params;
    if(int const ret = affineBench::parseCmd(argc, argv, params))
        return ret;

    using namespace alpaka;

    return onHost::executeForEachIfHasDevice(
        [&](auto const& backend)
        { return affineAlpaka::example(onHost::makeDeviceSpec(backend), getExecutor(backend), params); },
        onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors));
}
