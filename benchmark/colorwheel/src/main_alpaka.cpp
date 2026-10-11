/* Copyright 2026 René Widera
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** @file Portable, single-source alpaka implementation of the HeCBench `colorwheel` benchmark.
 *
 * The pinned HeCBench reference (`src/colorwheel-cuda/main.cu` and its OpenMP twin
 * `src/colorwheel-omp/main.cpp`) draws the colour encoding of a 2D vector field (Daniel Scharstein,
 * 2007, `colorcode.cpp`). Every pixel holds a flow vector
 *
 *     fx = x/half_size * range - range,   fy = y/half_size * range - range
 *
 * with `range = 1.04*truerange` and `half_size = size/2`; the pixel on either coordinate axis
 * (`x == half_size` or `y == half_size`) stays black. The vector direction selects a hue along a
 * 55-entry colour ramp; its magnitude controls saturation inside the unit disc and scales the
 * brightness by 0.75 outside it. Each channel is a lossy `int(255*col)` truncation.
 *
 * This port keeps exactly that mapping in single precision, as the reference does, and a single
 * kernel source runs on the host CPU and on NVIDIA GPUs. The colour-ramp table is built once per
 * block into shared memory and reused by every pixel the block encodes; the table values are the
 * reference integers divided by 255 in single precision, so the arithmetic is identical to the
 * reference's local, per-thread table. Only the launch configuration changes per backend, never the
 * algorithm.
 *
 * Validation compares the accelerator image against the independent double-precision host reference
 * in `helper.hpp` with the reference's own absolute per-channel tolerance of 1. See the README for
 * the tolerance rationale. The reference sources are kept outside this repository and are not
 * vendored.
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

namespace colorwheelAlpaka
{
    /** Encode a flow field into an RGB colour-wheel image.
     *
     * One thread encodes one pixel. The colour ramp is staged in shared memory by the block before the
     * pixel loop; `table[3*k + c] = cw[k][c]/255` in single precision, exactly the values the
     * reference computes locally. The block then processes `blockSize` consecutive pixels with the
     * reference's flat `y*size + x` ordering. Axis pixels (`x == half_size` or `y == half_size`) are
     * not written and keep their pre-set black value.
     */
    struct ColorKernel
    {
        ALPAKA_FN_ACC void operator()(
            auto const& acc,
            alpaka::concepts::IMdSpan auto pix,
            int size,
            int halfSize,
            float range,
            float trueRange) const
        {
            using namespace alpaka;

            constexpr uint32_t blockSize = 256u;
            //! Shared colour ramp: MAXCOLS entries, each three single-precision channels.
            auto& table = declareSharedVar<float[3u * colorwheel::MAXCOLS], uniqueId()>(acc);

            // Build the ramp once per block. The six segments and their order are identical to the
            // reference `setcols` sequences; every value is the reference integer divided by 255.
            for(auto i : onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{Vec{colorwheel::MAXCOLS}}))
            {
                int const idx = static_cast<int>(i.x());
                int r = 0, g = 0, b = 0;
                if(idx < colorwheel::RY)
                {
                    r = 255;
                    g = 255 * idx / colorwheel::RY;
                }
                else if(idx < colorwheel::RY + colorwheel::YG)
                {
                    int const j = idx - colorwheel::RY;
                    r = 255 - 255 * j / colorwheel::YG;
                    g = 255;
                }
                else if(idx < colorwheel::RY + colorwheel::YG + colorwheel::GC)
                {
                    int const j = idx - colorwheel::RY - colorwheel::YG;
                    g = 255;
                    b = 255 * j / colorwheel::GC;
                }
                else if(idx < colorwheel::RY + colorwheel::YG + colorwheel::GC + colorwheel::CB)
                {
                    int const j = idx - colorwheel::RY - colorwheel::YG - colorwheel::GC;
                    g = 255 - 255 * j / colorwheel::CB;
                    b = 255;
                }
                else if(idx < colorwheel::RY + colorwheel::YG + colorwheel::GC + colorwheel::CB + colorwheel::BM)
                {
                    int const j = idx - colorwheel::RY - colorwheel::YG - colorwheel::GC - colorwheel::CB;
                    r = 255 * j / colorwheel::BM;
                    b = 255;
                }
                else
                {
                    int const j = idx - colorwheel::RY - colorwheel::YG - colorwheel::GC - colorwheel::CB
                                  - colorwheel::BM;
                    r = 255;
                    b = 255 - 255 * j / colorwheel::MR;
                }
                table[3 * idx + 0] = static_cast<float>(r) / 255.0f;
                table[3 * idx + 1] = static_cast<float>(g) / 255.0f;
                table[3 * idx + 2] = static_cast<float>(b) / 255.0f;
            }

            onAcc::syncBlockThreads(acc);

            int const total = size * size;
            uint32_t const numBlocks
                = (static_cast<uint32_t>(total) + blockSize - 1u) / blockSize;

            for(auto blockIdx : onAcc::makeIdxMap(acc, onAcc::worker::blocksInGrid, IdxRange{Vec{numBlocks}}))
            {
                for(auto localId :
                    onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{Vec{blockSize}}))
                {
                    uint32_t const gid = blockIdx.x() * blockSize + localId.x();
                    if(gid >= static_cast<uint32_t>(total))
                        continue;

                    int const y = static_cast<int>(gid) / size;
                    int const x = static_cast<int>(gid) % size;
                    if(x == halfSize || y == halfSize)
                        continue; // black coordinate axes

                    float const fx = static_cast<float>(x) / static_cast<float>(halfSize) * range - range;
                    float const fy = static_cast<float>(y) / static_cast<float>(halfSize) * range - range;
                    float const nfx = fx / trueRange;
                    float const nfy = fy / trueRange;

                    float const rad = math::sqrt(nfx * nfx + nfy * nfy);
                    float const a = math::atan2(-nfy, -nfx) / static_cast<float>(colorwheel::kPi);
                    float const fk = (a + 1.0f) / 2.0f * static_cast<float>(colorwheel::MAXCOLS - 1);
                    int const k0 = static_cast<int>(fk);
                    int const k1 = (k0 + 1) % colorwheel::MAXCOLS;
                    float const f = fk - static_cast<float>(k0);

                    size_t const pixIdx = static_cast<size_t>(gid) * 3u;
                    for(int c = 0; c < 3; c++)
                    {
                        float const col0 = table[3 * k0 + c];
                        float const col1 = table[3 * k1 + c];
                        float col = (1.0f - f) * col0 + f * col1;
                        if(rad <= 1.0f)
                            col = 1.0f - rad * (1.0f - col); // increase saturation with radius
                        else
                            col *= 0.75f; // out of range
                        pix[pixIdx + static_cast<size_t>(2 - c)] = static_cast<uint8_t>(static_cast<int>(255.0f * col));
                    }
                }
                onAcc::syncBlockThreads(acc);
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

    /** Enqueue `repeat` colour-wheel launches on the device image. */
    template<typename T_Queue, typename T_Exec>
    void runBatch(
        T_Queue& queue,
        T_Exec const& exec,
        auto& dPix,
        int size,
        int halfSize,
        float range,
        float trueRange,
        int repeat)
    {
        using namespace alpaka;

        constexpr uint32_t blockSize = 256u;
        uint32_t const total = static_cast<uint32_t>(size) * static_cast<uint32_t>(size);
        uint32_t const numBlocks = (total + blockSize - 1u) / blockSize;
        auto const launch = onHost::FrameSpec{Vec{numBlocks}, Vec{blockSize}, exec};
        KernelBundle const kernel{ColorKernel{}, dPix, size, halfSize, range, trueRange};

        for(int r = 0; r < repeat; ++r)
            queue.enqueue(launch, kernel);
    }

    auto example(auto const& deviceSpec, auto const& exec, colorwheel::Params const& params) -> int
    {
        using namespace alpaka;

        auto devSelector = onHost::makeDeviceSelector(deviceSpec);
        if(!devSelector.isAvailable())
            return EXIT_SUCCESS;
        onHost::Device device = devSelector.makeDevice(0);
        std::string const backend = deviceSpec.getApi().getName();

        int const size = params.size;
        int const halfSize = size / 2;
        float const trueRange = params.trueRange;
        float const range = 1.04f * trueRange; // slightly bigger to show out-of-range coding
        std::size_t const imgSize = static_cast<std::size_t>(size) * size * 3u;

        std::printf(
            "Using alpaka backend %s executor %s on %s, range=%g, size=%d, repeat=%d, samples=%d\n",
            backend.c_str(),
            onHost::demangledName(exec).c_str(),
            deviceSpec.getDeviceKind().getName().c_str(),
            static_cast<double>(trueRange),
            size,
            params.repeat,
            params.samples);

        // Independent double-precision host reference.
        std::vector<unsigned char> hRef;
        if(!colorwheel::referenceImage(size, halfSize, range, trueRange, hRef))
            return EXIT_FAILURE;

        // Device image, pre-set to zero so the skipped axis pixels stay black (reference cudaMemset).
        auto dPix = onHost::alloc<uint8_t>(device, imgSize);
        std::vector<unsigned char> hZero(imgSize, 0u);
        auto queue = device.makeQueue(queueKind::blocking);
        onHost::memcpy(queue, dPix, hZero);
        onHost::wait(queue);

        if(!params.verifyOnly)
        {
            // warm-up
            runBatch(queue, exec, dPix, size, halfSize, range, trueRange, 1);
            onHost::wait(queue);

            std::vector<double> samples;
            samples.reserve(static_cast<std::size_t>(params.samples));
            for(int s = 0; s < params.samples; ++s)
            {
                onHost::wait(queue);
                auto const start = std::chrono::steady_clock::now();
                runBatch(queue, exec, dPix, size, halfSize, range, trueRange, params.repeat);
                onHost::wait(queue);
                auto const end = std::chrono::steady_clock::now();
                double const ns
                    = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
                samples.push_back(ns * 1e-3 / static_cast<double>(params.repeat));
            }

            Stats const stats = summarize(samples);
            std::printf(
                "RESULT backend=%s range=%g size=%d median_us=%.4f min_us=%.4f max_us=%.4f relspread=%.4f\n",
                backend.c_str(),
                static_cast<double>(trueRange),
                size,
                stats.median,
                stats.min,
                stats.max,
                stats.spread);
            std::printf("  raw_samples_us=");
            for(std::size_t i = 0; i < samples.size(); ++i)
                std::printf("%s%.4f", i == 0 ? "" : ",", samples[i]);
            std::printf("\n");
        }

        // Validation: one fresh run, copied back, checked.
        runBatch(queue, exec, dPix, size, halfSize, range, trueRange, 1);
        auto hOut = onHost::allocHost<uint8_t>(imgSize);
        onHost::memcpy(queue, hOut, dPix);
        onHost::wait(queue);

        int rc = colorwheel::checkFinite(onHost::data(hOut), size, halfSize, range, trueRange);
        if(rc != EXIT_SUCCESS)
            return rc;

        rc = colorwheel::verify(onHost::data(hOut), size, halfSize, range, trueRange);
        std::printf("colorwheel verification %s\n", rc == EXIT_SUCCESS ? "PASSED" : "FAILED");
        return rc;
    }
} // namespace colorwheelAlpaka

auto main(int argc, char* argv[]) -> int
{
    colorwheel::Params params;
    if(int const ret = colorwheel::parseCmd(argc, argv, params))
        return ret;

    using namespace alpaka;

    return onHost::executeForEachIfHasDevice(
        [&](auto const& backend)
        { return colorwheelAlpaka::example(onHost::makeDeviceSpec(backend), getExecutor(backend), params); },
        onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors));
}
