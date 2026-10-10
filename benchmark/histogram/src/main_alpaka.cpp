/* Copyright 2026 René Widera
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** @file Portable, single-source alpaka implementation of the HeCBench `histogram` benchmark.
 *
 * The pinned HeCBench reference (`src/histogram-cuda/histogram_compare.cu` and its OpenMP twin
 * `src/histogram-omp/histogram_compare_base.cpp`) generates an image of random 8-bit RGBA pixels with a
 * Mersenne Twister, then histograms it into 256 bins per active channel and compares the device result
 * against a CPU reference with an exact integer comparison.
 *
 * This port keeps the reference pipeline exactly:
 *  - the same MT19937 seeding and the same `RandomBits()` per-channel generation,
 *  - the same bucket scheme (8-bit samples used directly; float samples `k/256` scaled by 256),
 *  - the same 256-bin, 1- and 3-channel genres,
 *  - an independent host reference recomputed from the same pixels and an exact integer comparison.
 *
 * One kernel source runs on the host CPU and on NVIDIA GPUs. The reference implements both a global
 * memory and a shared memory privatised variant; this port uses the per-block shared memory variant
 * (block-private counters, merged by a second accumulation kernel), which is the fast variant of the
 * reference. The launch configuration changes per backend, the algorithm does not.
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

namespace histogramAlpaka
{
    using namespace histogramBench;

    //! Threads per block.
    constexpr std::uint32_t threadsPerBlock = 256u;

    /** Decode a pixel into bin indices on the accelerator, matching the reference `DecodePixel`.
     *
     * @attention This duplicates `histogramBench::decodePixel` because the host reference must not pull in
     * accelerator-only attributes.
     */
    template<int NUM_BINS, int ACTIVE_CHANNELS>
    ALPAKA_FN_ACC void decodePixelAcc(Uchar1 pixel, unsigned int (&bins)[ACTIVE_CHANNELS])
    {
        bins[0] = static_cast<unsigned int>(pixel.x);
    }

    template<int NUM_BINS, int ACTIVE_CHANNELS>
    ALPAKA_FN_ACC void decodePixelAcc(Uchar4 pixel, unsigned int (&bins)[ACTIVE_CHANNELS])
    {
        unsigned char const samples[4] = {pixel.x, pixel.y, pixel.z, pixel.w};
        for(int channel = 0; channel < ACTIVE_CHANNELS; ++channel)
            bins[channel] = static_cast<unsigned int>(samples[channel]);
    }

    template<int NUM_BINS, int ACTIVE_CHANNELS>
    ALPAKA_FN_ACC void decodePixelAcc(Float4 pixel, unsigned int (&bins)[ACTIVE_CHANNELS])
    {
        float const samples[4] = {pixel.x, pixel.y, pixel.z, pixel.w};
        for(int channel = 0; channel < ACTIVE_CHANNELS; ++channel)
            bins[channel] = static_cast<unsigned int>(samples[channel] * static_cast<float>(NUM_BINS));
    }

    /** First pass: privatise the histogram in block shared memory and write per-block partials.
     *
     * This is the alpaka translation of `histogram_smem_atomics`: each block owns a shared counter array
     * with the reference bank-conflict padding (`+ CHANNEL`), every thread atomically increments it for its
     * pixels, and the block copies its counters into its slice of the global partial buffer.
     *
     * The image is partitioned by a grid-stride loop: a physical thread with block index @p blockIdx and
     * local thread index @p localThread covers global thread id `blockIdx * threadsPerBlock + localThread`
     * and then strides by `numBlocks * threadsPerBlock`. Every pixel belongs to exactly one (block, thread)
     * pair, so the partial histograms partition the input and the second pass only sums them.
     */
    template<typename T_Pixel, int T_activeChannels, int T_numBins>
    struct HistogramSmemKernel
    {
        static constexpr std::uint32_t partStride = static_cast<std::uint32_t>(T_activeChannels) * T_numBins;
        //! shared counters plus the reference `+3` padding
        static constexpr std::uint32_t sharedCount = partStride + 3u;

        ALPAKA_FN_ACC void operator()(
            auto const& acc,
            alpaka::concepts::IMdSpan auto pixels,
            alpaka::concepts::IMdSpan auto part,
            std::uint32_t numPixels,
            std::uint32_t numBlocks) const
        {
            using namespace alpaka;

            //! block-private shared counters
            auto& smem = onAcc::declareSharedVar<unsigned int[sharedCount], uniqueId()>(acc);

            for(auto blockIdxMD : onAcc::makeIdxMap(acc, onAcc::worker::blocksInGrid, IdxRange{Vec{numBlocks}}))
            {
                std::uint32_t const blockIdx = blockIdxMD.x();

                // initialise shared memory
                for(auto i : onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{Vec{sharedCount}}))
                    smem[i.x()] = 0;
                onAcc::syncBlockThreads(acc);

                // grid-stride partition of the image; a single logical global thread id per
                // (block, local thread) pair keeps the host and device executors identical.
                std::uint32_t const stride = numBlocks * threadsPerBlock;
                for(auto localThreadMD :
                    onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{Vec{threadsPerBlock}}))
                {
                    std::uint32_t const base = blockIdx * threadsPerBlock + localThreadMD.x();
                    for(std::uint32_t pixelIdx = base; pixelIdx < numPixels; pixelIdx += stride)
                    {
                        unsigned int bins[T_activeChannels];
                        decodePixelAcc<T_numBins>(pixels[pixelIdx], bins);

                        for(int channel = 0; channel < T_activeChannels; ++channel)
                            onAcc::atomicAdd(
                                acc,
                                &smem[T_numBins * channel + bins[channel] + static_cast<std::uint32_t>(channel)],
                                1u,
                                onAcc::scope::Block{});
                    }
                }
                onAcc::syncBlockThreads(acc);

                // store the block partial histogram
                auto* out = &part[blockIdx * partStride];
                for(auto i : onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{Vec{T_numBins}}))
                {
                    for(int channel = 0; channel < T_activeChannels; ++channel)
                        out[i.x() + T_numBins * channel]
                            = smem[i.x() + T_numBins * channel + static_cast<std::uint32_t>(channel)];
                }
                onAcc::syncBlockThreads(acc);
            }
        }
    };

    /** Second pass: accumulate the per-block partial histograms into the final result. */
    template<std::uint32_t T_count>
    struct HistogramAccumKernel
    {
        ALPAKA_FN_ACC void operator()(
            auto const& acc,
            alpaka::concepts::IMdSpan auto part,
            alpaka::concepts::IMdSpan auto hist,
            std::uint32_t numBlocks) const
        {
            using namespace alpaka;

            for(auto i : onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{Vec{T_count}}))
            {
                unsigned int total = 0;
                for(std::uint32_t block = 0; block < numBlocks; ++block)
                    total += part[block * T_count + i.x()];
                hist[i.x()] = total;
            }
        }
    };

    /** Number of blocks for a given problem size.
     *
     * The grid-stride loop is correct for any positive block count: every pixel belongs to exactly one
     * (block, thread) pair, so the partial histograms partition the input. A fixed grid is used, bounded
     * by the work actually available, so a tiny image does not launch empty blocks. This mirrors the
     * reference, which always launches a fixed 16x16 grid.
     */
    constexpr std::uint32_t fixedBlocks = 256u;

    inline std::uint32_t numBlocksFor(std::size_t numPixels)
    {
        std::size_t const needed = (numPixels + threadsPerBlock - 1u) / threadsPerBlock;
        return static_cast<std::uint32_t>(std::min<std::size_t>(fixedBlocks, std::max<std::size_t>(1u, needed)));
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

    /** Run both histogram passes once (validation of the pipeline, untimed). */
    template<typename T_Pixel, int T_activeChannels, int T_numBins>
    void runOnce(
        auto& queue,
        auto const& frameSpec,
        auto& dPixels,
        auto& dPart,
        auto& dHist,
        std::size_t numPixels,
        std::uint32_t numBlocks)
    {
        using namespace alpaka;
        queue.enqueue(
            frameSpec,
            KernelBundle{
                HistogramSmemKernel<T_Pixel, T_activeChannels, T_numBins>{},
                dPixels,
                dPart,
                static_cast<std::uint32_t>(numPixels),
                numBlocks});
        constexpr std::uint32_t accumExtent = static_cast<std::uint32_t>(T_activeChannels * T_numBins);
        queue.enqueue(
            onHost::FrameSpec{
                Vec{(accumExtent + threadsPerBlock - 1u) / threadsPerBlock},
                Vec{threadsPerBlock},
                frameSpec.getExecutor()},
            KernelBundle{HistogramAccumKernel<accumExtent>{}, dPart, dHist, numBlocks});
    }

    /** Run and validate one genre (one pixel type + channel count).
     *
     * The device result is compared against the host reference computed from the same host pixels with an
     * exact integer comparison (tolerance 0). A nonzero return value signals failure.
     */
    template<typename T_Pixel, int T_activeChannels, int T_numBins>
    int runGenre(
        auto const& deviceSpec,
        auto const& exec,
        char const* backend,
        char const* label,
        auto const& hPixels,
        std::size_t numPixels,
        histogramBench::Params const& params)
    {
        using namespace alpaka;

        constexpr std::uint32_t count = static_cast<std::uint32_t>(T_activeChannels * T_numBins);
        constexpr std::uint32_t partStride = count;

        auto devSelector = onHost::makeDeviceSelector(deviceSpec);
        if(!devSelector.isAvailable())
            return EXIT_SUCCESS;
        onHost::Device device = devSelector.makeDevice(0);
        onHost::Queue queue = device.makeQueue(queueKind::nonBlocking);

        // Host reference, recomputed directly from the same input pixels.
        auto hRef = onHost::allocHost<unsigned int>(count);
        histogramBench::histogramGold<T_activeChannels, T_numBins>(onHost::data(hPixels), numPixels, onHost::data(hRef));

        auto hOut = onHost::allocHost<unsigned int>(count);
        for(std::uint32_t i = 0; i < count; ++i)
            hOut[i] = 0u;

        auto dPixels = onHost::alloc<T_Pixel>(device, numPixels);
        onHost::memcpy(queue, dPixels, hPixels, Vec{numPixels});

        std::uint32_t const numBlocks = numBlocksFor(numPixels);
        auto dPart = onHost::alloc<unsigned int>(device, static_cast<std::size_t>(numBlocks) * partStride);
        auto dHist = onHost::alloc<unsigned int>(device, count);

        auto const frameSpec = onHost::FrameSpec{Vec{numBlocks}, Vec{threadsPerBlock}, exec};

        std::printf(
            "genre=%s backend=%s pixels=%zu channels=%d bins=%d blocks=%u threads=%u\n",
            label,
            backend,
            numPixels,
            T_activeChannels,
            T_numBins,
            numBlocks,
            threadsPerBlock);

        if(!params.verifyOnly)
        {
            // warm-up
            runOnce<T_Pixel, T_activeChannels, T_numBins>(
                queue, frameSpec, dPixels, dPart, dHist, numPixels, numBlocks);
            onHost::wait(queue);

            std::vector<double> samples;
            samples.reserve(static_cast<std::size_t>(params.samples));
            for(int s = 0; s < params.samples; ++s)
            {
                onHost::wait(queue);
                auto const start = std::chrono::steady_clock::now();
                for(int r = 0; r < params.repeat; ++r)
                    runOnce<T_Pixel, T_activeChannels, T_numBins>(
                        queue,
                        frameSpec,
                        dPixels,
                        dPart,
                        dHist,
                        numPixels,
                        numBlocks);
                onHost::wait(queue);
                auto const end = std::chrono::steady_clock::now();
                double const ns
                    = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
                samples.push_back(ns * 1e-3 / static_cast<double>(params.repeat));
            }

            Stats const stats = summarize(samples);
            std::printf(
                "RESULT backend=%s genre=%s size=%zu channels=%d median_us=%.4f min_us=%.4f max_us=%.4f "
                "relspread=%.4f\n",
                backend,
                label,
                numPixels,
                T_activeChannels,
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
        runOnce<T_Pixel, T_activeChannels, T_numBins>(
                queue, frameSpec, dPixels, dPart, dHist, numPixels, numBlocks);
        onHost::memcpy(queue, hOut, dHist);
        onHost::wait(queue);

        int const rc = histogramBench::verify(
            onHost::data(hOut),
            onHost::data(hRef),
            T_activeChannels,
            T_numBins,
            numPixels,
            label);
        std::printf("histogram %s verification %s\n", label, rc == EXIT_SUCCESS ? "PASSED" : "FAILED");
        return rc;
    }

    auto example(auto const& deviceSpec, auto const& exec, histogramBench::Params const& params) -> int
    {
        using namespace alpaka;

        auto devSelector = onHost::makeDeviceSelector(deviceSpec);
        if(!devSelector.isAvailable())
            return EXIT_SUCCESS;
        onHost::Device device = devSelector.makeDevice(0);
        std::string const backend = deviceSpec.getApi().getName();

        std::size_t const numPixels = params.numPixels;
        std::printf(
            "Using alpaka backend %s on %s, pixels=%zu, entropy=%d, repeat=%d, samples=%d\n",
            backend.c_str(),
            deviceSpec.getDeviceKind().getName().c_str(),
            numPixels,
            params.entropy,
            params.repeat,
            params.samples);

        // Master RGBA image, generated exactly like `GenerateRandomImage()` in the reference.
        auto hUchar4 = onHost::allocHost<Uchar4>(numPixels);
        histogramBench::generateRandomPixels(onHost::data(hUchar4), numPixels, params.entropy);

        // Genre 1: 1-channel 8-bit
        auto hUchar1 = onHost::allocHost<Uchar1>(numPixels);
        histogramBench::makeUchar1(onHost::data(hUchar1), onHost::data(hUchar4), numPixels);
        int rc = runGenre<Uchar1, 1, numBins>(
            deviceSpec,
            exec,
            backend.c_str(),
            "uchar1",
            hUchar1,
            numPixels,
            params);
        if(rc != EXIT_SUCCESS)
            return rc;

        // Genre 2: 3/4-channel 8-bit
        rc = runGenre<Uchar4, 3, numBins>(
            deviceSpec,
            exec,
            backend.c_str(),
            "uchar4",
            hUchar4,
            numPixels,
            params);
        if(rc != EXIT_SUCCESS)
            return rc;

        // Genre 3: 3/4-channel float in [0, 1)
        auto hFloat4 = onHost::allocHost<Float4>(numPixels);
        histogramBench::makeFloat4(onHost::data(hFloat4), onHost::data(hUchar4), numPixels);
        rc = histogramBench::checkFinite(onHost::data(hFloat4), numPixels, "float4 input");
        if(rc != EXIT_SUCCESS)
            return rc;
        rc = runGenre<Float4, 3, numBins>(
            deviceSpec,
            exec,
            backend.c_str(),
            "float4",
            hFloat4,
            numPixels,
            params);
        if(rc != EXIT_SUCCESS)
            return rc;

        std::printf("Histogram verification PASSED\n");
        return EXIT_SUCCESS;
    }
} // namespace histogramAlpaka

auto main(int argc, char* argv[]) -> int
{
    histogramBench::Params params;
    if(int const ret = histogramBench::parseCmd(argc, argv, params))
        return ret;

    using namespace alpaka;

    return onHost::executeForEachIfHasDevice(
        [&](auto const& backend)
        { return histogramAlpaka::example(onHost::makeDeviceSpec(backend), getExecutor(backend), params); },
        onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors));
}
