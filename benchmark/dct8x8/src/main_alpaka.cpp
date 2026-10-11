/* Copyright 2026 René Widera
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** @file Portable, single-source alpaka implementation of the HeCBench `dct8x8` benchmark.
 *
 * The pinned HeCBench reference (`src/dct8x8-cuda/main.cu` + `kernels.cu` + `DCT8x8_gold.cu`, with
 * the OpenMP twin `src/dct8x8-omp/`) transforms every full 8x8 block of a float image with a
 * separable 8x8 DCT (`DCT_FORWARD`) or its inverse (`DCT_INVERSE`). The reference uses the hardcoded
 * AAN fast transform from the NVIDIA CUDA SDK sample; it is an orthonormal DCT-II applied over
 * rows then columns for the forward direction and the transpose (DCT-III) for the inverse.
 *
 * This port keeps the reference transform definition, scaling, block layout and comparison exactly:
 *  - the same hardcoded 8-point (i)DCT and constants (`C_a`..`C_norm`) as `DCT8x8.h`,
 *  - the same pass order (rows then columns) and the same `BLOCK_SIZE = 8` full-block rule
 *    (`i + 8 - 1 < imageH`, `j + 8 - 1 < imageW`): cells outside a full block are copied from the
 *    input,
 *  - the same relative-L2 comparison (`< 1e-6`) against an independent double-precision host
 *    reference, with additional NaN/Inf rejection.
 *
 * One portable kernel source drives the host CPU backend and the CUDA backend; there is no native
 * CUDA/HIP kernel and no backend branch selecting a different algorithm. Each thread block owns one
 * 8x8 block: the 64 threads load the block into shared memory, apply the row pass and the column
 * pass, and write the result back.
 *
 * The HeCBench reference sources are kept outside this repository and are not vendored.
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
#include <vector>

namespace dct8x8Alpaka
{
    using namespace dct8x8Bench;

    //! Constants of the hardcoded 8-point (i)DCT, reproduced verbatim from the reference `DCT8x8.h`.
    constexpr float C_a = 1.3870398453221474618216191915664f; // sqrt(2)*cos(1*pi/16)
    constexpr float C_b = 1.3065629648763765278566431734272f; // sqrt(2)*cos(2*pi/16)
    constexpr float C_c = 1.1758756024193587169744671046113f; // sqrt(2)*cos(3*pi/16)
    constexpr float C_d = 0.78569495838710218127789736765722f; // sqrt(2)*cos(5*pi/16)
    constexpr float C_e = 0.54119610014619698439972320536639f; // sqrt(2)*cos(6*pi/16)
    constexpr float C_f = 0.27589937928294301233595756366937f; // sqrt(2)*cos(7*pi/16)
    constexpr float C_norm = 0.35355339059327376220042218105242f; // 1/sqrt(8)

    //! Forward 8-point DCT (identical to `DCT8` in the reference `kernels.cu`).
    ALPAKA_FN_ACC inline void dct8(float* D)
    {
        float X07P = D[0] + D[7];
        float X16P = D[1] + D[6];
        float X25P = D[2] + D[5];
        float X34P = D[3] + D[4];

        float X07M = D[0] - D[7];
        float X61M = D[6] - D[1];
        float X25M = D[2] - D[5];
        float X43M = D[4] - D[3];

        float X07P34PP = X07P + X34P;
        float X07P34PM = X07P - X34P;
        float X16P25PP = X16P + X25P;
        float X16P25PM = X16P - X25P;

        D[0] = C_norm * (X07P34PP + X16P25PP);
        D[2] = C_norm * (C_b * X07P34PM + C_e * X16P25PM);
        D[4] = C_norm * (X07P34PP - X16P25PP);
        D[6] = C_norm * (C_e * X07P34PM - C_b * X16P25PM);

        D[1] = C_norm * (C_a * X07M - C_c * X61M + C_d * X25M - C_f * X43M);
        D[3] = C_norm * (C_c * X07M + C_f * X61M - C_a * X25M + C_d * X43M);
        D[5] = C_norm * (C_d * X07M + C_a * X61M + C_f * X25M - C_c * X43M);
        D[7] = C_norm * (C_f * X07M + C_d * X61M + C_c * X25M + C_a * X43M);
    }

    //! Inverse 8-point DCT (identical to `IDCT8` in the reference `kernels.cu`).
    ALPAKA_FN_ACC inline void idct8(float* D)
    {
        float Y04P = D[0] + D[4];
        float Y2b6eP = C_b * D[2] + C_e * D[6];

        float Y04P2b6ePP = Y04P + Y2b6eP;
        float Y04P2b6ePM = Y04P - Y2b6eP;
        float Y7f1aP3c5dPP = C_f * D[7] + C_a * D[1] + C_c * D[3] + C_d * D[5];
        float Y7a1fM3d5cMP = C_a * D[7] - C_f * D[1] + C_d * D[3] - C_c * D[5];

        float Y04M = D[0] - D[4];
        float Y2e6bM = C_e * D[2] - C_b * D[6];

        float Y04M2e6bMP = Y04M + Y2e6bM;
        float Y04M2e6bMM = Y04M - Y2e6bM;
        float Y1c7dM3f5aPM = C_c * D[1] - C_d * D[7] - C_f * D[3] - C_a * D[5];
        float Y1d7cP3a5fMM = C_d * D[1] + C_c * D[7] - C_a * D[3] + C_f * D[5];

        D[0] = C_norm * (Y04P2b6ePP + Y7f1aP3c5dPP);
        D[7] = C_norm * (Y04P2b6ePP - Y7f1aP3c5dPP);
        D[4] = C_norm * (Y04P2b6ePM + Y7a1fM3d5cMP);
        D[3] = C_norm * (Y04P2b6ePM - Y7a1fM3d5cMP);

        D[1] = C_norm * (Y04M2e6bMP + Y1c7dM3f5aPM);
        D[5] = C_norm * (Y04M2e6bMM - Y1d7cP3a5fMM);
        D[2] = C_norm * (Y04M2e6bMM + Y1d7cP3a5fMM);
        D[6] = C_norm * (Y04M2e6bMP - Y1c7dM3f5aPM);
    }

    /** Transforms every full 8x8 block of the image.
     *
     * One work item owns one output 8x8 block. It loads the block once into registers, applies the
     * row pass then the column pass with the reference's hardcoded 8-point (i)DCT, and stores the
     * block back. The row-then-column pass order and the per-element mapping are identical to
     * `DCT8x8CPU` in the reference.
     *
     * The work item distribution uses `onAcc::worker::threadsInGrid` and an index map so the same
     * kernel is correct on every backend: on the host CPU the executor fans the work items out over
     * OpenMP threads, on CUDA it maps them to device threads. There is no shared memory, no barrier
     * and no backend branch. Blocks whose lower-right edge would exceed the image are skipped
     * (`x0 + 8 - 1 >= width` or `y0 + 8 - 1 >= height`), exactly like the reference; every cell of
     * such a remainder keeps the value copied from the input.
     */
    template<bool forward>
    struct Dct8x8Kernel
    {
        template<typename T_Acc>
        ALPAKA_FN_ACC void operator()(
            T_Acc const& acc,
            alpaka::concepts::IMdSpan auto data,
            std::uint32_t width,
            std::uint32_t height,
            std::uint32_t blocksX) const
        {
            using namespace alpaka;

            std::uint32_t const numBlocks = blocksX * ((height + 7u) / 8u);

            for(auto [wi] : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{numBlocks}))
            {
                std::uint32_t const b = static_cast<std::uint32_t>(wi);
                std::uint32_t const bx = b % blocksX;
                std::uint32_t const by = b / blocksX;
                std::uint32_t const x0 = bx * 8u;
                std::uint32_t const y0 = by * 8u;

                // Only full 8x8 blocks are transformed, exactly like the reference.
                if(x0 + 7u >= width || y0 + 7u >= height)
                    continue;

                // Load the block once and keep it in registers so neither pass reads the image
                // again; neighbouring work items own horizontally adjacent blocks, so a warp reads a
                // contiguous span of the row.
                float tile[64];
                for(std::uint32_t r = 0; r < 8u; ++r)
                    for(std::uint32_t c = 0; c < 8u; ++c)
                        tile[r * 8u + c] = data[(y0 + r) * width + (x0 + c)];

                // Row pass, in registers.
                for(std::uint32_t r = 0; r < 8u; ++r)
                {
                    float D[8];
                    for(std::uint32_t n = 0; n < 8u; ++n)
                        D[n] = tile[r * 8u + n];
                    if constexpr(forward)
                        dct8(D);
                    else
                        idct8(D);
                    for(std::uint32_t n = 0; n < 8u; ++n)
                        tile[r * 8u + n] = D[n];
                }

                // Column pass, in registers.
                for(std::uint32_t c = 0; c < 8u; ++c)
                {
                    float D[8];
                    for(std::uint32_t n = 0; n < 8u; ++n)
                        D[n] = tile[n * 8u + c];
                    if constexpr(forward)
                        dct8(D);
                    else
                        idct8(D);
                    for(std::uint32_t n = 0; n < 8u; ++n)
                        tile[n * 8u + c] = D[n];
                }

                for(std::uint32_t r = 0; r < 8u; ++r)
                    for(std::uint32_t c = 0; c < 8u; ++c)
                        data[(y0 + r) * width + (x0 + c)] = tile[r * 8u + c];
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

    /** Enqueue the transform once over the whole image. */
    inline void runTransform(
        auto& queue,
        auto const& exec,
        std::uint32_t width,
        std::uint32_t height,
        bool forward,
        auto& data)
    {
        using namespace alpaka;

        std::uint32_t const blocksX = (width + 7u) / 8u;
        std::uint32_t const blocksY = (height + 7u) / 8u;
        std::uint32_t const numWorkItems = blocksX * blocksY;

        constexpr std::uint32_t threadsPerBlock = 256u;
        std::uint32_t const frames = (numWorkItems + threadsPerBlock - 1u) / threadsPerBlock;
        auto const frameSpec = onHost::FrameSpec{Vec{frames}, Vec{threadsPerBlock}, exec};
        if(forward)
            queue.enqueue(frameSpec, KernelBundle{Dct8x8Kernel<true>{}, data, width, height, blocksX});
        else
            queue.enqueue(frameSpec, KernelBundle{Dct8x8Kernel<false>{}, data, width, height, blocksX});
    }

    /** Run and validate the benchmark for one device and executor. */
    auto example(auto const& deviceSpec, auto const& exec, dct8x8Bench::Params const& params) -> int
    {
        using namespace alpaka;

        auto devSelector = onHost::makeDeviceSelector(deviceSpec);
        if(!devSelector.isAvailable())
            return EXIT_SUCCESS;
        onHost::Device device = devSelector.makeDevice(0);
        onHost::Queue queue = device.makeQueue(queueKind::nonBlocking);
        std::string const backend = deviceSpec.getApi().getName();

        std::uint32_t const width = static_cast<std::uint32_t>(params.width);
        std::uint32_t const height = static_cast<std::uint32_t>(params.height);
        std::size_t const count = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
        bool const forward = !params.inverse;
        char const* const dirName = forward ? "forward" : "inverse";

        std::printf(
            "Using alpaka backend %s on %s, width=%u height=%u repeat=%d samples=%d seed=%u dir=%s\n",
            backend.c_str(),
            deviceSpec.getDeviceKind().getName().c_str(),
            width,
            height,
            params.repeat,
            params.samples,
            params.seed,
            dirName);

        // Deterministic input, generated exactly like the reference `main`.
        auto hInput = onHost::allocHost<float>(count);
        dct8x8Bench::generateInput(onHost::data(hInput), width, height, params.seed);

        // Trusted double-precision host reference.
        std::vector<double> expected;
        dct8x8Bench::referenceTransform(onHost::data(hInput), expected, width, height, forward ? 1 : -1);

        auto dInput = onHost::alloc<float>(device, count);
        auto dOutput = onHost::alloc<float>(device, count);
        onHost::memcpy(queue, dInput, hInput);
        onHost::memcpy(queue, dOutput, hInput);
        onHost::wait(queue);

        if(!params.verifyOnly)
        {
            // warm-up
            runTransform(queue, exec, width, height, forward, dOutput);
            onHost::wait(queue);

            std::vector<double> samples;
            samples.reserve(static_cast<std::size_t>(params.samples));
            for(int s = 0; s < params.samples; ++s)
            {
                onHost::wait(queue);
                auto const start = std::chrono::steady_clock::now();
                for(int r = 0; r < params.repeat; ++r)
                    runTransform(queue, exec, width, height, forward, dOutput);
                onHost::wait(queue);
                auto const end = std::chrono::steady_clock::now();
                // One measured sample runs `repeat` transforms end-to-end; report the per-transform
                // time so it is directly comparable to the reference's `Average ... execution time`.
                double const us
                    = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count())
                      * 1e-3 / static_cast<double>(params.repeat);
                samples.push_back(us);
            }
            Stats const stats = summarize(samples);
            std::printf(
                "RESULT backend=%s size=%ux%u dir=%s median_us=%.4f min_us=%.4f max_us=%.4f relspread=%.4f\n",
                backend.c_str(),
                width,
                height,
                dirName,
                stats.median,
                stats.min,
                stats.max,
                stats.spread);
            std::printf("  raw_samples_us=");
            for(std::size_t i = 0; i < samples.size(); ++i)
                std::printf("%s%.4f", i == 0 ? "" : ",", samples[i]);
            std::printf("\n");
        }

        // Validation: run the transform once into dOutput and compare to the double reference.
        onHost::memcpy(queue, dOutput, hInput);
        onHost::wait(queue);
        runTransform(queue, exec, width, height, forward, dOutput);
        auto hResult = onHost::allocHost<float>(count);
        onHost::memcpy(queue, hResult, dOutput);
        onHost::wait(queue);

        std::printf("Comparing against host double-precision reference...\n");
        int const rc
            = dct8x8Bench::verify(onHost::data(hResult), expected, count, dct8x8Bench::defaultTolerance, dirName);
        if(rc != EXIT_SUCCESS)
        {
            std::printf("dct8x8 verification FAILED\n");
            return rc;
        }
        std::printf("dct8x8 %ux%u dir=%s verification PASSED\n", width, height, dirName);
        return EXIT_SUCCESS;
    }
} // namespace dct8x8Alpaka

auto main(int argc, char* argv[]) -> int
{
    dct8x8Bench::Params params;
    if(int const ret = dct8x8Bench::parseCmd(argc, argv, params))
        return ret;

    using namespace alpaka;

    return onHost::executeForEachIfHasDevice(
        [&](auto const& backend)
        { return dct8x8Alpaka::example(onHost::makeDeviceSpec(backend), getExecutor(backend), params); },
        onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors));
}
