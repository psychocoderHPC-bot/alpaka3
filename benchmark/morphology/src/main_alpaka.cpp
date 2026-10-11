/* Copyright 2026 René Widera
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** @file Portable, single-source alpaka implementation of the HeCBench `morphology` benchmark.
 *
 * The pinned HeCBench reference (`src/morphology-cuda/main.cu` + `morphology.cu`, OpenMP twin
 * `src/morphology-omp/main.cpp` + `morphology.cpp`) applies a 2D separable morphological dilation
 * and erosion with a rectangular structuring element. The HeCBench accelerator reference (its
 * `.cu`/`.cpp` kernels, not this port) uses the van Herk/Gil-Werman (vHGW) algorithm with a
 * block-local two-way scan; it is O(1) per output element and independent of the
 * structuring-element size.
 *
 * This port keeps the reference pipeline exactly:
 *  - the same pass order: one horizontal pass then one vertical pass per operation,
 *  - dilation (`max`, border 255) and erosion (`min`, border 0) with the reference's odd/even
 *    radii and border rules,
 *  - the same comparison: an exact byte comparison of the final image after `repeat`
 *    `dilate`+`erode` pairs plus the reference's `sum(image) == 255` check.
 *
 * The portable kernel replaces the fixed-size block scan of the CUDA reference with a thread-local
 * sliding-window (monotonic deque) over a segment of the row (horizontal pass) or column (vertical
 * pass). It is O(1) per output element and depends on the structuring-element size only through the
 * segment overlap, so it stays fast for the large structuring elements in the measured workload.
 * One kernel source runs on the host CPU and on NVIDIA GPUs; there is no native CUDA/HIP kernel and
 * no backend branch.
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
#include <string>
#include <vector>

namespace morphologyAlpaka
{
    using namespace morphologyBench;

    namespace config
    {
        //! Largest structuring-element size supported by the per-thread deque.
        constexpr int maxSel = 256;
        //! Output cells handled by one work item; controls the deque overlap and launch granularity.
        constexpr int segLen = 16;
        //! Capacity of the linear (non-circular) per-thread deque.
        //!
        //! The deque holds the current window (<= maxSel entries) plus one slot for each push in
        //! the segment; `tail` is an absolute index that only ever moves forward, so the array
        //! must cover the initial window fill plus the segment's background pushes. Because the
        //! values are 8-bit, a push cannot avoid a pop for more than 256 consecutive monotone
        //! steps, so `maxSel + segLen` is a strict bound with margin for the general case.
        constexpr int dequeCapacity = maxSel + segLen;
        //! Threads per block for a device launch.
        constexpr std::uint32_t threadsPerBlock = 128u;
    } // namespace config

    /** Horizontal pass.
     *
     * A work item handles a contiguous segment of `segLen` output columns of one image row. The
     * valid output columns are `[hsize/2, width - hsize/2)`, matching the pinned reference; every
     * other column of the intermediate buffer is left at its initial zero. A monotonic deque over
     * the window `[c - hsize/2, c + R]` yields each output in amortised O(1).
     */
    struct HorizontalPass
    {
        ALPAKA_FN_ACC void operator()(
            auto const& acc,
            alpaka::concepts::IMdSpan auto img,
            alpaka::concepts::IMdSpan auto tmp,
            std::uint32_t width,
            std::uint32_t height,
            std::uint32_t hsize,
            bool dilation) const
        {
            using namespace alpaka;

            int const L = static_cast<int>(hsize) / 2;
            int const R = (hsize % 2u != 0u) ? static_cast<int>(hsize) / 2 : static_cast<int>(hsize) / 2 - 1;
            std::uint32_t const numSeg
                = (width + static_cast<std::uint32_t>(config::segLen) - 1u) / static_cast<std::uint32_t>(config::segLen);
            std::uint32_t const total = height * numSeg;

            for(auto [wi] : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{total}))
            {
                int const row = static_cast<int>(wi) / static_cast<int>(numSeg);
                int const seg = static_cast<int>(wi) % static_cast<int>(numSeg);

                int c0 = seg * config::segLen;
                int cend = std::min(seg * config::segLen + config::segLen, static_cast<int>(width) - L);
                if(c0 < L)
                    c0 = L;
                if(c0 >= cend)
                    continue;

                std::uint32_t const rowBase = static_cast<std::uint32_t>(row) * width;

                int dq[config::dequeCapacity];
                int head = 0;
                int tail = 0;
                auto valueAt = [&](int i) { return img[rowBase + static_cast<std::uint32_t>(i)]; };
                auto popBackWhile = [&](std::uint8_t candidate)
                {
                    while(tail > head)
                    {
                        std::uint8_t const back = valueAt(dq[tail - 1]);
                        bool const dominated = dilation ? (candidate >= back) : (candidate <= back);
                        if(dominated)
                            --tail;
                        else
                            break;
                    }
                };

                for(int i = c0 - L; i <= c0 + R; ++i)
                {
                    std::uint8_t const value = valueAt(i);
                    popBackWhile(value);
                    dq[tail++] = i;
                }
                tmp[rowBase + static_cast<std::uint32_t>(c0)] = valueAt(dq[head]);

                for(int c = c0 + 1; c < cend; ++c)
                {
                    int const add = c + R;
                    std::uint8_t const value = valueAt(add);
                    popBackWhile(value);
                    dq[tail++] = add;
                    int const remove = c - L - 1;
                    while(head < tail && dq[head] <= remove)
                        ++head;
                    tmp[rowBase + static_cast<std::uint32_t>(c)] = valueAt(dq[head]);
                }
            }
        }
    };

    /** Vertical pass.
     *
     * A work item handles a contiguous segment of `segLen` output rows of one image column. Rows
     * outside `[vsize/2, height - vsize/2)` are the operation's border value (0 for erosion, 255
     * for dilation); the interior uses a monotonic deque over `[r - vsize/2, r + Rv]`. This matches
     * the pinned reference, which writes every column of those border rows.
     */
    struct VerticalPass
    {
        ALPAKA_FN_ACC void operator()(
            auto const& acc,
            alpaka::concepts::IMdSpan auto tmp,
            alpaka::concepts::IMdSpan auto img,
            std::uint32_t width,
            std::uint32_t height,
            std::uint32_t vsize,
            bool dilation) const
        {
            using namespace alpaka;

            int const Lv = static_cast<int>(vsize) / 2;
            int const Rv = (vsize % 2u != 0u) ? static_cast<int>(vsize) / 2 : static_cast<int>(vsize) / 2 - 1;
            std::uint32_t const numSeg
                = (height + static_cast<std::uint32_t>(config::segLen) - 1u) / static_cast<std::uint32_t>(config::segLen);
            std::uint32_t const total = numSeg * width;
            std::uint8_t const border = dilation ? white : black;

            for(auto [wi] : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{total}))
            {
                int const seg = static_cast<int>(wi) / static_cast<int>(width);
                int const col = static_cast<int>(wi) % static_cast<int>(width);

                int const chunkBegin = seg * config::segLen;
                int const chunkEnd = std::min(seg * config::segLen + config::segLen, static_cast<int>(height));

                // border rows owned by this chunk
                for(int r = chunkBegin; r < chunkEnd; ++r)
                {
                    if(r < Lv || r >= static_cast<int>(height) - static_cast<int>(vsize) / 2)
                        img[static_cast<std::uint32_t>(r) * width + static_cast<std::uint32_t>(col)] = border;
                }

                int r0 = chunkBegin;
                int rend = std::min(seg * config::segLen + config::segLen, static_cast<int>(height) - Lv);
                if(r0 < Lv)
                    r0 = Lv;
                if(r0 >= rend)
                    continue;

                int dq[config::dequeCapacity];
                int head = 0;
                int tail = 0;
                auto valueAt = [&](int r)
                { return tmp[static_cast<std::uint32_t>(r) * width + static_cast<std::uint32_t>(col)]; };
                auto popBackWhile = [&](std::uint8_t candidate)
                {
                    while(tail > head)
                    {
                        std::uint8_t const back = valueAt(dq[tail - 1]);
                        bool const dominated = dilation ? (candidate >= back) : (candidate <= back);
                        if(dominated)
                            --tail;
                        else
                            break;
                    }
                };

                for(int i = r0 - Lv; i <= r0 + Rv; ++i)
                {
                    std::uint8_t const value = valueAt(i);
                    popBackWhile(value);
                    dq[tail++] = i;
                }
                img[static_cast<std::uint32_t>(r0) * width + static_cast<std::uint32_t>(col)] = valueAt(dq[head]);

                for(int r = r0 + 1; r < rend; ++r)
                {
                    int const add = r + Rv;
                    std::uint8_t const value = valueAt(add);
                    popBackWhile(value);
                    dq[tail++] = add;
                    int const remove = r - Lv - 1;
                    while(head < tail && dq[head] <= remove)
                        ++head;
                    img[static_cast<std::uint32_t>(r) * width + static_cast<std::uint32_t>(col)] = valueAt(dq[head]);
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

    /** Enqueue one operation (one horizontal + one vertical pass) for the given mode. */
    inline void runOperation(
        auto& queue,
        auto const& exec,
        std::uint32_t width,
        std::uint32_t height,
        std::uint32_t hsize,
        std::uint32_t vsize,
        bool dilation,
        auto& img,
        auto& tmp)
    {
        using namespace alpaka;

        std::uint32_t const hSeg
            = (width + static_cast<std::uint32_t>(config::segLen) - 1u) / static_cast<std::uint32_t>(config::segLen);
        std::uint32_t const hTotal = height * hSeg;
        std::uint32_t const hFrames = (hTotal + config::threadsPerBlock - 1u) / config::threadsPerBlock;
        auto const hFrameSpec = onHost::FrameSpec{Vec{hFrames}, Vec{config::threadsPerBlock}, exec};

        std::uint32_t const vSeg
            = (height + static_cast<std::uint32_t>(config::segLen) - 1u) / static_cast<std::uint32_t>(config::segLen);
        std::uint32_t const vTotal = vSeg * width;
        std::uint32_t const vFrames = (vTotal + config::threadsPerBlock - 1u) / config::threadsPerBlock;
        auto const vFrameSpec = onHost::FrameSpec{Vec{vFrames}, Vec{config::threadsPerBlock}, exec};

        onHost::memset(queue, tmp, 0x00);
        queue.enqueue(hFrameSpec, KernelBundle{HorizontalPass{}, img, tmp, width, height, hsize, dilation});
        queue.enqueue(vFrameSpec, KernelBundle{VerticalPass{}, tmp, img, width, height, vsize, dilation});
    }

    /** Run and validate the benchmark for one device and executor. */
    auto example(auto const& deviceSpec, auto const& exec, morphologyBench::Params const& params) -> int
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
        std::uint32_t const hsize = static_cast<std::uint32_t>(params.hsize);
        std::uint32_t const vsize = static_cast<std::uint32_t>(params.vsize);
        std::size_t const count = static_cast<std::size_t>(width) * height;

        std::printf(
            "Using alpaka backend %s on %s, width=%u height=%u hsize=%u vsize=%u repeat=%d samples=%d seed=%u\n",
            backend.c_str(),
            deviceSpec.getDeviceKind().getName().c_str(),
            width,
            height,
            hsize,
            vsize,
            params.repeat,
            params.samples,
            params.seed);

        // Deterministic input image, generated exactly like the reference `main`.
        auto hImage = onHost::allocHost<std::uint8_t>(count);
        morphologyBench::generateImage(onHost::data(hImage), params.width, params.height, params.seed);

        // Trusted host reference: the full `repeat` dilate+erode pipeline.
        std::vector<std::uint8_t> expected(onHost::data(hImage), onHost::data(hImage) + count);
        morphologyBench::fullReference(
            expected.data(),
            params.width,
            params.height,
            params.hsize,
            params.vsize,
            params.repeat);

        auto dImg = onHost::alloc<std::uint8_t>(device, count);
        auto dTmp = onHost::alloc<std::uint8_t>(device, count);
        onHost::memcpy(queue, dImg, hImage);
        onHost::wait(queue);

        if(!params.verifyOnly)
        {
            // warm-up
            runOperation(queue, exec, width, height, hsize, vsize, true, dImg, dTmp);
            runOperation(queue, exec, width, height, hsize, vsize, false, dImg, dTmp);
            onHost::wait(queue);

            std::vector<double> samples;
            samples.reserve(static_cast<std::size_t>(params.samples));
            for(int s = 0; s < params.samples; ++s)
            {
                onHost::wait(queue);
                auto const start = std::chrono::steady_clock::now();
                for(int r = 0; r < params.repeat; ++r)
                {
                    runOperation(queue, exec, width, height, hsize, vsize, true, dImg, dTmp);
                    runOperation(queue, exec, width, height, hsize, vsize, false, dImg, dTmp);
                }
                onHost::wait(queue);
                auto const end = std::chrono::steady_clock::now();
                // One measured sample runs `repeat` (dilate+erode) pairs; report the average time
                // of one morphological operation so the number is directly comparable to the
                // reference's per-operation `Average kernel execution time`.
                double const us
                    = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count())
                      * 1e-3 / (2.0 * static_cast<double>(params.repeat));
                samples.push_back(us);
            }
            Stats const stats = summarize(samples);
            std::printf(
                "RESULT backend=%s size=%ux%u hsize=%u vsize=%u median_us=%.4f min_us=%.4f max_us=%.4f "
                "relspread=%.4f\n",
                backend.c_str(),
                width,
                height,
                hsize,
                vsize,
                stats.median,
                stats.min,
                stats.max,
                stats.spread);
            std::printf("  raw_samples_us=");
            for(std::size_t i = 0; i < samples.size(); ++i)
                std::printf("%s%.4f", i == 0 ? "" : ",", samples[i]);
            std::printf("\n");
        }

        // Validation: reset the input, run the full pipeline once, then compare exactly.
        onHost::memcpy(queue, dImg, hImage);
        onHost::wait(queue);
        for(int n = 0; n < params.repeat; ++n)
        {
            runOperation(queue, exec, width, height, hsize, vsize, true, dImg, dTmp);
            runOperation(queue, exec, width, height, hsize, vsize, false, dImg, dTmp);
        }
        auto hResult = onHost::allocHost<std::uint8_t>(count);
        onHost::memcpy(queue, hResult, dImg);
        onHost::wait(queue);

        int const rc = morphologyBench::verify(onHost::data(hResult), expected.data(), count, "full pipeline");
        if(rc != EXIT_SUCCESS)
        {
            std::printf("morphology verification FAILED\n");
            return rc;
        }

        // Reproduce the reference's own report line `sum(image) == 255 -> PASS/FAIL`.
        //
        // The pinned reference's `main` prints PASS iff the byte sum is exactly 255, but it always
        // returns 0. For some small image/sel combinations the reference itself prints FAIL (the
        // single white pixel is removed by the erode pass), so this line is informational and does
        // not change the exit code. The authoritative check is the exact byte comparison above.
        std::uint64_t sum = 0u;
        for(std::size_t i = 0; i < count; ++i)
            sum += onHost::data(hResult)[i];
        std::printf(
            "image byte-sum=%llu (reference report line: %s)\n",
            static_cast<unsigned long long>(sum),
            sum == static_cast<std::uint64_t>(white) ? "PASS" : "FAIL");

        // The algorithm is binary only for the reference's black/white image; for the random image
        // (seed != 0) every output byte is compared exactly against the host reference already.
        if(params.seed == 0u)
        {
            for(std::size_t i = 0; i < count; ++i)
                if(onHost::data(hResult)[i] != black && onHost::data(hResult)[i] != white)
                {
                    std::fprintf(stderr, "morphology verification FAILED: non-binary intensity detected\n");
                    return EXIT_FAILURE;
                }
        }

        std::printf("morphology %ux%u hsize=%u vsize=%u verification PASSED\n", width, height, hsize, vsize);
        return EXIT_SUCCESS;
    }
} // namespace morphologyAlpaka

auto main(int argc, char* argv[]) -> int
{
    morphologyBench::Params params;
    if(int const ret = morphologyBench::parseCmd(argc, argv, params))
        return ret;

    using namespace alpaka;

    return onHost::executeForEachIfHasDevice(
        [&](auto const& backend)
        { return morphologyAlpaka::example(onHost::makeDeviceSpec(backend), getExecutor(backend), params); },
        onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors));
}
