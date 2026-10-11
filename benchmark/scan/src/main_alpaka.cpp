/* Copyright 2026 René Widera
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** @file Portable, single-source alpaka implementation of the HeCBench `scan` benchmark.
 *
 * The pinned HeCBench reference (`src/scan-cuda/main.cu`, OpenMP twin `src/scan-omp/main.cpp`) splits
 * an input array of `nelems = ceil(n / N) * N` elements into independent chunks of `N` elements and
 * performs an *exclusive* prefix sum inside each chunk with a shared-memory Blelloch/Harris scan
 * (the bank-conflict-aware variant is `scan_bcao`). Chunks do not carry into each other: the last
 * element of every chunk is dropped, exactly like the reference's `t_out[i] = t_out[i-1] + t_in[i-1]`
 * host reference.
 *
 * This port keeps the reference semantics exactly:
 *  - the same input generator (`srand(123)`, `rand() % 5 + 1`) and element types (char, short, int, long),
 *  - the same chunk sizes (128, 256, 512, 1024, 2048),
 *  - both reference kernel variants: plain shared-memory scan and the bank-conflict-aware `scan_bcao`,
 *  - a grid-stride loop over chunks, matching the reference's `16 * multiProcessorCount` grid,
 *  - an independent sequential host exclusive scan and an exact element-wise comparison.
 *
 * One kernel source runs on the host CPU and on NVIDIA GPUs. The launch shape changes per backend
 * (threads per block are clamped by the backend), the algorithm does not. Native CUDA/HIP/OpenCL
 * kernels are not used and the reference sources remain outside this repository.
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

namespace scanAlpaka
{
    using namespace scanBench;

    //! Human-readable element type name for logging, matching the reference's element list.
    template<typename T>
    constexpr char const* typeName()
    {
        if constexpr(std::is_same_v<T, char>)
            return "char";
        else if constexpr(std::is_same_v<T, short>)
            return "short";
        else if constexpr(std::is_same_v<T, int>)
            return "int";
        else if constexpr(std::is_same_v<T, long>)
            return "long";
        else
            return "unknown";
    }

    /** Plain shared-memory exclusive scan, a direct translation of the reference `scan<T, N>`.
     *
     * One block processes one or more chunks in a grid-stride loop. Inside a chunk, `N` values are
     * staged in shared memory and reduced/scaned with the reference's up-sweep and down-sweep. The
     * last shared element is zeroed and every chunk restarts from zero, matching the reference.
     */
    template<typename T, int N>
    struct ScanKernel
    {
        ALPAKA_FN_ACC void operator()(
            auto const& acc,
            alpaka::concepts::IMdSpan auto g_odata,
            alpaka::concepts::IMdSpan auto const& g_idata,
            std::uint64_t numChunks) const
        {
            using namespace alpaka;

            auto& temp = onAcc::declareSharedVar<T[N], uniqueId()>(acc);
            constexpr int halfN = N / 2;

            for(auto bidMD : onAcc::makeIdxMap(acc, onAcc::worker::blocksInGrid, IdxRange{Vec{numChunks}}))
            {
                std::uint64_t const bid = static_cast<std::uint64_t>(bidMD.x());
                std::uint64_t const base = bid * static_cast<std::uint64_t>(N);

                // load N elements into shared memory: thread t holds indices 2t and 2t+1
                for(auto thidMD : onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{Vec{halfN}}))
                {
                    int const thid = static_cast<int>(thidMD.x());
                    temp[2 * thid] = g_idata[base + static_cast<std::uint64_t>(2 * thid)];
                    temp[2 * thid + 1] = g_idata[base + static_cast<std::uint64_t>(2 * thid + 1)];
                }

                // up-sweep / reduce
                int offset = 1;
                for(int d = N >> 1; d > 0; d >>= 1)
                {
                    onAcc::syncBlockThreads(acc);
                    for(auto thidMD : onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{Vec{d}}))
                    {
                        int const thid = static_cast<int>(thidMD.x());
                        int const ai = offset * (2 * thid + 1) - 1;
                        int const bi = offset * (2 * thid + 2) - 1;
                        temp[bi] += temp[ai];
                    }
                    offset *= 2;
                }

                // clear the last element
                for(auto thidMD : onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{Vec{1}}))
                {
                    if(static_cast<int>(thidMD.x()) == 0)
                        temp[N - 1] = static_cast<T>(0);
                }

                // down-sweep
                for(int d = 1; d < N; d *= 2)
                {
                    offset >>= 1;
                    onAcc::syncBlockThreads(acc);
                    for(auto thidMD : onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{Vec{d}}))
                    {
                        int const thid = static_cast<int>(thidMD.x());
                        int const ai = offset * (2 * thid + 1) - 1;
                        int const bi = offset * (2 * thid + 2) - 1;
                        T t = temp[ai];
                        temp[ai] = temp[bi];
                        temp[bi] += t;
                    }
                }

                // write back
                for(auto thidMD : onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{Vec{halfN}}))
                {
                    int const thid = static_cast<int>(thidMD.x());
                    g_odata[base + static_cast<std::uint64_t>(2 * thid)] = temp[2 * thid];
                    g_odata[base + static_cast<std::uint64_t>(2 * thid + 1)] = temp[2 * thid + 1];
                }
                onAcc::syncBlockThreads(acc);
            }
        }
    };

    /** Bank-conflict-aware shared-memory scan, a direct translation of the reference `scan_bcao<T, N>`.
     *
     * Shared memory is padded with `OFFSET(x) = x >> LOG_MEM_BANKS` (32 banks, as in the reference),
     * and the `2N` staging array is distributed with the same bank-conflict-aware addressing.
     */
    template<typename T, int N>
    struct ScanBcaoKernel
    {
        static constexpr int logMemBanks = 5;
        static constexpr int offsetOf(int n)
        {
            return n >> logMemBanks;
        }

        ALPAKA_FN_ACC void operator()(
            auto const& acc,
            alpaka::concepts::IMdSpan auto g_odata,
            alpaka::concepts::IMdSpan auto const& g_idata,
            std::uint64_t numChunks) const
        {
            using namespace alpaka;

            auto& temp = onAcc::declareSharedVar<T[2 * N], uniqueId()>(acc);
            constexpr int halfN = N / 2;

            for(auto bidMD : onAcc::makeIdxMap(acc, onAcc::worker::blocksInGrid, IdxRange{Vec{numChunks}}))
            {
                std::uint64_t const bid = static_cast<std::uint64_t>(bidMD.x());
                std::uint64_t const base = bid * static_cast<std::uint64_t>(N);

                // load with bank-conflict-aware addressing
                for(auto thidMD : onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{Vec{halfN}}))
                {
                    int const thid = static_cast<int>(thidMD.x());
                    int const a = thid;
                    int const b = a + halfN;
                    int const oa = offsetOf(a);
                    int const ob = offsetOf(b);
                    temp[a + oa] = g_idata[base + static_cast<std::uint64_t>(a)];
                    temp[b + ob] = g_idata[base + static_cast<std::uint64_t>(b)];
                }

                // up-sweep / reduce
                int offset = 1;
                for(int d = N >> 1; d > 0; d >>= 1)
                {
                    onAcc::syncBlockThreads(acc);
                    for(auto thidMD : onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{Vec{d}}))
                    {
                        int const thid = static_cast<int>(thidMD.x());
                        int ai = offset * (2 * thid + 1) - 1;
                        int bi = offset * (2 * thid + 2) - 1;
                        ai += offsetOf(ai);
                        bi += offsetOf(bi);
                        temp[bi] += temp[ai];
                    }
                    offset *= 2;
                }

                // clear the last element
                for(auto thidMD : onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{Vec{1}}))
                {
                    if(static_cast<int>(thidMD.x()) == 0)
                        temp[N - 1 + offsetOf(N - 1)] = static_cast<T>(0);
                }

                // down-sweep
                for(int d = 1; d < N; d *= 2)
                {
                    offset >>= 1;
                    onAcc::syncBlockThreads(acc);
                    for(auto thidMD : onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{Vec{d}}))
                    {
                        int const thid = static_cast<int>(thidMD.x());
                        int ai = offset * (2 * thid + 1) - 1;
                        int bi = offset * (2 * thid + 2) - 1;
                        ai += offsetOf(ai);
                        bi += offsetOf(bi);
                        T t = temp[ai];
                        temp[ai] = temp[bi];
                        temp[bi] += t;
                    }
                }

                // write back (a barrier is required here, as in the reference `scan_bcao`)
                onAcc::syncBlockThreads(acc);
                for(auto thidMD : onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{Vec{halfN}}))
                {
                    int const thid = static_cast<int>(thidMD.x());
                    int const a = thid;
                    int const b = a + halfN;
                    int const oa = offsetOf(a);
                    int const ob = offsetOf(b);
                    g_odata[base + static_cast<std::uint64_t>(a)] = temp[a + oa];
                    g_odata[base + static_cast<std::uint64_t>(b)] = temp[b + ob];
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

    //! Number of blocks in the grid, matching the reference `16 * multiProcessorCount`, cropped to the work.
    inline std::uint32_t gridSize(std::uint64_t numChunks, std::uint32_t multiProcessorCount)
    {
        std::uint64_t const suggested = 16ull * static_cast<std::uint64_t>(multiProcessorCount);
        std::uint64_t const usable = std::max<std::uint64_t>(1ull, std::min<std::uint64_t>(suggested, numChunks));
        return static_cast<std::uint32_t>(usable);
    }

    /** Run and validate one scan variant (device kernel selected by @p variant) for one element type.
     *
     * The device result is compared element-wise against an independent sequential host exclusive scan
     * with the exact same chunk semantics. Any mismatch prints the index and returns EXIT_FAILURE.
     */
    template<typename T, int N, typename T_Variant>
    int runVariant(
        auto const& deviceSpec,
        auto const& exec,
        char const* backend,
        char const* variantName,
        std::uint64_t n,
        scanBench::Params const& params)
    {
        using namespace alpaka;

        std::uint64_t const numChunks = (n + static_cast<std::uint64_t>(N) - 1u) / static_cast<std::uint64_t>(N);
        std::uint64_t const nelems = numChunks * static_cast<std::uint64_t>(N);

        auto devSelector = onHost::makeDeviceSelector(deviceSpec);
        if(!devSelector.isAvailable())
            return EXIT_SUCCESS;
        onHost::Device device = devSelector.makeDevice(0);
        onHost::Queue queue = device.makeQueue(queueKind::nonBlocking);

        auto hIn = onHost::allocHost<T>(nelems);
        auto hRef = onHost::allocHost<T>(nelems);
        auto hOut = onHost::allocHost<T>(nelems);

        scanBench::generateInput(onHost::data(hIn), nelems);
        scanBench::scanGold<T>(onHost::data(hIn), onHost::data(hRef), nelems, static_cast<std::uint32_t>(N), numChunks);
        if(int const rc = scanBench::checkFinite(onHost::data(hIn), nelems, "scan input"); rc != EXIT_SUCCESS)
            return rc;

        auto dIn = onHost::alloc<T>(device, nelems);
        auto dOut = onHost::alloc<T>(device, nelems);
        onHost::memcpy(queue, dIn, hIn, Vec{nelems});

        std::uint32_t const grid = gridSize(numChunks, device.getDeviceProperties().multiProcessorCount);
        auto const frame = onHost::FrameSpec{Vec{grid}, Vec{static_cast<std::uint32_t>(N / 2)}, exec};

        std::printf(
            "variant=%s backend=%s type=%-5s sizeof=%zu N=%d n=%llu chunks=%llu grid=%u\n",
            variantName,
            backend,
            typeName<T>(),
            sizeof(T),
            N,
            static_cast<unsigned long long>(n),
            static_cast<unsigned long long>(numChunks),
            grid);

        auto const kernels = [&]() { return KernelBundle{T_Variant{}, dOut, dIn, numChunks}; };

        if(!params.verifyOnly)
        {
            // warm-up
            queue.enqueue(frame, kernels());
            onHost::wait(queue);

            std::vector<double> samples;
            samples.reserve(static_cast<std::size_t>(params.samples));
            for(int s = 0; s < params.samples; ++s)
            {
                onHost::wait(queue);
                auto const start = std::chrono::steady_clock::now();
                for(int r = 0; r < params.repeat; ++r)
                    queue.enqueue(frame, kernels());
                onHost::wait(queue);
                auto const end = std::chrono::steady_clock::now();
                double const ns
                    = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
                samples.push_back(ns * 1e-3 / static_cast<double>(params.repeat));
            }

            Stats const stats = summarize(samples);
            std::printf(
                "RESULT backend=%s variant=%s type=%s sizeof=%zu N=%d chunks=%llu median_us=%.4f min_us=%.4f "
                "max_us=%.4f relspread=%.4f\n",
                backend,
                variantName,
                typeName<T>(),
                sizeof(T),
                N,
                static_cast<unsigned long long>(numChunks),
                stats.median,
                stats.min,
                stats.max,
                stats.spread);
            std::printf("  raw_samples_us=");
            for(std::size_t i = 0; i < samples.size(); ++i)
                std::printf("%s%.4f", i == 0 ? "" : ",", samples[i]);
            std::printf("\n");
        }

        // validation: fresh run, copy back, exact element-wise comparison
        queue.enqueue(frame, kernels());
        onHost::memcpy(queue, hOut, dOut);
        onHost::wait(queue);

        int const rc = scanBench::verify<T>(onHost::data(hOut), onHost::data(hRef), nelems, variantName);
        std::printf(
            "scan %s type=%s N=%d verification %s\n",
            variantName,
            typeName<T>(),
            N,
            rc == EXIT_SUCCESS ? "PASSED" : "FAILED");
        return rc;
    }

    /** Run all chunk sizes and element types selected by @p params. */
    template<int N, typename T>
    int runChunk(
        auto const& deviceSpec,
        auto const& exec,
        char const* backend,
        std::uint64_t n,
        scanBench::Params const& params)
    {
        int rc = runVariant<T, N, ScanKernel<T, N>>(deviceSpec, exec, backend, "scan", n, params);
        if(rc != EXIT_SUCCESS)
            return rc;
        return runVariant<T, N, ScanBcaoKernel<T, N>>(deviceSpec, exec, backend, "scan_bcao", n, params);
    }

    template<typename T>
    int runType(auto const& deviceSpec, auto const& exec, char const* backend, scanBench::Params const& params)
    {
        int rc = EXIT_SUCCESS;
        for(std::uint32_t const chunk : {128u, 256u, 512u, 1024u, 2048u})
        {
            if(params.chunk != 0u && params.chunk != chunk)
                continue;
            switch(chunk)
            {
            case 128u:
                rc = runChunk<128, T>(deviceSpec, exec, backend, params.numElements, params);
                break;
            case 256u:
                rc = runChunk<256, T>(deviceSpec, exec, backend, params.numElements, params);
                break;
            case 512u:
                rc = runChunk<512, T>(deviceSpec, exec, backend, params.numElements, params);
                break;
            case 1024u:
                rc = runChunk<1024, T>(deviceSpec, exec, backend, params.numElements, params);
                break;
            case 2048u:
                rc = runChunk<2048, T>(deviceSpec, exec, backend, params.numElements, params);
                break;
            }
            if(rc != EXIT_SUCCESS)
                return rc;
        }
        return rc;
    }

    auto example(auto const& deviceSpec, auto const& exec, scanBench::Params const& params) -> int
    {
        using namespace alpaka;

        auto devSelector = onHost::makeDeviceSelector(deviceSpec);
        if(!devSelector.isAvailable())
            return EXIT_SUCCESS;
        std::string const backend = deviceSpec.getApi().getName();
        std::printf(
            "Using alpaka backend %s, n=%llu chunk=%u repeat=%d samples=%d type=%s\n",
            backend.c_str(),
            static_cast<unsigned long long>(params.numElements),
            params.chunk,
            params.repeat,
            params.samples,
            params.type.c_str());

        int rc = EXIT_SUCCESS;
        if(params.type == "char" || params.type == "all")
            rc = runType<char>(deviceSpec, exec, backend.c_str(), params);
        if(rc == EXIT_SUCCESS && (params.type == "short" || params.type == "all"))
            rc = runType<short>(deviceSpec, exec, backend.c_str(), params);
        if(rc == EXIT_SUCCESS && (params.type == "int" || params.type == "all"))
            rc = runType<int>(deviceSpec, exec, backend.c_str(), params);
        if(rc == EXIT_SUCCESS && (params.type == "long" || params.type == "all"))
            rc = runType<long>(deviceSpec, exec, backend.c_str(), params);

        if(rc == EXIT_SUCCESS)
            std::printf("scan verification PASSED\n");
        return rc;
    }
} // namespace scanAlpaka

auto main(int argc, char* argv[]) -> int
{
    scanBench::Params params;
    if(int const ret = scanBench::parseCmd(argc, argv, params))
        return ret;

    using namespace alpaka;

    return onHost::executeForEachIfHasDevice(
        [&](auto const& backend)
        { return scanAlpaka::example(onHost::makeDeviceSpec(backend), getExecutor(backend), params); },
        onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors));
}
