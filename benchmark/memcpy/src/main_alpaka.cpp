/* Copyright 2026 René Widera
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** @file Portable alpaka port of the HeCBench `memcpy` benchmark.
 *
 * The HeCBench reference (`src/memcpy-cuda/main.cu`) allocates a raw byte buffer per size and measures
 * the bandwidth of asynchronous host-to-device and device-to-host copies. It has no device-to-device
 * mode and does not verify the payload. This port keeps the sizes, the per-size measurement scope and
 * the two copy directions, adds a device-to-device mode, and adds a byte-exact correctness check.
 *
 * There is no compute kernel: `onHost::memcpy` is the portable operation and the same source drives
 * the host CPU backend and every device backend. Only the backend selected by `onHost::memcpy`
 * differs; the algorithm and the measurement scope do not.
 */

#include <alpaka/alpaka.hpp>

#include "helper.hpp"
#include "misc.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace memcpyAlpaka
{
    using memcpyBench::CopyMode;

    /** Time @p repeat back-to-back calls of @p copyFn with one synchronisation at the end and return
     * the per-copy latency in microseconds.
     *
     * This is exactly the HeCBench reference scope: the reference enqueues `repeat` copies between
     * two steady-clock reads and divides the wall time of the whole region by `repeat`.
     */
    template<typename T_Queue, typename T_Copy>
    double timeCopies(T_Queue& queue, T_Copy const& copyFn, int repeat)
    {
        using namespace alpaka;
        onHost::wait(queue);
        auto const start = std::chrono::steady_clock::now();
        for(int j = 0; j < repeat; ++j)
            copyFn();
        onHost::wait(queue);
        auto const end = std::chrono::steady_clock::now();
        double const ns
            = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
        return ns * 1e-3 / static_cast<double>(repeat);
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

    /** Fill @p bytes at @p dst with the bit-inverse of the reference pattern.
     *
     * Because `b != (b xor 0xFF)` for every byte, poisoning the destination makes a no-op or a broken
     * transfer visible at every byte position.
     */
    inline void poisonPattern(std::uint8_t* dst, std::size_t bytes)
    {
        for(std::size_t i = 0; i < bytes; ++i)
            dst[i] = static_cast<std::uint8_t>(memcpyBench::patternByte(i) ^ 0xFFu);
    }

    int example(auto const deviceSpec, auto const exec, memcpyBench::Params const& params)
    {
        using namespace alpaka;
        static_cast<void>(exec);

        auto devSelector = onHost::makeDeviceSelector(deviceSpec);
        if(!devSelector.isAvailable())
            return EXIT_SUCCESS;
        onHost::Device device = devSelector.makeDevice(0);
        onHost::Queue queue = device.makeQueue();

        auto const devProps = device.getDeviceProperties();
        std::string const backend = deviceSpec.getApi().getName();

        // Requested set of sizes.
        std::vector<std::size_t> requested = params.sizes;
        if(requested.empty())
            requested = memcpyBench::defaultSizes();

        // Hard byte cap: the user cap, further bounded by a third of the device global memory so a
        // device-to-device test can hold two payloads.
        std::size_t cap = params.maxBytes;
        if(devProps.globalMemCapacityBytes > 0)
            cap = std::min(cap, devProps.globalMemCapacityBytes / 3u);

        std::vector<std::size_t> sizes;
        sizes.reserve(requested.size());
        for(std::size_t const s : requested)
        {
            if(s == 0)
            {
                std::cerr << "Error: a requested size is zero.\n";
                return EXIT_FAILURE;
            }
            if(s <= cap)
                sizes.push_back(s);
            else
                std::printf(
                    "skip size=%zu: exceeds the %zu byte cap (user %zu, device memory %zu)\n",
                    s,
                    cap,
                    params.maxBytes,
                    devProps.globalMemCapacityBytes);
        }
        if(sizes.empty())
        {
            std::cerr << "Error: no requested size fits within the byte cap.\n";
            return EXIT_FAILURE;
        }

        std::size_t const capacity = *std::max_element(sizes.begin(), sizes.end());

        std::cout << "Using alpaka backend " << backend << " on " << deviceSpec.getDeviceKind().getName() << " '"
                  << devProps.getName() << "'" << std::endl;
        std::cout << "Device global memory: " << devProps.globalMemCapacityBytes << " bytes" << std::endl;
        std::cout << "Allocation capacity: " << capacity << " bytes, byte cap: " << cap << " bytes" << std::endl;
        std::cout << "Sizes: " << sizes.size() << ", repeat: " << params.repeat << ", samples: " << params.samples
                  << std::endl;

        // Allocate one payload of the maximum requested size and copy only `size` bytes per test.
        auto hSrc = onHost::allocHost<std::uint8_t>(capacity);
        auto hPoison = onHost::allocHost<std::uint8_t>(capacity);
        auto hDst = onHost::allocHost<std::uint8_t>(capacity);
        auto dA = onHost::alloc<std::uint8_t>(device, capacity);
        auto dB = onHost::alloc<std::uint8_t>(device, capacity);

        memcpyBench::fillPattern(onHost::data(hSrc), capacity);
        poisonPattern(onHost::data(hPoison), capacity);
        {
            Vec<std::size_t, 1u> const ext{capacity};
            onHost::memcpy(queue, dA, hSrc, ext); // dA holds the reference pattern
            onHost::memcpy(queue, dB, hPoison, ext); // dB holds the poison pattern
            onHost::wait(queue);
        }

        std::vector<CopyMode> modes;
        if(params.mode == CopyMode::all)
            modes = {CopyMode::hostToDevice, CopyMode::deviceToHost, CopyMode::deviceToDevice};
        else
            modes = {params.mode};

        int error = EXIT_SUCCESS;
        for(std::size_t const size : sizes)
        {
            Vec<std::size_t, 1u> const slice{size};

            for(CopyMode const mode : modes)
            {
                auto enqueueCopy = [&]()
                {
                    switch(mode)
                    {
                    case CopyMode::hostToDevice:
                        onHost::memcpy(queue, dA, hSrc, slice);
                        break;
                    case CopyMode::deviceToHost:
                        onHost::memcpy(queue, hDst, dA, slice);
                        break;
                    case CopyMode::deviceToDevice:
                        onHost::memcpy(queue, dB, dA, slice);
                        break;
                    case CopyMode::all:
                        break;
                    }
                };

                if(!params.verifyOnly)
                {
                    // warm-up
                    for(int j = 0; j < params.repeat; ++j)
                        enqueueCopy();
                    onHost::wait(queue);

                    std::vector<double> samples;
                    samples.reserve(static_cast<std::size_t>(params.samples));
                    for(int s = 0; s < params.samples; ++s)
                        samples.push_back(timeCopies(queue, enqueueCopy, params.repeat));

                    Stats const stats = summarize(samples);
                    std::printf(
                        "RESULT backend=%s mode=%s size=%zu median_us=%.4f min_us=%.4f max_us=%.4f relspread=%.4f "
                        "bandwidth_GBps=%.4f\n",
                        backend.c_str(),
                        memcpyBench::modeName(mode),
                        size,
                        stats.median,
                        stats.min,
                        stats.max,
                        stats.spread,
                        (size > 0 && stats.median > 0.0) ? (static_cast<double>(size) / 1e9) / (stats.median * 1e-6)
                                                         : 0.0);
                    std::printf("  raw_samples_us=");
                    for(std::size_t i = 0; i < samples.size(); ++i)
                        std::printf("%s%.4f", i == 0 ? "" : ",", samples[i]);
                    std::printf("\n");
                }

                /* Byte-exact verification on a freshly poisoned destination.
                 *
                 * The destination is first overwritten with the inverse pattern, then one real copy is
                 * performed. If the copy were skipped, truncated, shifted, or partially applied, at
                 * least one byte of the poison pattern would remain and the comparison would fail.
                 */
                bool valid = false;
                switch(mode)
                {
                case CopyMode::hostToDevice:
                    onHost::memcpy(queue, dA, hPoison, slice);
                    onHost::memcpy(queue, dA, hSrc, slice);
                    onHost::memcpy(queue, hDst, dA, slice);
                    onHost::wait(queue);
                    valid = memcpyBench::verifyBytes(
                                onHost::data(hDst),
                                size,
                                memcpyBench::modeName(mode),
                                size,
                                backend.c_str())
                            == EXIT_SUCCESS;
                    break;
                case CopyMode::deviceToHost:
                    onHost::memcpy(queue, hDst, hPoison, slice);
                    onHost::memcpy(queue, hDst, dA, slice);
                    onHost::wait(queue);
                    valid = memcpyBench::verifyBytes(
                                onHost::data(hDst),
                                size,
                                memcpyBench::modeName(mode),
                                size,
                                backend.c_str())
                            == EXIT_SUCCESS;
                    break;
                case CopyMode::deviceToDevice:
                    onHost::memcpy(queue, dB, hPoison, slice);
                    onHost::memcpy(queue, dB, dA, slice);
                    onHost::memcpy(queue, hDst, dB, slice);
                    onHost::wait(queue);
                    valid = memcpyBench::verifyBytes(
                                onHost::data(hDst),
                                size,
                                memcpyBench::modeName(mode),
                                size,
                                backend.c_str())
                            == EXIT_SUCCESS;
                    break;
                case CopyMode::all:
                    break;
                }

                if(!valid)
                    error = EXIT_FAILURE;
            }
        }

        std::printf("Memcpy verification %s\n", error ? "FAILED" : "PASSED");
        return error;
    }
} // namespace memcpyAlpaka

auto main(int argc, char* argv[]) -> int
{
    memcpyBench::Params params;
    if(int const ret = memcpyBench::parseCmd(argc, argv, params))
        return ret;

    using namespace alpaka;

    return onHost::executeForEachIfHasDevice(
        [&](auto const& backend)
        { return memcpyAlpaka::example(onHost::makeDeviceSpec(backend), getExecutor(backend), params); },
        onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors));
}
