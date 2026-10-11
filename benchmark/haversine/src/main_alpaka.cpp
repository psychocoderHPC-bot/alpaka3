/* Copyright (c) 2020, NVIDIA CORPORATION.
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** @file Portable, single-source alpaka implementation of the HeCBench `haversine` benchmark.
 *
 * The pinned HeCBench reference (`src/haversine-cuda/distance.cu`, Apache-2.0, NVIDIA) computes the
 * great-circle distance between `num_cities` query locations and a small set of reference cities
 * with the haversine formula. `main.cu` reads city coordinates, replicates them once per reference
 * city, writes the reference city into the `z`/`w` slots, runs a double-precision host reference,
 * then launches one CUDA thread per pair.
 *
 * This port keeps the exact computation:
 *  - the same `double` value type and `double4`-equivalent input layout,
 *  - the same `DEGREE_TO_RADIAN`, `EARTH_RADIUS_KM`, half-angle and `asin(sqrt(...))` formula,
 *  - one work item per pair, no reduction or reordering,
 *  - a single portable kernel body that runs on the host CPU and on CUDA GPUs with no native
 *    CUDA/HIP kernel and no backend branch. Only the launch configuration is selected per backend.
 *
 * The reference CUDA target is compiled with `-use_fast_math`; this port uses the default-precision
 * libm/device math, which is at least as accurate, so the comparison against the double host
 * reference is not loosened by the port.
 *
 * The parallelism uses `onAcc::makeIdxMap` over a flat 1D `IdxRange`, the alpaka form that is
 * correct on the host CPU executors as well as on CUDA (a multidimensional `threadsInGrid` range is
 * not honoured by the host executors). The frame extent is a launch-shape hint; alpaka maps it onto
 * each backend's native thread hierarchy.
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

namespace haversineAlpaka
{
    using namespace haversineBench;

    /** Haversine great-circle distance, one pair per logical worker.
     *
     * The per-element expression is a direct transcription of the pinned `compute_haversine_distance`
     * kernel. `n` pairs are distributed over the flat logical range; `makeIdxMap` drops the workers
     * outside `[0, n)`.
     */
    struct DistanceKernel
    {
        ALPAKA_FN_ACC void operator()(
            auto const& acc,
            alpaka::concepts::IMdSpan auto points,
            alpaka::concepts::IMdSpan auto distance,
            std::uint64_t n) const
        {
            using namespace alpaka;

            if(n == 0u)
                return;

            for(auto i : onAcc::makeIdxMap(acc, onAcc::worker::linearThreadsInGrid, IdxRange{Vec{n}}))
            {
                auto const p = points[i];
                double const ay = p.x * degreeToRadian; // a_lat
                double const ax = p.y * degreeToRadian; // a_lon
                double const by = p.z * degreeToRadian; // b_lat
                double const bx = p.w * degreeToRadian; // b_lon

                // haversine formula
                double const x = (bx - ax) / 2.0;
                double const y = (by - ay) / 2.0;
                double const sinysqrd = math::sin(y) * math::sin(y);
                double const sinxsqrd = math::sin(x) * math::sin(x);
                double const scale = math::cos(ay) * math::cos(by);
                distance[i] = 2.0 * earthRadiusKm * math::asin(math::sqrt(sinysqrd + sinxsqrd * scale));
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

    auto example(auto const& deviceSpec, auto const& exec, haversineBench::Params const& params) -> int
    {
        using namespace alpaka;

        auto devSelector = onHost::makeDeviceSelector(deviceSpec);
        if(!devSelector.isAvailable())
            return EXIT_SUCCESS;
        onHost::Device device = devSelector.makeDevice(0);
        std::string const backend = deviceSpec.getApi().getName();

        // Load or generate the distinct locations.
        auto hLocations = onHost::allocHost<GeoPoint>(params.numCities);
        std::uint32_t numCities = params.numCities;
        if(!params.input.empty())
        {
            std::uint32_t const loaded
                = loadLocations(onHost::data(hLocations), params.numCities, params.input.c_str());
            if(loaded == 0u)
            {
                std::printf("error: no locations read from %s\n", params.input.c_str());
                return EXIT_FAILURE;
            }
            numCities = loaded;
        }
        else
        {
            generateLocations(onHost::data(hLocations), params.numCities, params.seed);
        }

        std::uint64_t const total = static_cast<std::uint64_t>(numCities) * numRefCities;

        std::printf(
            "Using alpaka backend %s executor %s on %s, num_cities=%u, pairs=%llu, repeat=%d, samples=%d\n",
            backend.c_str(),
            onHost::demangledName(exec).c_str(),
            deviceSpec.getDeviceKind().getName().c_str(),
            numCities,
            static_cast<unsigned long long>(total),
            params.repeat,
            params.samples);

        auto hPoints = onHost::allocHost<GeoPoint>(total);
        buildPairs(onHost::data(hPoints), onHost::data(hLocations), numCities);

        if(int rc = checkInputFinite(onHost::data(hPoints), total); rc != EXIT_SUCCESS)
            return rc;

        // Independent double-precision host reference.
        std::vector<double> hReference(static_cast<std::size_t>(total));
        hostReference(hReference.data(), onHost::data(hPoints), total);

        auto dPoints = onHost::alloc<GeoPoint>(device, total);
        auto dDistance = onHost::alloc<double>(device, total);
        auto hDistance = onHost::allocHost<double>(total);

        auto queue = device.makeQueue(queueKind::nonBlocking);
        onHost::memcpy(queue, dPoints, hPoints);
        onHost::wait(queue);

        auto enqueueOnce = [&]()
        {
            auto const fe = static_cast<std::uint64_t>(params.frame);
            auto const numFrames = alpaka::divCeil(Vec{total}, Vec{fe});
            onHost::FrameSpec const frameSpec{numFrames, Vec{fe}, exec};
            queue.enqueue(frameSpec, KernelBundle{DistanceKernel{}, dPoints, dDistance, total});
        };

        if(!params.verifyOnly)
        {
            // Warm-up.
            enqueueOnce();
            onHost::wait(queue);

            std::vector<double> samples;
            samples.reserve(static_cast<std::size_t>(params.samples));
            for(int s = 0; s < params.samples; ++s)
            {
                onHost::wait(queue);
                auto const start = std::chrono::steady_clock::now();
                for(int r = 0; r < params.repeat; ++r)
                    enqueueOnce();
                onHost::wait(queue);
                auto const end = std::chrono::steady_clock::now();
                double const ns
                    = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
                samples.push_back(ns * 1e-3 / static_cast<double>(params.repeat));
            }

            Stats const stats = summarize(samples);
            std::printf(
                "RESULT backend=%s executor=%s num_cities=%u pairs=%llu median_us=%.4f min_us=%.4f max_us=%.4f "
                "relspread=%.4f\n",
                backend.c_str(),
                onHost::demangledName(exec).c_str(),
                numCities,
                static_cast<unsigned long long>(total),
                stats.median,
                stats.min,
                stats.max,
                stats.spread);
            std::printf("  raw_samples_us=");
            for(std::size_t i = 0; i < samples.size(); ++i)
                std::printf("%s%.4f", i == 0 ? "" : ",", samples[i]);
            std::printf("\n");
        }

        // Validation: one fresh run, copied back, compared to the double reference.
        enqueueOnce();
        onHost::memcpy(queue, hDistance, dDistance);
        onHost::wait(queue);

        int rc = haversineBench::verify(total, onHost::data(hDistance), hReference.data());
        std::printf("haversine verification %s\n", rc == EXIT_SUCCESS ? "PASSED" : "FAILED");

        if(!params.dump.empty())
        {
            FILE* fp = std::fopen(params.dump.c_str(), "wb");
            if(fp != nullptr)
            {
                std::fwrite(onHost::data(hDistance), sizeof(double), static_cast<std::size_t>(total), fp);
                std::fclose(fp);
                std::printf("wrote %llu doubles to %s\n", static_cast<unsigned long long>(total), params.dump.c_str());
            }
            else
            {
                std::printf("error: could not open %s for writing\n", params.dump.c_str());
                return EXIT_FAILURE;
            }
        }
        return rc;
    }
} // namespace haversineAlpaka

auto main(int argc, char* argv[]) -> int
{
    haversineBench::Params params;
    if(int const ret = haversineBench::parseCmd(argc, argv, params))
        return ret;

    using namespace alpaka;

    return onHost::executeForEachIfHasDevice(
        [&](auto const& backend)
        { return haversineAlpaka::example(onHost::makeDeviceSpec(backend), getExecutor(backend), params); },
        onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors));
}
