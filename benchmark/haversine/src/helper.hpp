/* Copyright (c) 2020, NVIDIA CORPORATION.
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

/** @file Host reference, input generation and verification for the HeCBench `haversine` benchmark.
 *
 * The pinned HeCBench reference (`src/haversine-cuda/main.cu`, `distance.cu`, Apache-2.0, NVIDIA)
 * computes the great-circle distance between `num_cities` query locations and a small set of
 * reference cities with the haversine formula. Each element of the device input is a `double4`
 * `(a_lat, a_lon, b_lat, b_lon)` and the kernel evaluates
 *
 *     ax = a_lon * DEG, ay = a_lat * DEG, bx = b_lon * DEG, by = b_lat * DEG
 *     x  = (bx - ax) / 2, y = (by - ay) / 2
 *     d  = 2 * EARTH_RADIUS_KM * asin( sqrt( sin(y)^2 + sin(x)^2 * cos(ay) * cos(by) ) )
 *
 * with `DEGREE_TO_RADIAN = pi/180` and `EARTH_RADIUS_KM = 6371`. The reference recomputes the same
 * expression on the host and reports the maximum absolute deviation (it does not fail on it).
 *
 * This file contains no HeCBench code. It provides:
 *  - a deterministic location generator that reproduces the reference's input layout
 *    (`num_cities` distinct points, duplicated once per reference city, with the reference city
 *    copied into the `z`/`w` slots), and an optional reader for a reference-format file so the
 *    original HeCBench input can be replayed;
 *  - an independent double-precision host evaluation of the pinned formula run in linear order;
 *  - a comparator with explicit NaN/Inf rejection and a tolerance justified by double precision.
 */

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace haversineBench
{
    //! Exactly the pinned `DEGREE_TO_RADIAN` macro from `src/haversine-cuda/distance.h`.
    constexpr double degreeToRadian = 3.14159265358979323846 / 180.0;
    //! Exactly the pinned `EARTH_RADIUS_KM` macro from `src/haversine-cuda/distance.h`.
    constexpr double earthRadiusKm = 6371.0;

    //! HeCBench default number of distinct city locations (`num_cities = 2^21`).
    constexpr std::uint32_t defaultNumCities = 2097152u;
    //! HeCBench `num_ref_cities`.
    constexpr std::uint32_t numRefCities = 6u;
    /** Reference city indices into the location table (1-based), exactly as the pinned `index_map`.
     *
     *  bombay, melbourne, waltham, moscow, glasgow, morocco.
     */
    constexpr std::uint32_t referenceCityIndex[numRefCities] = {436483u, 1952407u, 627919u, 377884u, 442703u, 1863423u};

    /** A point in the reference `double4` layout `(a_lat, a_lon, b_lat, b_lon)`.
     *
     * `aligned(16)` mirrors CUDA `double4` alignment. It is a plain aggregate so it can be copied
     * between host and device bytewise.
     */
    struct alignas(16) GeoPoint
    {
        double x;
        double y;
        double z;
        double w;
    };

    /** Evaluate the pinned haversine formula for one pair in double precision.
     *
     * This is a direct, independent transcription of the pinned expression. It uses the host
     * `std::sin`/`std::cos`/`std::asin`/`std::sqrt`; the device kernel uses the device math library,
     * which can differ by a few ULP. See @ref verify for the tolerance rationale.
     */
    inline double haversineDistance(double a_lat, double a_lon, double b_lat, double b_lon)
    {
        double const ax = a_lon * degreeToRadian;
        double const ay = a_lat * degreeToRadian;
        double const bx = b_lon * degreeToRadian;
        double const by = b_lat * degreeToRadian;

        double const x = (bx - ax) / 2.0;
        double const y = (by - ay) / 2.0;
        double const sinysqrd = std::sin(y) * std::sin(y);
        double const sinxsqrd = std::sin(x) * std::sin(x);
        double const scale = std::cos(ay) * std::cos(by);
        return 2.0 * earthRadiusKm * std::asin(std::sqrt(sinysqrd + sinxsqrd * scale));
    }

    namespace detail
    {
        /** xorshift64* step, deterministic and platform-independent. */
        inline std::uint64_t nextRandom(std::uint64_t& state)
        {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            return state * 0x2545F4914F6CDD1Dull;
        }

        inline double unit(std::uint64_t& state)
        {
            // 53-bit mantissa -> [0,1)
            return static_cast<double>(nextRandom(state) >> 11) * (1.0 / 9007199254740992.0);
        }
    } // namespace detail

    /** Generate `numCities` deterministic locations in `[-90,90] x [-180,180]`.
     *
     * The reference reads real city coordinates from an external data file. This port is
     * self-contained: it generates deterministic pseudo-random coordinates with the same structure,
     * so the benchmark can run without the 42 MB HeCBench `locations.txt`. Pass the original file
     * with `--input` to replay the pinned data (see @ref loadLocations).
     */
    inline void generateLocations(GeoPoint* points, std::uint32_t numCities, std::uint64_t seed)
    {
        std::uint64_t state = seed | 1u;
        for(std::uint32_t i = 0; i < numCities; ++i)
        {
            points[i].x = detail::unit(state) * 180.0 - 90.0;  // latitude
            points[i].y = detail::unit(state) * 360.0 - 180.0; // longitude
            points[i].z = 0.0;
            points[i].w = 0.0;
        }
    }

    /** Read up to `numCities` `lat lon` lines from a reference-format file.
     *
     * Returns the number of locations read. Mirrors the pinned `main.cu` parse loop
     * (`fscanf(fp, "%lf %lf\n", &lat, &lon)`), stopping at `numCities` entries.
     */
    inline std::uint32_t loadLocations(GeoPoint* points, std::uint32_t numCities, char const* path)
    {
        FILE* fp = std::fopen(path, "r");
        if(fp == nullptr)
        {
            std::perror("Error opening the input file");
            return 0u;
        }
        std::uint32_t city = 0;
        double lat = 0.0;
        double lon = 0.0;
        while(std::fscanf(fp, "%lf %lf\n", &lat, &lon) != EOF)
        {
            points[city].x = lat;
            points[city].y = lon;
            points[city].z = 0.0;
            points[city].w = 0.0;
            ++city;
            if(city == numCities)
                break;
        }
        std::fclose(fp);
        return city;
    }

    /** Build the full device input exactly like the pinned reference.
     *
     * The first `numCities` entries hold the distinct locations; that block is then copied
     * `numRefCities` times, and for each block `c` the reference-city coordinates are written into
     * the `z` (latitude) and `w` (longitude) slots from `index_map`. The index is clamped to
     * `numCities - 1` so the layout is well defined for small validation sizes; for the default
     * `numCities` the clamp is inactive and the construction is exact.
     *
     * @param points    output buffer of `numCities * numRefCities` elements
     * @param locations the `numCities` distinct locations (`x`/`y`; `z`/`w` ignored)
     */
    inline void buildPairs(GeoPoint* points, GeoPoint const* locations, std::uint32_t numCities)
    {
        for(std::uint32_t i = 0; i < numCities; ++i)
            points[i] = locations[i];

        for(std::uint32_t c = 1; c < numRefCities; ++c)
            std::memcpy(points + static_cast<std::size_t>(c) * numCities, points, sizeof(GeoPoint) * numCities);

        for(std::uint32_t c = 0; c < numRefCities; ++c)
        {
            std::uint32_t index = referenceCityIndex[c] - 1u;
            if(index >= numCities)
                index = numCities - 1u;
            for(std::uint32_t j = 0; j < numCities; ++j)
            {
                GeoPoint& p = points[static_cast<std::size_t>(c) * numCities + j];
                p.z = locations[index].x;
                p.w = locations[index].y;
            }
        }
    }

    /** Independent double-precision host evaluation of the pinned formula.
     *
     * Streams the input in linear order and evaluates @ref haversineDistance for every pair. This is
     * the oracle the device result is compared against.
     */
    inline void hostReference(double* expected, GeoPoint const* points, std::uint64_t total)
    {
        for(std::uint64_t i = 0; i < total; ++i)
        {
            GeoPoint const& p = points[i];
            expected[i] = haversineDistance(p.x, p.y, p.z, p.w);
        }
    }

    /** Reject non-finite input locations or finite-but-out-of-range coordinates. */
    inline int checkInputFinite(GeoPoint const* points, std::uint64_t total)
    {
        for(std::uint64_t i = 0; i < total; ++i)
        {
            GeoPoint const& p = points[i];
            if(!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z) || !std::isfinite(p.w))
            {
                std::printf("error at input[%llu]: non-finite coordinate\n", static_cast<unsigned long long>(i));
                return EXIT_FAILURE;
            }
            if(std::fabs(p.x) > 90.0000001 || std::fabs(p.y) > 180.0000001 || std::fabs(p.z) > 90.0000001
               || std::fabs(p.w) > 180.0000001)
            {
                std::printf(
                    "error at input[%llu]: coordinate out of range (%g,%g,%g,%g)\n",
                    static_cast<unsigned long long>(i),
                    p.x,
                    p.y,
                    p.z,
                    p.w);
                return EXIT_FAILURE;
            }
        }
        return EXIT_SUCCESS;
    }

    /** Compare the device result against the double host reference.
     *
     * Explicitly rejects NaN/Inf in either buffer. The tolerance combines an absolute term
     * (1e-6 km = 1 mm) and a relative term (1e-9). The device and the host both evaluate the same
     * expression in double precision; the only difference is the libm implementation of
     * `sin`/`cos`/`asin`/`sqrt`, which differs by a few ULP. For distances of order 1e4 km an ULP is
     * about 2e-12 km, so the observed deviation is far below the absolute term. The terms are not
     * widened to hide bugs: a wrong formula produces km-scale differences.
     */
    inline int verify(
        std::uint64_t total,
        double const* got,
        double const* ref,
        double atol = 1e-6,
        double rtol = 1e-9)
    {
        int error = EXIT_SUCCESS;
        unsigned reported = 0;
        double maxDiff = 0.0;
        std::uint64_t maxIndex = 0;
        for(std::uint64_t i = 0; i < total; ++i)
        {
            double const g = got[i];
            double const r = ref[i];
            if(!std::isfinite(g) || !std::isfinite(r))
            {
                std::printf(
                    "error at distance[%llu]: non-finite value got %g reference %g\n",
                    static_cast<unsigned long long>(i),
                    g,
                    r);
                return EXIT_FAILURE;
            }
            double const diff = std::fabs(g - r);
            if(diff > maxDiff)
            {
                maxDiff = diff;
                maxIndex = i;
            }
            double const tol = atol + rtol * std::fabs(r);
            if(diff > tol)
            {
                if(reported < 10u)
                    std::printf(
                        "error at distance[%llu]: got %.17g reference %.17g (|diff|=%.3g tol=%.3g)\n",
                        static_cast<unsigned long long>(i),
                        g,
                        r,
                        diff,
                        tol);
                ++reported;
                error = EXIT_FAILURE;
            }
        }
        std::printf(
            "maximum absolute error in distance is %.6e km at index %llu\n",
            maxDiff,
            static_cast<unsigned long long>(maxIndex));
        if(error != EXIT_SUCCESS)
            std::printf("haversine verification FAILED (%u element(s) outside tolerance)\n", reported);
        return error;
    }
} // namespace haversineBench
